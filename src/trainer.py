import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data import create_datasets
from src.losses import reconstruction_refitting_loss, signature_refitting_loss
from src.metrics import exposure_sparsity_metrics, refitting_metrics
from src.model import build_model_from_config
from src.simulate import sample_synthetic_batch, simulation_reference_bank_from_payload
from src.utils import ensure_dir, infer_device, save_json, seed_everything


def _gather_reference_signatures(signature_bank: torch.Tensor, pool_indices: torch.Tensor) -> torch.Tensor:
    flat_indices = pool_indices.reshape(-1)
    gathered = torch.index_select(signature_bank, 0, flat_indices)
    return gathered.reshape(pool_indices.shape[0], pool_indices.shape[1], signature_bank.shape[1])


def _resolve_demographic_conditioning_mode(config: Dict) -> str:
    value = str(config.get("model", {}).get("demographic_conditioning", "both")).strip().lower()
    if value in {"none", "off", "false", "disabled"}:
        return "none"
    if value in {"adaln", "conditioning", "condition"}:
        return "adaln"
    if value in {"tokens", "token"}:
        return "tokens"
    if value in {"both", "all", "full"}:
        return "both"
    raise ValueError(
        "Unsupported model.demographic_conditioning value '{0}'. Use 'none', 'tokens', 'adaln', or 'both'.".format(
            value
        )
    )


def _build_scheduler(optimizer: AdamW, config: Dict, steps_per_epoch: int):
    warmup_epochs = int(config["training"].get("warmup_epochs", 5))
    hold_epochs = int(config["training"].get("hold_epochs", 60))
    total_epochs = int(config["training"]["epochs"])
    min_lr_ratio = float(config["training"].get("min_lr_ratio", 0.05))
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    hold_steps = max(warmup_steps, hold_epochs * steps_per_epoch)
    total_steps = max(hold_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        step = step + 1
        if step <= warmup_steps:
            return step / warmup_steps
        if step <= hold_steps:
            return 1.0
        decay_steps = max(1, total_steps - hold_steps)
        progress = min(1.0, (step - hold_steps) / decay_steps)
        return 1.0 - (1.0 - min_lr_ratio) * progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _normalise_sample_id(sample_id, barcode_length: int) -> str:
    value = str(sample_id).strip()
    return value[:barcode_length] if barcode_length > 0 else value


def _load_sample_labels(sample_weighting_cfg: Dict, config: Dict) -> Dict[str, str]:
    metadata_path = Path(sample_weighting_cfg.get("metadata_path", "outputs/debug/matched_dataset_overview.csv"))
    if not metadata_path.exists() and config.get("config_path"):
        candidate_path = Path(config["config_path"]).parent / metadata_path
        if candidate_path.exists():
            metadata_path = candidate_path
    if not metadata_path.exists():
        raise FileNotFoundError("sample_weighting.metadata_path does not exist: {0}".format(metadata_path))

    sample_id_col = str(sample_weighting_cfg.get("sample_id_col", "sample_id"))
    label_col = str(sample_weighting_cfg.get("label_col", "cancer_type"))
    delimiter = str(sample_weighting_cfg.get("delimiter", ","))
    barcode_length = int(sample_weighting_cfg.get("barcode_length", 16))

    labels = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("sample_weighting.metadata_path has no header: {0}".format(metadata_path))
        missing_columns = [column for column in (sample_id_col, label_col) if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                "sample_weighting.metadata_path is missing columns {0}; available columns are {1}".format(
                    missing_columns,
                    reader.fieldnames,
                )
            )
        for row in reader:
            sample_key = _normalise_sample_id(row[sample_id_col], barcode_length)
            labels[sample_key] = str(row[label_col]).strip()
    return labels


def _build_train_sample_weights(config: Dict, train_dataset) -> tuple:
    sample_weighting_cfg = config["training"].get("sample_weighting", {})
    if not bool(sample_weighting_cfg.get("enabled", False)):
        return None, None

    target_label = str(sample_weighting_cfg.get("target_label", "TCGA-PAAD")).strip()
    target_weight = float(sample_weighting_cfg.get("target_weight", 5.0))
    base_weight = float(sample_weighting_cfg.get("base_weight", 1.0))
    if target_weight <= 0.0 or base_weight <= 0.0:
        raise ValueError("sample_weighting target_weight and base_weight must both be positive.")

    barcode_length = int(sample_weighting_cfg.get("barcode_length", 16))
    labels = _load_sample_labels(sample_weighting_cfg, config)
    weights = []
    missing_samples = 0
    target_samples = 0
    for row_idx in train_dataset.indices:
        sample_key = _normalise_sample_id(train_dataset.sample_ids[int(row_idx)], barcode_length)
        label = labels.get(sample_key)
        if label is None:
            missing_samples += 1
            weights.append(base_weight)
            continue
        if label == target_label:
            target_samples += 1
            weights.append(target_weight)
        else:
            weights.append(base_weight)

    if target_samples == 0:
        raise ValueError(
            "sample_weighting is enabled but no training samples matched target_label='{0}'.".format(target_label)
        )

    weight_tensor = torch.tensor(weights, dtype=torch.double)
    raw_mean_weight = float(weight_tensor.mean().item())
    if bool(sample_weighting_cfg.get("normalize_mean_to_one", True)):
        weight_tensor = weight_tensor / max(raw_mean_weight, 1e-12)

    total_samples = int(len(weights))
    metadata = {
        "enabled": True,
        "metadata_path": str(sample_weighting_cfg.get("metadata_path", "outputs/debug/matched_dataset_overview.csv")),
        "sample_id_col": str(sample_weighting_cfg.get("sample_id_col", "sample_id")),
        "label_col": str(sample_weighting_cfg.get("label_col", "cancer_type")),
        "target_label": target_label,
        "target_weight": target_weight,
        "base_weight": base_weight,
        "barcode_length": barcode_length,
        "normalize_mean_to_one": bool(sample_weighting_cfg.get("normalize_mean_to_one", True)),
        "train_samples": total_samples,
        "target_train_samples": int(target_samples),
        "non_target_train_samples": int(total_samples - target_samples),
        "missing_metadata_train_samples": int(missing_samples),
        "raw_mean_weight": raw_mean_weight,
        "effective_target_sampling_fraction": float((target_samples * target_weight) / max(sum(weights), 1e-12)),
    }
    return weight_tensor, metadata


def _build_label_filtered_dataset(config: Dict, dataset, filter_cfg_key: str = "validation_filter") -> tuple:
    filter_cfg = config["training"].get(filter_cfg_key, {})
    if not bool(filter_cfg.get("enabled", False)):
        return None, None

    target_label = str(filter_cfg.get("target_label", "TCGA-PAAD")).strip()
    barcode_length = int(filter_cfg.get("barcode_length", 16))
    labels = _load_sample_labels(filter_cfg, config)
    filtered_indices = []
    missing_samples = 0
    for row_idx in dataset.indices:
        row_idx = int(row_idx)
        sample_key = _normalise_sample_id(dataset.sample_ids[row_idx], barcode_length)
        label = labels.get(sample_key)
        if label is None:
            missing_samples += 1
            continue
        if label == target_label:
            filtered_indices.append(row_idx)

    if not filtered_indices:
        raise ValueError(
            "training.{0} is enabled but no validation samples matched target_label='{1}'.".format(
                filter_cfg_key,
                target_label,
            )
        )

    filtered_dataset = type(dataset)(
        normalized_profiles=dataset.normalized_profiles,
        count_matrix=dataset.count_matrix,
        depths=dataset.depths,
        reference_pool_indices=dataset.reference_pool_indices,
        sample_ids=dataset.sample_ids,
        indices=torch.tensor(filtered_indices, dtype=torch.long).numpy(),
        rna_embeddings=dataset.rna_embeddings,
        cnv_embeddings=dataset.cnv_embeddings,
        methylation_embeddings=dataset.methylation_embeddings,
        demographic_vectors=dataset.demographic_vectors,
        known_exposures=dataset.known_exposures,
        known_exposures_valid_mask=dataset.known_exposures_valid_mask,
    )
    metadata = {
        "enabled": True,
        "metadata_path": str(filter_cfg.get("metadata_path", "outputs/debug/matched_dataset_overview.csv")),
        "sample_id_col": str(filter_cfg.get("sample_id_col", "sample_id")),
        "label_col": str(filter_cfg.get("label_col", "cancer_type")),
        "target_label": target_label,
        "barcode_length": barcode_length,
        "validation_samples": int(len(dataset.indices)),
        "target_validation_samples": int(len(filtered_indices)),
        "non_target_validation_samples": int(len(dataset.indices) - len(filtered_indices)),
        "missing_metadata_validation_samples": int(missing_samples),
    }
    return filtered_dataset, metadata


def _sparse_objective_score(val_metrics: Dict[str, float], training_cfg: Dict) -> float:
    sparse_cfg = training_cfg.get("sparse_objective", {})
    score = float(val_metrics["loss"])

    cosine_weight = float(sparse_cfg.get("cosine_weight", 0.0))
    mse_weight = float(sparse_cfg.get("mse_weight", 0.0))
    active_weight = float(sparse_cfg.get("active_gt_0.01_weight", 0.0))
    effective_weight = float(sparse_cfg.get("effective_signatures_weight", 0.0))
    top3_weight = float(sparse_cfg.get("top3_mass_weight", 0.0))

    score += mse_weight * float(val_metrics.get("mse", val_metrics.get("reconstruction_mse", 0.0)))
    score -= cosine_weight * float(val_metrics.get("cosine_similarity", 0.0))

    active_target = float(sparse_cfg.get("active_gt_0.01_target", 0.0))
    active_value = float(val_metrics.get("active_gt_0.01", 0.0))
    score += active_weight * max(0.0, active_value - active_target)

    effective_target = float(sparse_cfg.get("effective_signatures_target", 0.0))
    effective_value = float(val_metrics.get("effective_signatures", 0.0))
    score += effective_weight * max(0.0, effective_value - effective_target)

    top3_target = float(sparse_cfg.get("top3_mass_target", 0.0))
    top3_value = float(val_metrics.get("top3_mass", 0.0))
    score += top3_weight * max(0.0, top3_target - top3_value)
    return score


def _active_recovery_counts(
    predicted_exposures: torch.Tensor,
    target_exposures: torch.Tensor,
    depths: torch.Tensor,
    training_cfg: Dict,
    predicted_active_probabilities: torch.Tensor = None,
) -> Dict[str, float]:
    threshold = float(training_cfg.get("active_metric_threshold", training_cfg.get("active_supervision_threshold", 0.01)))
    min_expected = float(
        training_cfg.get(
            "active_metric_min_expected_mutations",
            training_cfg.get("active_supervision_min_expected_mutations", 10.0),
        )
    )
    mode = str(training_cfg.get("active_metric_mode", "burden_aware")).strip().lower()
    if depths.ndim == 1:
        depths = depths.unsqueeze(1)

    score_source = str(training_cfg.get("active_metric_score_source", "exposure")).strip().lower()
    target_active = target_exposures >= threshold
    if score_source == "active_probabilities":
        if predicted_active_probabilities is None:
            raise ValueError("active_metric_score_source='active_probabilities' requires model active_probabilities output.")
        probability_threshold = float(training_cfg.get("active_metric_probability_threshold", 0.5))
        pred_active = predicted_active_probabilities >= probability_threshold
    elif score_source == "exposure":
        pred_active = predicted_exposures >= threshold
    else:
        raise ValueError("Unsupported active_metric_score_source '{0}'. Expected exposure or active_probabilities.".format(score_source))
    if mode == "burden_aware":
        target_active = target_active & ((target_exposures * depths) >= min_expected)
        if score_source == "exposure":
            pred_active = pred_active & ((predicted_exposures * depths) >= min_expected)
    elif mode != "exposure":
        raise ValueError("Unsupported active_metric_mode '{0}'. Expected exposure or burden_aware.".format(mode))

    tp = (pred_active & target_active).sum().item()
    fp = (pred_active & ~target_active).sum().item()
    fn = (~pred_active & target_active).sum().item()
    return {
        "active_tp": float(tp),
        "active_fp": float(fp),
        "active_fn": float(fn),
    }


def _active_recovery_labels_and_scores(
    predicted_exposures: torch.Tensor,
    target_exposures: torch.Tensor,
    depths: torch.Tensor,
    training_cfg: Dict,
    predicted_active_probabilities: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = float(training_cfg.get("active_metric_threshold", training_cfg.get("active_supervision_threshold", 0.01)))
    min_expected = float(
        training_cfg.get(
            "active_metric_min_expected_mutations",
            training_cfg.get("active_supervision_min_expected_mutations", 10.0),
        )
    )
    mode = str(training_cfg.get("active_metric_mode", "burden_aware")).strip().lower()
    if depths.ndim == 1:
        depths = depths.unsqueeze(1)

    target_active = target_exposures >= threshold
    if mode == "burden_aware":
        target_active = target_active & ((target_exposures * depths) >= min_expected)
    elif mode != "exposure":
        raise ValueError("Unsupported active_metric_mode '{0}'. Expected exposure or burden_aware.".format(mode))

    score_source = str(training_cfg.get("active_metric_score_source", "exposure")).strip().lower()
    if score_source == "active_probabilities":
        if predicted_active_probabilities is None:
            raise ValueError("active_metric_score_source='active_probabilities' requires model active_probabilities output.")
        scores = predicted_active_probabilities
    elif score_source == "exposure":
        if mode == "burden_aware":
            exposure_score = predicted_exposures / max(threshold, 1e-8)
            burden_score = predicted_exposures * depths / max(min_expected, 1e-8)
            scores = torch.minimum(exposure_score, burden_score)
        else:
            scores = predicted_exposures
    else:
        raise ValueError("Unsupported active_metric_score_source '{0}'. Expected exposure or active_probabilities.".format(score_source))
    return target_active.detach().reshape(-1).float().cpu(), scores.detach().reshape(-1).float().cpu()


def _average_precision_from_scores(labels: torch.Tensor, scores: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    labels = labels.float()
    scores = scores.float()
    positives = labels.sum()
    if positives.item() <= 0.0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    true_positives = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
    precision_at_rank = true_positives / ranks
    ap = (precision_at_rank * sorted_labels).sum() / positives
    return float(ap.item())


def _add_active_recovery_metrics(metrics: Dict[str, float]) -> None:
    tp = float(metrics.get("active_tp", 0.0))
    fp = float(metrics.get("active_fp", 0.0))
    fn = float(metrics.get("active_fn", 0.0))
    precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    metrics["active_precision"] = precision
    metrics["active_recall"] = recall
    metrics["active_f1"] = f1


def _get_pyplot():
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-{0}".format(os.environ.get("USER", "codex")))
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        return None


def _simulation_epoch_pass(
    model,
    simulation_bank,
    num_cosmic_signatures: int,
    device: torch.device,
    training_cfg: Dict,
    optimizer: AdamW = None,
    scheduler=None,
    epoch_index: int = 0,
    seed: int = 42,
    is_validation: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    totals = defaultdict(float)
    num_batches = int(training_cfg.get("val_batches_per_epoch", 20) if is_validation else training_cfg.get("train_batches_per_epoch", 100))
    base_seed = seed + (100000 if is_validation else 0) + epoch_index
    rng = __import__("numpy").random.default_rng(base_seed)

    for _ in range(num_batches):
        batch = sample_synthetic_batch(
            reference_bank=simulation_bank,
            num_cosmic_signatures=num_cosmic_signatures,
            batch_size=int(training_cfg["batch_size"]),
            num_channels=model.num_channels,
            rng=rng,
            device=device,
            simulation_cfg=training_cfg.get("simulation", {}),
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(batch["input_profile"], batch["reference_signatures"])
            losses = signature_refitting_loss(
                predicted_exposures=outputs["exposures"],
                target_exposures=batch["true_exposures"],
                predicted_confidence=outputs["confidence"],
                reconstruction=outputs["reconstruction"],
                target_profile=batch["target_profile"],
                confidence_scale=float(training_cfg.get("confidence_scale", 10.0)),
                composition_weight=float(training_cfg.get("composition_weight", 1.0)),
                confidence_weight=float(training_cfg.get("confidence_weight", 1.0)),
            )
            if is_train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("grad_clip_norm", 1.0)))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        metrics = refitting_metrics(outputs["reconstruction"].detach(), batch["target_profile"])
        metrics.update(exposure_sparsity_metrics(outputs["exposures"].detach()))
        for key in ["loss", "composition_mse", "confidence_mse", "reconstruction_mse"]:
            totals[key] += float(losses[key].item())
        for key, value in metrics.items():
            totals[key] += value

    if num_batches == 0:
        return {key: 0.0 for key in totals}
    return {key: value / num_batches for key, value in totals.items()}


def _real_data_epoch_pass(
    model,
    loader: DataLoader,
    signature_bank: torch.Tensor,
    device: torch.device,
    training_cfg: Dict,
    optimizer: AdamW = None,
    scheduler=None,
    grad_clip_norm: float = 1.0,
    candidate_signature_mask: torch.Tensor = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    totals = defaultdict(float)
    num_batches = 0
    active_labels = []
    active_scores = []

    for batch in loader:
        input_profile = batch["counts"].to(device)
        target_profile = batch["profile"].to(device)
        target_depths = batch["depth"].to(device)
        pool_indices = batch["reference_pool_indices"].to(device)
        rna_embedding = batch["rna_embedding"].to(device) if "rna_embedding" in batch else None
        cnv_embedding = batch["cnv_embedding"].to(device) if "cnv_embedding" in batch else None
        methylation_embedding = batch["methylation_embedding"].to(device) if "methylation_embedding" in batch else None
        demographic_vector = batch["demographic_vector"].to(device) if "demographic_vector" in batch else None
        known_exposures = batch["known_exposures"].to(device) if "known_exposures" in batch else None
        known_exposures_valid = (
            batch["known_exposures_valid"].to(device)
            if "known_exposures_valid" in batch
            else None
        )
        model_known_exposures = (
            known_exposures
            if bool(training_cfg.get("feed_known_exposures_to_model", True))
            else None
        )
        reference_signatures = _gather_reference_signatures(signature_bank, pool_indices)
        target_exposures = None
        candidate_pool_mask = None
        if known_exposures is not None:
            target_exposures = torch.gather(known_exposures, dim=1, index=pool_indices)
        if candidate_signature_mask is not None:
            candidate_pool_mask = candidate_signature_mask[pool_indices]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(
                input_profile,
                reference_signatures,
                rna_embedding=rna_embedding,
                cnv_embedding=cnv_embedding,
                methylation_embedding=methylation_embedding,
                demographic_vector=demographic_vector,
                known_exposures=model_known_exposures,
            )
            losses = reconstruction_refitting_loss(
                reconstruction=outputs["reconstruction"],
                target_profile=target_profile,
                predicted_exposures=outputs.get("exposures"),
                predicted_active_logits=outputs.get("active_logits"),
                target_exposures=target_exposures,
                target_exposure_valid_mask=known_exposures_valid,
                target_depths=target_depths,
                residual_prediction=outputs.get("residual_prediction"),
                target_residual=outputs.get("residual_input").detach() if outputs.get("residual_input") is not None else None,
                loss_name=str(training_cfg.get("loss", "kl")),
                kl_weight=float(training_cfg.get("kl_weight", 1.0)),
                mse_weight=float(training_cfg.get("mse_weight", 1.0)),
                residual_weight=float(training_cfg.get("residual_weight", 0.0)),
                cosine_weight=float(training_cfg.get("cosine_weight", 0.0)),
                exposure_entropy_weight=float(training_cfg.get("exposure_entropy_weight", 0.0)),
                exposure_topk_tail_weight=float(training_cfg.get("exposure_topk_tail_weight", 0.0)),
                exposure_topk_tail_k=int(training_cfg.get("exposure_topk_tail_k", 3)),
                exposure_soft_active_weight=float(training_cfg.get("exposure_soft_active_weight", 0.0)),
                exposure_soft_active_threshold=float(training_cfg.get("exposure_soft_active_threshold", 0.01)),
                exposure_soft_active_temperature=float(training_cfg.get("exposure_soft_active_temperature", 0.002)),
                exposure_soft_active_target=float(training_cfg.get("exposure_soft_active_target", 4.0)),
                exposure_supervision_weight=float(training_cfg.get("exposure_supervision_weight", 0.0)),
                exposure_supervision_loss=str(training_cfg.get("exposure_supervision_loss", "mse")),
                exposure_effective_supervision_weight=float(training_cfg.get("exposure_effective_supervision_weight", 0.0)),
                exposure_top3_supervision_weight=float(training_cfg.get("exposure_top3_supervision_weight", 0.0)),
                exposure_top3_supervision_k=int(training_cfg.get("exposure_top3_supervision_k", 3)),
                active_supervision_weight=float(training_cfg.get("active_supervision_weight", 0.0)),
                active_supervision_mode=str(training_cfg.get("active_supervision_mode", "exposure")),
                active_supervision_threshold=float(training_cfg.get("active_supervision_threshold", 0.01)),
                active_supervision_min_expected_mutations=float(training_cfg.get("active_supervision_min_expected_mutations", 10.0)),
                active_supervision_temperature=float(training_cfg.get("active_supervision_temperature", 0.002)),
                active_supervision_positive_weight=float(training_cfg.get("active_supervision_positive_weight", 1.0)),
                active_supervision_loss=str(training_cfg.get("active_supervision_loss", "bce")),
                active_supervision_focal_gamma=float(training_cfg.get("active_supervision_focal_gamma", 2.0)),
                active_supervision_burden_weighted=bool(training_cfg.get("active_supervision_burden_weighted", False)),
                candidate_mask=candidate_pool_mask,
                candidate_presence_weight=float(training_cfg.get("candidate_presence_weight", 0.0)),
                candidate_mass_weight=float(training_cfg.get("candidate_mass_weight", 0.0)),
                candidate_mixture_weight=float(training_cfg.get("candidate_mixture_weight", 0.0)),
                cosmic_preservation_weight=float(training_cfg.get("cosmic_preservation_weight", 0.0)),
                candidate_positive_weight=float(training_cfg.get("candidate_positive_weight", 1.0)),
                candidate_presence_threshold=float(training_cfg.get("candidate_presence_threshold", 0.01)),
                candidate_presence_temperature=float(training_cfg.get("candidate_presence_temperature", 0.01)),
            )
            if is_train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        metrics = refitting_metrics(outputs["reconstruction"].detach(), target_profile)
        if outputs.get("exposures") is not None:
            metrics.update(exposure_sparsity_metrics(outputs["exposures"].detach()))
        if target_exposures is not None and outputs.get("exposures") is not None:
            metric_predicted_exposures = outputs["exposures"].detach()
            metric_target_exposures = target_exposures.detach()
            metric_depths = target_depths.detach()
            metric_active_probabilities = (
                outputs.get("active_probabilities").detach()
                if outputs.get("active_probabilities") is not None
                else None
            )
            if known_exposures_valid is not None:
                metric_valid = known_exposures_valid.to(dtype=torch.bool).reshape(-1)
                metric_predicted_exposures = metric_predicted_exposures[metric_valid]
                metric_target_exposures = metric_target_exposures[metric_valid]
                metric_depths = metric_depths[metric_valid]
                if metric_active_probabilities is not None:
                    metric_active_probabilities = metric_active_probabilities[metric_valid]
            if metric_target_exposures.shape[0] == 0:
                metric_predicted_exposures = None
        if target_exposures is not None and outputs.get("exposures") is not None and metric_predicted_exposures is not None:
            active_counts = _active_recovery_counts(
                predicted_exposures=metric_predicted_exposures,
                target_exposures=metric_target_exposures,
                depths=metric_depths,
                training_cfg=training_cfg,
                predicted_active_probabilities=metric_active_probabilities,
            )
            for key, value in active_counts.items():
                totals[key] += value
            labels, scores = _active_recovery_labels_and_scores(
                predicted_exposures=metric_predicted_exposures,
                target_exposures=metric_target_exposures,
                depths=metric_depths,
                training_cfg=training_cfg,
                predicted_active_probabilities=metric_active_probabilities,
            )
            active_labels.append(labels)
            active_scores.append(scores)
        for key in [
            "loss",
            "composition_mse",
            "confidence_mse",
            "reconstruction_mse",
            "exposure_entropy",
            "exposure_entropy_loss",
            "exposure_topk_tail_loss",
            "exposure_soft_active_count",
            "exposure_soft_active_loss",
            "exposure_supervision_loss",
            "exposure_effective_supervision_loss",
            "exposure_top3_supervision_loss",
            "active_supervision_bce",
            "candidate_presence_bce",
            "candidate_mass_loss",
            "candidate_mixture_loss",
            "cosmic_preservation_loss",
            "candidate_target_mass_mean",
            "candidate_predicted_mass_mean",
            "candidate_predicted_mass_positive_mean",
            "candidate_predicted_mass_negative_mean",
        ]:
            totals[key] += float(losses[key].item())
        for key, value in metrics.items():
            totals[key] += value
        num_batches += 1

    if num_batches == 0:
        return {key: 0.0 for key in totals}
    averaged = {
        key: (value if key in {"active_tp", "active_fp", "active_fn"} else value / num_batches)
        for key, value in totals.items()
    }
    _add_active_recovery_metrics(averaged)
    if active_labels:
        averaged["active_average_precision"] = _average_precision_from_scores(
            torch.cat(active_labels),
            torch.cat(active_scores),
        )
    return averaged


def _save_training_plots(history, target_path: Path) -> str:
    plt = _get_pyplot()
    if plt is None:
        print("Skipping training curves because matplotlib is not installed.")
        return ""

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train"]["loss"] for entry in history]
    val_loss = [entry["val"]["loss"] for entry in history]
    train_cosine = [entry["train"]["cosine_similarity"] for entry in history]
    val_cosine = [entry["val"]["cosine_similarity"] for entry in history]

    ensure_dir(str(target_path.parent))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="Train", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Validation", linewidth=2)
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_cosine, label="Train", linewidth=2)
    axes[1].plot(epochs, val_cosine, label="Validation", linewidth=2)
    axes[1].set_title("Cosine Similarity vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(target_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(target_path)


def _save_metric_plot(history, target_path: Path, metric_key: str, title: str, y_label: str) -> str:
    plt = _get_pyplot()
    if plt is None:
        print("Skipping {0} plot because matplotlib is not installed.".format(metric_key))
        return ""
    if any(metric_key not in entry["train"] or metric_key not in entry["val"] for entry in history):
        print("Skipping {0} plot because that metric is not present in the training history.".format(metric_key))
        return ""

    epochs = [entry["epoch"] for entry in history]
    train_values = [entry["train"][metric_key] for entry in history]
    val_values = [entry["val"][metric_key] for entry in history]

    ensure_dir(str(target_path.parent))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(epochs, train_values, label="Train", linewidth=2)
    ax.plot(epochs, val_values, label="Validation", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(target_path)


def train_from_config(config: Dict) -> Dict:
    seed_everything(int(config["experiment"]["seed"]))
    device = infer_device(config["experiment"].get("device", "auto"))
    train_dataset, val_dataset, test_dataset, payload = create_datasets(config)
    training_mode = config["training"].get("mode", "real_data")
    rna_dim = int(payload["rna_embeddings"].shape[1]) if training_mode != "simulation" and "rna_embeddings" in payload else 0
    cnv_dim = int(payload["cnv_embeddings"].shape[1]) if training_mode != "simulation" and "cnv_embeddings" in payload else 0
    methylation_dim = (
        int(payload["methylation_embeddings"].shape[1])
        if training_mode != "simulation" and "methylation_embeddings" in payload
        else 0
    )
    architecture = str(config.get("model", {}).get("architecture", "sigformer")).strip().lower()
    demographic_conditioning = _resolve_demographic_conditioning_mode(config)
    demographic_tokens_value = config.get("model", {}).get("demographic_tokens", None)
    if demographic_tokens_value is None:
        uses_demographic_tokens = architecture in {"residual_multimodal_transformer", "residual_aware_multimodal_transformer"}
    else:
        uses_demographic_tokens = str(demographic_tokens_value).strip().lower() in {"true", "on", "1", "yes"}
    demographic_dim = (
        int(payload["demographic_vectors"].shape[1])
        if (
            training_mode != "simulation"
            and "demographic_vectors" in payload
            and (
                demographic_conditioning in {"adaln", "tokens", "both"}
                or uses_demographic_tokens
                or architecture in {"concat_multimodal_transformer", "concat_fusion_multimodal_transformer"}
            )
        )
        else 0
    )

    model = build_model_from_config(
        config=config,
        num_channels=len(payload["mutation_types"]),
        rna_dim=rna_dim,
        cnv_dim=cnv_dim,
        methylation_dim=methylation_dim,
        demographic_dim=demographic_dim,
    ).to(device)

    initialization_metadata = None
    initial_checkpoint_path = config["training"].get("initial_checkpoint_path")
    if initial_checkpoint_path:
        initial_checkpoint_path = Path(initial_checkpoint_path)
        if not initial_checkpoint_path.exists():
            raise FileNotFoundError("training.initial_checkpoint_path does not exist: {0}".format(initial_checkpoint_path))
        initial_payload = torch.load(initial_checkpoint_path, map_location="cpu", weights_only=False)
        initial_state = initial_payload.get("model_state", initial_payload)
        strict_initialization = bool(config["training"].get("initial_checkpoint_strict", True))
        incompatibility = model.load_state_dict(initial_state, strict=strict_initialization)
        initialization_metadata = {
            "checkpoint_path": str(initial_checkpoint_path.resolve()),
            "strict": strict_initialization,
            "missing_keys": list(incompatibility.missing_keys),
            "unexpected_keys": list(incompatibility.unexpected_keys),
            "source_signature_count": len(initial_payload.get("signature_names", [])),
        }
        print(
            "Initialized model from {0} (strict={1}, missing={2}, unexpected={3})".format(
                initial_checkpoint_path,
                strict_initialization,
                len(incompatibility.missing_keys),
                len(incompatibility.unexpected_keys),
            )
        )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    if training_mode == "simulation":
        steps_per_epoch = int(config["training"].get("train_batches_per_epoch", 100))
        sample_weighting_metadata = None
    else:
        sample_weights, sample_weighting_metadata = _build_train_sample_weights(config, train_dataset)
        if sample_weights is None:
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=True,
                num_workers=int(config["training"].get("num_workers", 0)),
            )
        else:
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(config["training"]["batch_size"]),
                sampler=train_sampler,
                num_workers=int(config["training"].get("num_workers", 0)),
            )
            print(
                "Using weighted training sampler: target_label={0} target_train_samples={1}/{2} target_weight={3:.3g} expected_target_fraction={4:.3f}".format(
                    sample_weighting_metadata["target_label"],
                    sample_weighting_metadata["target_train_samples"],
                    sample_weighting_metadata["train_samples"],
                    sample_weighting_metadata["target_weight"],
                    sample_weighting_metadata["effective_target_sampling_fraction"],
                )
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["training"].get("num_workers", 0)),
        )
        val_target_dataset, validation_filter_metadata = _build_label_filtered_dataset(config, val_dataset)
        if val_target_dataset is not None:
            val_target_loader = DataLoader(
                val_target_dataset,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=False,
                num_workers=int(config["training"].get("num_workers", 0)),
            )
            print(
                "Using filtered validation metrics: target_label={0} target_validation_samples={1}/{2}".format(
                    validation_filter_metadata["target_label"],
                    validation_filter_metadata["target_validation_samples"],
                    validation_filter_metadata["validation_samples"],
                )
            )
        else:
            val_target_loader = None
            validation_filter_metadata = None
        signature_bank = torch.tensor(payload["signature_bank"], dtype=torch.float32, device=device)
        candidate_signature_mask = None
        candidate_loss_enabled = any(
            float(config["training"].get(key, 0.0)) > 0.0
            for key in (
                "candidate_presence_weight",
                "candidate_mass_weight",
                "candidate_mixture_weight",
                "cosmic_preservation_weight",
            )
        )
        if candidate_loss_enabled:
            configured_candidate_names = config["training"].get("candidate_signature_names")
            if configured_candidate_names is None:
                candidate_prefix = str(config["training"].get("candidate_signature_prefix", "Aug_"))
                configured_candidate_names = [
                    name for name in payload["signature_names"] if str(name).startswith(candidate_prefix)
                ]
            candidate_name_set = {str(name) for name in configured_candidate_names}
            missing_candidate_names = sorted(candidate_name_set.difference(map(str, payload["signature_names"])))
            if missing_candidate_names:
                raise ValueError(
                    "training.candidate_signature_names contains signatures absent from the bank: {0}".format(
                        missing_candidate_names
                    )
                )
            candidate_signature_mask = torch.tensor(
                [str(name) in candidate_name_set for name in payload["signature_names"]],
                dtype=torch.bool,
                device=device,
            )
            if not bool(candidate_signature_mask.any().item()):
                raise ValueError("Candidate-specific supervision is enabled but no candidate signatures were identified.")
            print(
                "Using candidate-specific supervision for {0} signatures: {1}".format(
                    int(candidate_signature_mask.sum().item()),
                    ", ".join(sorted(candidate_name_set)),
                )
            )
        steps_per_epoch = max(1, len(train_loader))
    scheduler = _build_scheduler(
        optimizer=optimizer,
        config=config,
        steps_per_epoch=steps_per_epoch,
    )
    if training_mode == "simulation":
        simulation_bank, num_cosmic_signatures = simulation_reference_bank_from_payload(payload, config)

    checkpoint_dir = ensure_dir(config["training"]["checkpoint_dir"])
    checkpoint_path = checkpoint_dir / config["training"].get("checkpoint_name", "best.pt")
    checkpoint_meta_path = checkpoint_path.with_suffix(".json")
    stable_checkpoint_path = checkpoint_dir / "best.pt"
    stable_checkpoint_meta_path = stable_checkpoint_path.with_suffix(".json")
    figure_dir = ensure_dir(config["training"].get("figure_dir", "outputs/figures"))
    figure_path = figure_dir / "{0}_curves.png".format(checkpoint_path.stem)
    composition_figure_path = figure_dir / "{0}_composition_mse.png".format(checkpoint_path.stem)
    confidence_figure_path = figure_dir / "{0}_confidence_mse.png".format(checkpoint_path.stem)
    reconstruction_figure_path = figure_dir / "{0}_reconstruction_mse.png".format(checkpoint_path.stem)
    active_figure_path = figure_dir / "{0}_active_gt_0p01.png".format(checkpoint_path.stem)
    effective_figure_path = figure_dir / "{0}_effective_signatures.png".format(checkpoint_path.stem)
    top3_figure_path = figure_dir / "{0}_top3_mass.png".format(checkpoint_path.stem)
    best_val_cosine = -1.0
    checkpoint_metric = str(config["training"].get("checkpoint_metric", "val_cosine")).strip().lower()
    if checkpoint_metric not in {
        "val_cosine",
        "val_loss",
        "val_sparse_objective",
        "val_active_f1",
        "val_active_precision",
        "val_active_recall",
        "val_active_ap",
        "val_active_average_precision",
        "val_paad_cosine",
        "val_paad_loss",
        "val_paad_active_f1",
        "val_paad_active_precision",
        "val_paad_active_recall",
        "val_paad_active_ap",
        "val_paad_active_average_precision",
    }:
        raise ValueError(
            "Unsupported training.checkpoint_metric '{0}'. Use 'val_cosine', 'val_loss', 'val_sparse_objective', 'val_active_f1', 'val_active_precision', 'val_active_recall', 'val_active_ap', 'val_paad_cosine', 'val_paad_loss', 'val_paad_active_f1', or 'val_paad_active_ap'.".format(
                checkpoint_metric
            )
        )
    lower_is_better_metrics = {"val_loss", "val_sparse_objective", "val_paad_loss"}
    best_checkpoint_score = float("inf") if checkpoint_metric in lower_is_better_metrics else -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_patience = int(config["training"].get("early_stopping_patience", 0))
    early_stopping_min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    history = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        val_target_metrics = None
        if training_mode == "simulation":
            train_metrics = _simulation_epoch_pass(
                model=model,
                simulation_bank=simulation_bank,
                num_cosmic_signatures=num_cosmic_signatures,
                device=device,
                training_cfg=config["training"],
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_index=epoch,
                seed=int(config["experiment"]["seed"]),
                is_validation=False,
            )
            val_metrics = _simulation_epoch_pass(
                model=model,
                simulation_bank=simulation_bank,
                num_cosmic_signatures=num_cosmic_signatures,
                device=device,
                training_cfg=config["training"],
                optimizer=None,
                scheduler=None,
                epoch_index=0,
                seed=int(config["experiment"]["seed"]),
                is_validation=True,
            )
        else:
            train_metrics = _real_data_epoch_pass(
                model=model,
                loader=train_loader,
                signature_bank=signature_bank,
                device=device,
                training_cfg=config["training"],
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
                candidate_signature_mask=candidate_signature_mask,
            )
            val_metrics = _real_data_epoch_pass(
                model=model,
                loader=val_loader,
                signature_bank=signature_bank,
                device=device,
                training_cfg=config["training"],
                optimizer=None,
                scheduler=None,
                grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
                candidate_signature_mask=candidate_signature_mask,
            )
            if val_target_loader is not None:
                val_target_metrics = _real_data_epoch_pass(
                    model=model,
                    loader=val_target_loader,
                    signature_bank=signature_bank,
                    device=device,
                    training_cfg=config["training"],
                    optimizer=None,
                    scheduler=None,
                    grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
                    candidate_signature_mask=candidate_signature_mask,
                )
                for key, value in val_target_metrics.items():
                    val_metrics["paad_{0}".format(key)] = value
            else:
                val_target_metrics = None
        history_entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if training_mode != "simulation" and val_target_metrics is not None:
            history_entry["val_paad"] = val_target_metrics
        history.append(history_entry)
        message = "epoch={0} train_loss={1:.4f} train_recon={2:.4f} val_loss={3:.4f} val_recon={4:.4f} val_cos={5:.4f}".format(
            epoch,
            train_metrics["loss"],
            train_metrics["reconstruction_mse"],
            val_metrics["loss"],
            val_metrics["reconstruction_mse"],
            val_metrics["cosine_similarity"],
        )
        if "exposure_entropy" in val_metrics:
            message = "{0} val_entropy={1:.4f}".format(message, val_metrics["exposure_entropy"])
        if "active_gt_0.01" in val_metrics:
            message = "{0} val_active01={1:.2f} val_eff={2:.2f} val_top3={3:.3f}".format(
                message,
                val_metrics["active_gt_0.01"],
                val_metrics["effective_signatures"],
                val_metrics["top3_mass"],
            )
        if "active_f1" in val_metrics:
            message = "{0} val_active_f1={1:.4f} val_active_p={2:.4f} val_active_r={3:.4f}".format(
                message,
                val_metrics["active_f1"],
                val_metrics["active_precision"],
                val_metrics["active_recall"],
            )
        if "active_average_precision" in val_metrics:
            message = "{0} val_active_ap={1:.4f}".format(message, val_metrics["active_average_precision"])
        if val_target_metrics is not None:
            message = "{0} val_paad_cos={1:.4f}".format(
                message,
                val_target_metrics.get("cosine_similarity", 0.0),
            )
            if "active_f1" in val_target_metrics:
                message = "{0} val_paad_active_f1={1:.4f}".format(
                    message,
                    val_target_metrics["active_f1"],
                )
            if "active_average_precision" in val_target_metrics:
                message = "{0} val_paad_active_ap={1:.4f}".format(
                    message,
                    val_target_metrics["active_average_precision"],
                )
        print(message)

        if checkpoint_metric == "val_sparse_objective":
            checkpoint_score = _sparse_objective_score(val_metrics, config["training"])
        elif checkpoint_metric == "val_loss":
            checkpoint_score = val_metrics["loss"]
        elif checkpoint_metric == "val_active_f1":
            checkpoint_score = val_metrics.get("active_f1", 0.0)
        elif checkpoint_metric == "val_active_precision":
            checkpoint_score = val_metrics.get("active_precision", 0.0)
        elif checkpoint_metric == "val_active_recall":
            checkpoint_score = val_metrics.get("active_recall", 0.0)
        elif checkpoint_metric in {"val_active_ap", "val_active_average_precision"}:
            checkpoint_score = val_metrics.get("active_average_precision", 0.0)
        elif checkpoint_metric == "val_paad_loss":
            checkpoint_score = val_metrics.get("paad_loss", float("inf"))
        elif checkpoint_metric == "val_paad_cosine":
            checkpoint_score = val_metrics.get("paad_cosine_similarity", -float("inf"))
        elif checkpoint_metric == "val_paad_active_f1":
            checkpoint_score = val_metrics.get("paad_active_f1", 0.0)
        elif checkpoint_metric == "val_paad_active_precision":
            checkpoint_score = val_metrics.get("paad_active_precision", 0.0)
        elif checkpoint_metric == "val_paad_active_recall":
            checkpoint_score = val_metrics.get("paad_active_recall", 0.0)
        elif checkpoint_metric in {"val_paad_active_ap", "val_paad_active_average_precision"}:
            checkpoint_score = val_metrics.get("paad_active_average_precision", 0.0)
        else:
            checkpoint_score = val_metrics["cosine_similarity"]
        if checkpoint_metric in lower_is_better_metrics:
            improved = checkpoint_score < (best_checkpoint_score - early_stopping_min_delta)
        else:
            improved = checkpoint_score > (best_checkpoint_score + early_stopping_min_delta)
        best_val_cosine = max(best_val_cosine, val_metrics["cosine_similarity"])
        if improved:
            best_checkpoint_score = checkpoint_score
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_payload = {
                "model_state": model.state_dict(),
                "config": config,
                "mutation_types": payload["mutation_types"],
                "signature_names": payload["signature_names"],
                "signature_bank": payload["signature_bank"],
                "rna_dim": rna_dim,
                "rna_metadata": payload.get("rna_metadata"),
                "cnv_dim": cnv_dim,
                "cnv_metadata": payload.get("cnv_metadata"),
                "methylation_dim": methylation_dim,
                "methylation_metadata": payload.get("methylation_metadata"),
                "demographic_dim": demographic_dim,
                "demographic_metadata": payload.get("demographic_metadata"),
                "demographic_conditioning": demographic_conditioning,
                "architecture": str(config.get("model", {}).get("architecture", "sigformer")),
                "initialization": initialization_metadata,
                "metrics": {
                    "best_val_cosine": best_val_cosine,
                    "checkpoint_metric": checkpoint_metric,
                    "best_checkpoint_score": best_checkpoint_score,
                    "best_epoch": best_epoch,
                    "history": history,
                },
            }
            torch.save(checkpoint_payload, checkpoint_path)
            torch.save(checkpoint_payload, stable_checkpoint_path)
            print("Saved checkpoint to {0}".format(checkpoint_path))
            meta_payload = {
                "checkpoint_path": str(checkpoint_path),
                "stable_checkpoint_path": str(stable_checkpoint_path),
                "figure_path": str(figure_path),
                "composition_figure_path": str(composition_figure_path),
                "confidence_figure_path": str(confidence_figure_path),
                "reconstruction_figure_path": str(reconstruction_figure_path),
                "active_gt_0.01_figure_path": str(active_figure_path),
                "effective_signatures_figure_path": str(effective_figure_path),
                "top3_mass_figure_path": str(top3_figure_path),
                "run_timestamp": config.get("run_timestamp"),
                "best_val_cosine": best_val_cosine,
                "checkpoint_metric": checkpoint_metric,
                "best_checkpoint_score": best_checkpoint_score,
                "best_epoch": best_epoch,
                "num_samples": len(payload["sample_ids"]),
                "num_channels": len(payload["mutation_types"]),
                "num_signatures": len(payload["signature_names"]),
                "rna_dim": rna_dim,
                "rna_metadata": payload.get("rna_metadata"),
                "cnv_dim": cnv_dim,
                "cnv_metadata": payload.get("cnv_metadata"),
                "methylation_dim": methylation_dim,
                "methylation_metadata": payload.get("methylation_metadata"),
                "demographic_dim": demographic_dim,
                "demographic_metadata": payload.get("demographic_metadata"),
                "demographic_conditioning": demographic_conditioning,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
                "training_mode": training_mode,
                "initialization": initialization_metadata,
                "sample_weighting": sample_weighting_metadata,
                "model": config["model"],
                "training": {k: v for k, v in config["training"].items() if k not in {"checkpoint_dir", "checkpoint_name", "figure_dir"}},
                "validation_filter": validation_filter_metadata,
                "history": history,
            }
            save_json(meta_payload, str(checkpoint_meta_path))
            stable_meta_payload = dict(meta_payload)
            stable_meta_payload["checkpoint_path"] = str(stable_checkpoint_path)
            stable_meta_payload["source_checkpoint_path"] = str(checkpoint_path)
            save_json(stable_meta_payload, str(stable_checkpoint_meta_path))
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(
                    "Early stopping at epoch {0}: no {1} improvement for {2} epochs. Best epoch={3} best_score={4:.6f}".format(
                        epoch,
                        checkpoint_metric,
                        early_stopping_patience,
                        best_epoch,
                        best_checkpoint_score,
                    )
                )
                break

    saved_figure_path = _save_training_plots(history, figure_path)
    saved_composition_figure_path = _save_metric_plot(history, composition_figure_path, "composition_mse", "Exposure MSE vs Epoch", "Exposure MSE")
    saved_confidence_figure_path = _save_metric_plot(history, confidence_figure_path, "confidence_mse", "Confidence MSE vs Epoch", "Confidence MSE")
    saved_reconstruction_figure_path = _save_metric_plot(history, reconstruction_figure_path, "reconstruction_mse", "Reconstruction MSE vs Epoch", "Reconstruction MSE")
    saved_active_figure_path = _save_metric_plot(history, active_figure_path, "active_gt_0.01", "Active Signatures > 0.01 vs Epoch", "Mean Active Signatures")
    saved_effective_figure_path = _save_metric_plot(history, effective_figure_path, "effective_signatures", "Effective Signatures vs Epoch", "Mean Effective Signatures")
    saved_top3_figure_path = _save_metric_plot(history, top3_figure_path, "top3_mass", "Top-3 Exposure Mass vs Epoch", "Mean Top-3 Mass")
    print("Saved training curves to {0}".format(saved_figure_path))
    print("Saved exposure MSE curves to {0}".format(saved_composition_figure_path))
    print("Saved confidence MSE curves to {0}".format(saved_confidence_figure_path))
    print("Saved reconstruction MSE curves to {0}".format(saved_reconstruction_figure_path))
    print("Saved active signature curves to {0}".format(saved_active_figure_path))
    print("Saved effective signature curves to {0}".format(saved_effective_figure_path))
    print("Saved top-3 exposure mass curves to {0}".format(saved_top3_figure_path))

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_meta_path": str(checkpoint_meta_path),
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "stable_checkpoint_meta_path": str(stable_checkpoint_meta_path),
        "figure_path": str(figure_path),
        "composition_figure_path": str(composition_figure_path),
        "confidence_figure_path": str(confidence_figure_path),
        "reconstruction_figure_path": str(reconstruction_figure_path),
        "active_gt_0.01_figure_path": str(active_figure_path),
        "effective_signatures_figure_path": str(effective_figure_path),
        "top3_mass_figure_path": str(top3_figure_path),
        "best_val_cosine": best_val_cosine,
        "checkpoint_metric": checkpoint_metric,
        "best_checkpoint_score": best_checkpoint_score,
        "best_epoch": best_epoch,
        "history": history,
    }
