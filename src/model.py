import torch
from torch import nn
import torch.nn.functional as F


def _config_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Could not interpret boolean-like config value '{0}'.".format(value))


def _resolve_demographic_mode(value) -> str:
    if value is None:
        return "both"
    text = str(value).strip().lower()
    if text in {"both", "all", "full"}:
        return "both"
    if text in {"adaln", "conditioning", "condition"}:
        return "adaln"
    if text in {"tokens", "token"}:
        return "tokens"
    if text in {"none", "off", "false", "disabled"}:
        return "none"
    raise ValueError(
        "Unsupported model.demographic_conditioning value '{0}'. Use 'none', 'tokens', 'adaln', or 'both'.".format(
            value
        )
    )


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model: int, condition_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, d_model * 2),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        normalized = self.norm(x)
        return (1.0 + scale.unsqueeze(1)) * normalized + shift.unsqueeze(1)


class SampleBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ff_multiplier: int, condition_dim: int = 0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.use_adaln = int(condition_dim) > 0
        if self.use_adaln:
            self.attn_norm = AdaptiveLayerNorm(d_model, condition_dim)
            self.ff_norm = AdaptiveLayerNorm(d_model, condition_dim)
        else:
            self.attn_norm = nn.LayerNorm(d_model)
            self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        if self.use_adaln:
            if condition is None:
                raise ValueError("SampleBlock expected a conditioning vector for AdaLN, but none was provided.")
            attn_input = self.attn_norm(x, condition)
            ff_input = self.ff_norm(x, condition)
        else:
            attn_input = self.attn_norm(x)
            ff_input = self.ff_norm(x)

        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_out)
        return x + self.dropout(self.ff(ff_input))


class ReferenceBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ff_multiplier: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ref_tokens: torch.Tensor, sample_tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(self.norm1(ref_tokens), self.norm1(ref_tokens), self.norm1(ref_tokens), need_weights=False)
        ref_tokens = ref_tokens + self.dropout(attn_out)
        cross_out, _ = self.cross_attn(self.norm2(ref_tokens), sample_tokens, sample_tokens, need_weights=False)
        ref_tokens = ref_tokens + self.dropout(cross_out)
        return ref_tokens + self.dropout(self.ff(ref_tokens))


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ff_multiplier: int, condition_dim: int = 0) -> None:
        super().__init__()
        self.use_adaln = int(condition_dim) > 0
        if self.use_adaln:
            self.norm1 = AdaptiveLayerNorm(d_model, condition_dim)
            self.norm2 = AdaptiveLayerNorm(d_model, condition_dim)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        if self.use_adaln:
            if condition is None:
                raise ValueError("SelfAttentionBlock expected a conditioning vector for AdaLN, but none was provided.")
            attn_in = self.norm1(x, condition)
        else:
            attn_in = self.norm1(x)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout(attn_out)
        if self.use_adaln:
            ff_in = self.norm2(x, condition)
        else:
            ff_in = self.norm2(x)
        ff_out = self.ff(ff_in)
        return x + self.dropout(ff_out)


class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ff_multiplier: int, condition_dim: int = 0) -> None:
        super().__init__()
        self.use_adaln = int(condition_dim) > 0
        if self.use_adaln:
            self.query_norm = AdaptiveLayerNorm(d_model, condition_dim)
            self.ff_norm = AdaptiveLayerNorm(d_model, condition_dim)
        else:
            self.query_norm = nn.LayerNorm(d_model)
            self.ff_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens: torch.Tensor, modality_tokens: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        if modality_tokens is None or modality_tokens.shape[1] == 0:
            return query_tokens
        if self.use_adaln:
            if condition is None:
                raise ValueError("CrossAttentionFusionBlock expected a conditioning vector for AdaLN, but none was provided.")
            query_in = self.query_norm(query_tokens, condition)
        else:
            query_in = self.query_norm(query_tokens)
        cross_out, _ = self.cross_attn(
            query_in,
            self.kv_norm(modality_tokens),
            self.kv_norm(modality_tokens),
            need_weights=False,
        )
        query_tokens = query_tokens + self.dropout(cross_out)
        if self.use_adaln:
            ff_in = self.ff_norm(query_tokens, condition)
        else:
            ff_in = self.ff_norm(query_tokens)
        return query_tokens + self.dropout(self.ff(ff_in))


class ModalityEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ff_multiplier: int, num_layers: int, condition_dim: int = 0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim) for _ in range(max(1, num_layers))]
        )

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens, condition=condition)
        return tokens


class BidirectionalCrossBranchFusion(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ff_multiplier: int, condition_dim: int = 0) -> None:
        super().__init__()
        self.known_to_residual = CrossAttentionFusionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)
        self.residual_to_known = CrossAttentionFusionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)

    def forward(self, known_tokens: torch.Tensor, residual_tokens: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        known_residual = self.known_to_residual(known_tokens, residual_tokens, condition=condition)
        residual_known = self.residual_to_known(residual_tokens, known_tokens, condition=condition)
        return known_residual, residual_known


class SigFormerRefitter(nn.Module):
    def __init__(
        self,
        num_channels: int,
        d_model: int,
        num_heads: int,
        num_sample_layers: int,
        num_reference_layers: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        rna_dim: int = 0,
        cnv_dim: int = 0,
        methylation_dim: int = 0,
        demographic_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.rna_dim = int(rna_dim)
        self.cnv_dim = int(cnv_dim)
        self.methylation_dim = int(methylation_dim)
        self.demographic_dim = int(demographic_dim)
        self.channel_embedding = nn.Embedding(num_channels, d_model)
        self.value_projection = nn.Linear(1, d_model)
        self.sample_input_norm = nn.LayerNorm(d_model)
        if self.rna_dim > 0:
            self.rna_projection = nn.Sequential(
                nn.LayerNorm(self.rna_dim),
                nn.Linear(self.rna_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.rna_token_norm = nn.LayerNorm(d_model)
            self.rna_feature_gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
            )
        else:
            self.rna_projection = None
            self.rna_token_norm = None
            self.rna_feature_gate = None
        if self.demographic_dim > 0:
            self.demographic_condition = nn.Sequential(
                nn.Linear(self.demographic_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
            )
        else:
            self.demographic_condition = None
        self.sample_blocks = nn.ModuleList(
            [
                SampleBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_multiplier=ff_multiplier,
                    condition_dim=d_model if self.demographic_dim > 0 else 0,
                )
                for _ in range(num_sample_layers)
            ]
        )
        self.reference_projection = nn.Linear(num_channels, d_model)
        self.reference_blocks = nn.ModuleList(
            [ReferenceBlock(d_model, num_heads, dropout, ff_multiplier) for _ in range(num_reference_layers)]
        )
        self.reference_norm = nn.LayerNorm(d_model)
        self.composition_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model + 2),
            nn.Linear(d_model + 2, 1),
        )

    def forward(
        self,
        sample_profile: torch.Tensor,
        reference_signatures: torch.Tensor,
        rna_embedding: torch.Tensor = None,
        cnv_embedding: torch.Tensor = None,
        methylation_embedding: torch.Tensor = None,
        demographic_vector: torch.Tensor = None,
        known_exposures: torch.Tensor = None,
    ):
        del cnv_embedding, methylation_embedding, known_exposures
        batch_size, num_channels = sample_profile.shape
        if num_channels != self.num_channels:
            raise ValueError("Expected {0} channels, got {1}".format(self.num_channels, num_channels))
        if self.rna_dim > 0:
            if rna_embedding is None:
                raise ValueError("This SigFormerRefitter was initialized with rna_dim > 0, but no RNA embedding was provided.")
            if rna_embedding.shape[-1] != self.rna_dim:
                raise ValueError("Expected RNA embedding dim {0}, got {1}".format(self.rna_dim, rna_embedding.shape[-1]))
        if self.demographic_dim > 0:
            if demographic_vector is None:
                raise ValueError(
                    "This SigFormerRefitter was initialized with demographic_dim > 0, but no demographic vector was provided."
                )
            if demographic_vector.shape[-1] != self.demographic_dim:
                raise ValueError(
                    "Expected demographic vector dim {0}, got {1}".format(
                        self.demographic_dim,
                        demographic_vector.shape[-1],
                    )
                )

        depth = torch.clamp(sample_profile.sum(dim=1, keepdim=True), min=1.0)
        normalized_input = sample_profile / depth * 50000.0
        channel_ids = torch.arange(self.num_channels, device=sample_profile.device).unsqueeze(0).expand(batch_size, -1)
        sample_tokens = self.value_projection(normalized_input.unsqueeze(-1)) + self.channel_embedding(channel_ids)
        if self.rna_projection is not None:
            rna_context = self.rna_projection(rna_embedding)
            gate, bias = self.rna_feature_gate(rna_context).chunk(2, dim=-1)
            sample_tokens = sample_tokens * torch.sigmoid(gate).unsqueeze(1) + bias.unsqueeze(1)
            rna_token = self.rna_token_norm(rna_context).unsqueeze(1)
            sample_tokens = torch.cat([sample_tokens, rna_token], dim=1)
        sample_tokens = self.sample_input_norm(sample_tokens)
        demographic_condition = self.demographic_condition(demographic_vector) if self.demographic_condition is not None else None
        for block in self.sample_blocks:
            sample_tokens = block(sample_tokens, condition=demographic_condition)

        reference_tokens = self.reference_projection(reference_signatures)
        for block in self.reference_blocks:
            reference_tokens = block(reference_tokens, sample_tokens)
        reference_tokens = self.reference_norm(reference_tokens)

        exposure_logits = self.composition_head(reference_tokens).squeeze(-1)
        exposures = torch.softmax(exposure_logits, dim=-1)
        reconstruction = torch.bmm(exposures.unsqueeze(1), reference_signatures).squeeze(1)
        reconstruction = reconstruction / torch.clamp(reconstruction.sum(dim=1, keepdim=True), min=1e-12)
        reconstruction_scaled = reconstruction * 50000.0
        residual_score = 1.0 - F.cosine_similarity(normalized_input, reconstruction_scaled, dim=1, eps=1e-8)
        confidence_features = torch.cat(
            [
                reference_tokens.detach(),
                torch.log(depth).unsqueeze(1).expand(-1, reference_tokens.shape[1], -1),
                residual_score.unsqueeze(1).unsqueeze(-1).expand(-1, reference_tokens.shape[1], -1),
            ],
            dim=-1,
        )
        confidence = torch.sigmoid(self.confidence_head(confidence_features).squeeze(-1))
        return {
            "exposure_logits": exposure_logits,
            "exposures": exposures,
            "reconstruction": reconstruction,
            "confidence": confidence,
            "depth": depth.squeeze(-1),
        }


class ResidualAwareMultimodalTransformer(nn.Module):
    def __init__(
        self,
        num_channels: int,
        d_model: int,
        num_heads: int,
        num_sample_layers: int,
        num_reference_layers: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        rna_dim: int = 0,
        cnv_dim: int = 0,
        methylation_dim: int = 0,
        demographic_dim: int = 0,
        latent_token_count: int = 16,
        num_candidate_embeddings: int = 4,
        metadata_token_count: int = 2,
        rna_token_count: int = 4,
        cnv_token_count: int = 4,
        methylation_token_count: int = 4,
        use_residual_branch: bool = True,
        use_cross_branch_fusion: bool = True,
        use_demographic_tokens: bool = True,
        use_demographic_conditioning: bool = True,
        use_active_head: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.rna_dim = int(rna_dim)
        self.cnv_dim = int(cnv_dim)
        self.methylation_dim = int(methylation_dim)
        self.demographic_dim = int(demographic_dim)
        self.latent_token_count = int(latent_token_count)
        self.num_candidate_embeddings = int(num_candidate_embeddings)
        self.metadata_token_count = int(metadata_token_count)
        self.rna_token_count = int(rna_token_count)
        self.cnv_token_count = int(cnv_token_count)
        self.methylation_token_count = int(methylation_token_count)
        self.use_residual_branch = bool(use_residual_branch)
        self.use_cross_branch_fusion = bool(use_cross_branch_fusion)
        self.use_active_head = bool(use_active_head)
        self.use_demographic_tokens = bool(use_demographic_tokens) and self.demographic_dim > 0
        self.use_demographic_conditioning = bool(use_demographic_conditioning) and self.demographic_dim > 0
        condition_dim = d_model if self.use_demographic_conditioning else 0

        self.mutation_channel_embedding = nn.Embedding(self.num_channels, d_model)
        self.mutation_value_projection = nn.Linear(1, d_model)
        self.residual_channel_embedding = nn.Embedding(self.num_channels, d_model)
        self.residual_value_projection = nn.Linear(1, d_model)
        self.signature_projection = nn.Linear(self.num_channels, d_model)
        self.signature_exposure_projection = nn.Linear(1, d_model)
        self.signature_index_embedding = nn.Embedding(512, d_model)

        self.mutation_encoder = ModalityEncoder(
            d_model, num_heads, dropout, ff_multiplier, num_sample_layers, condition_dim=condition_dim
        )
        self.signature_encoder = ModalityEncoder(
            d_model, num_heads, dropout, ff_multiplier, num_reference_layers, condition_dim=condition_dim
        )
        if self.use_residual_branch:
            self.residual_encoder = ModalityEncoder(
                d_model, num_heads, dropout, ff_multiplier, num_sample_layers, condition_dim=condition_dim
            )
        else:
            self.residual_encoder = None

        if self.use_demographic_conditioning:
            self.demographic_condition = nn.Sequential(
                nn.LayerNorm(self.demographic_dim),
                nn.Linear(self.demographic_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
            )
        else:
            self.demographic_condition = None

        if self.use_demographic_tokens:
            self.metadata_projection = nn.Sequential(
                nn.LayerNorm(self.demographic_dim),
                nn.Linear(self.demographic_dim, d_model * self.metadata_token_count),
                nn.GELU(),
                nn.Linear(d_model * self.metadata_token_count, d_model * self.metadata_token_count),
            )
            self.metadata_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, 1)
        else:
            self.metadata_projection = None
            self.metadata_encoder = None

        if self.rna_dim > 0:
            self.rna_projection = nn.Sequential(
                nn.LayerNorm(self.rna_dim),
                nn.Linear(self.rna_dim, d_model * self.rna_token_count),
                nn.GELU(),
                nn.Linear(d_model * self.rna_token_count, d_model * self.rna_token_count),
            )
            self.rna_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, 1, condition_dim=condition_dim)
        else:
            self.rna_projection = None
            self.rna_encoder = None

        if self.cnv_dim > 0:
            self.cnv_projection = nn.Sequential(
                nn.LayerNorm(self.cnv_dim),
                nn.Linear(self.cnv_dim, d_model * self.cnv_token_count),
                nn.GELU(),
                nn.Linear(d_model * self.cnv_token_count, d_model * self.cnv_token_count),
            )
            self.cnv_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, 1, condition_dim=condition_dim)
        else:
            self.cnv_projection = None
            self.cnv_encoder = None

        if self.methylation_dim > 0:
            self.methylation_projection = nn.Sequential(
                nn.LayerNorm(self.methylation_dim),
                nn.Linear(self.methylation_dim, d_model * self.methylation_token_count),
                nn.GELU(),
                nn.Linear(d_model * self.methylation_token_count, d_model * self.methylation_token_count),
            )
            self.methylation_encoder = ModalityEncoder(
                d_model, num_heads, dropout, ff_multiplier, 1, condition_dim=condition_dim
            )
        else:
            self.methylation_projection = None
            self.methylation_encoder = None

        self.latent_tokens = nn.Parameter(torch.randn(self.latent_token_count, d_model) * 0.02)
        self.latent_mutation_fusion = CrossAttentionFusionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)
        self.latent_signature_fusion = CrossAttentionFusionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)
        self.latent_residual_fusion = (
            CrossAttentionFusionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)
            if self.use_residual_branch
            else None
        )
        self.latent_metadata_fusion = CrossAttentionFusionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)
        self.latent_rna_fusion = CrossAttentionFusionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)
        self.latent_cnv_fusion = (
            CrossAttentionFusionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)
            if self.cnv_dim > 0
            else None
        )
        self.latent_methylation_fusion = (
            CrossAttentionFusionBlock(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)
            if self.methylation_dim > 0
            else None
        )

        self.baseline_exposure_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.branch_fusion = (
            BidirectionalCrossBranchFusion(d_model, num_heads, dropout, ff_multiplier, condition_dim=condition_dim)
            if self.use_residual_branch and self.use_cross_branch_fusion
            else None
        )
        self.combined_to_signature = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
        )
        self.refined_exposure_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.active_head = (
            nn.Sequential(
                nn.LayerNorm(d_model + 2),
                nn.Linear(d_model + 2, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            if self.use_active_head
            else None
        )
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.num_channels),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.num_channels),
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model + 2),
            nn.Linear(d_model + 2, 1),
        )

    def _make_mutation_tokens(self, normalized_input: torch.Tensor) -> torch.Tensor:
        batch_size = normalized_input.shape[0]
        channel_ids = torch.arange(self.num_channels, device=normalized_input.device).unsqueeze(0).expand(batch_size, -1)
        mutation_tokens = self.mutation_value_projection(normalized_input.unsqueeze(-1)) + self.mutation_channel_embedding(channel_ids)
        return mutation_tokens

    def _make_signature_tokens(
        self,
        reference_signatures: torch.Tensor,
        known_exposures: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, num_signatures, _ = reference_signatures.shape
        signature_ids = torch.arange(num_signatures, device=reference_signatures.device).unsqueeze(0).expand(batch_size, -1)
        signature_tokens = self.signature_projection(reference_signatures) + self.signature_index_embedding(signature_ids)
        if known_exposures is not None:
            signature_tokens = signature_tokens + self.signature_exposure_projection(known_exposures.unsqueeze(-1))
        return signature_tokens

    def _make_residual_tokens(self, residual_profile: torch.Tensor) -> torch.Tensor:
        batch_size = residual_profile.shape[0]
        channel_ids = torch.arange(self.num_channels, device=residual_profile.device).unsqueeze(0).expand(batch_size, -1)
        residual_tokens = self.residual_value_projection(residual_profile.unsqueeze(-1)) + self.residual_channel_embedding(channel_ids)
        return residual_tokens

    def _make_metadata_tokens(self, demographic_vector: torch.Tensor) -> torch.Tensor:
        if self.metadata_projection is None or demographic_vector is None:
            return None
        metadata_tokens = self.metadata_projection(demographic_vector).reshape(
            demographic_vector.shape[0], self.metadata_token_count, -1
        )
        return self.metadata_encoder(metadata_tokens)

    def _make_rna_tokens(self, rna_embedding: torch.Tensor) -> torch.Tensor:
        if self.rna_projection is None or rna_embedding is None:
            return None
        rna_tokens = self.rna_projection(rna_embedding).reshape(rna_embedding.shape[0], self.rna_token_count, -1)
        return rna_tokens

    def _make_cnv_tokens(self, cnv_embedding: torch.Tensor) -> torch.Tensor:
        if self.cnv_projection is None or cnv_embedding is None:
            return None
        cnv_tokens = self.cnv_projection(cnv_embedding).reshape(cnv_embedding.shape[0], self.cnv_token_count, -1)
        return cnv_tokens

    def _make_methylation_tokens(self, methylation_embedding: torch.Tensor) -> torch.Tensor:
        if self.methylation_projection is None or methylation_embedding is None:
            return None
        methylation_tokens = self.methylation_projection(methylation_embedding).reshape(
            methylation_embedding.shape[0], self.methylation_token_count, -1
        )
        return methylation_tokens

    def forward(
        self,
        sample_profile: torch.Tensor,
        reference_signatures: torch.Tensor,
        rna_embedding: torch.Tensor = None,
        cnv_embedding: torch.Tensor = None,
        methylation_embedding: torch.Tensor = None,
        demographic_vector: torch.Tensor = None,
        known_exposures: torch.Tensor = None,
    ):
        batch_size, num_channels = sample_profile.shape
        if num_channels != self.num_channels:
            raise ValueError("Expected {0} channels, got {1}".format(self.num_channels, num_channels))
        if self.rna_dim > 0 and rna_embedding is not None and rna_embedding.shape[-1] != self.rna_dim:
            raise ValueError("Expected RNA embedding dim {0}, got {1}".format(self.rna_dim, rna_embedding.shape[-1]))
        if self.cnv_dim > 0 and cnv_embedding is not None and cnv_embedding.shape[-1] != self.cnv_dim:
            raise ValueError("Expected CNV embedding dim {0}, got {1}".format(self.cnv_dim, cnv_embedding.shape[-1]))
        if (
            self.methylation_dim > 0
            and methylation_embedding is not None
            and methylation_embedding.shape[-1] != self.methylation_dim
        ):
            raise ValueError(
                "Expected methylation embedding dim {0}, got {1}".format(
                    self.methylation_dim, methylation_embedding.shape[-1]
                )
            )
        if self.demographic_dim > 0:
            if demographic_vector is None:
                raise ValueError(
                    "This ResidualAwareMultimodalTransformer was initialized with demographic_dim > 0, but no demographic vector was provided."
                )
            if demographic_vector.shape[-1] != self.demographic_dim:
                raise ValueError("Expected demographic vector dim {0}, got {1}".format(self.demographic_dim, demographic_vector.shape[-1]))

        depth = torch.clamp(sample_profile.sum(dim=1, keepdim=True), min=1.0)
        normalized_input = sample_profile / depth

        demographic_condition = self.demographic_condition(demographic_vector) if self.demographic_condition is not None else None

        mutation_tokens = self.mutation_encoder(self._make_mutation_tokens(normalized_input), condition=demographic_condition)
        signature_tokens = self.signature_encoder(
            self._make_signature_tokens(reference_signatures, known_exposures=known_exposures),
            condition=demographic_condition,
        )

        baseline_logits = self.baseline_exposure_head(signature_tokens).squeeze(-1)
        baseline_exposures = torch.softmax(baseline_logits, dim=-1)
        baseline_reconstruction = torch.bmm(baseline_exposures.unsqueeze(1), reference_signatures).squeeze(1)
        baseline_reconstruction = baseline_reconstruction / torch.clamp(
            baseline_reconstruction.sum(dim=1, keepdim=True), min=1e-12
        )

        if self.use_residual_branch:
            residual_profile = normalized_input - baseline_reconstruction
            residual_tokens = self.residual_encoder(self._make_residual_tokens(residual_profile), condition=demographic_condition)
        else:
            residual_profile = torch.zeros_like(normalized_input)
            residual_tokens = None
        metadata_tokens = self._make_metadata_tokens(demographic_vector)
        rna_tokens = self._make_rna_tokens(rna_embedding)
        if rna_tokens is not None:
            rna_tokens = self.rna_encoder(rna_tokens, condition=demographic_condition)
        cnv_tokens = self._make_cnv_tokens(cnv_embedding)
        if cnv_tokens is not None:
            cnv_tokens = self.cnv_encoder(cnv_tokens, condition=demographic_condition)
        methylation_tokens = self._make_methylation_tokens(methylation_embedding)
        if methylation_tokens is not None:
            methylation_tokens = self.methylation_encoder(methylation_tokens, condition=demographic_condition)

        latent_tokens = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        latent_tokens = self.latent_mutation_fusion(latent_tokens, mutation_tokens, condition=demographic_condition)
        latent_tokens = self.latent_signature_fusion(latent_tokens, signature_tokens, condition=demographic_condition)
        if residual_tokens is not None and self.latent_residual_fusion is not None:
            latent_tokens = self.latent_residual_fusion(latent_tokens, residual_tokens, condition=demographic_condition)
        if metadata_tokens is not None:
            latent_tokens = self.latent_metadata_fusion(latent_tokens, metadata_tokens, condition=demographic_condition)
        if rna_tokens is not None:
            latent_tokens = self.latent_rna_fusion(latent_tokens, rna_tokens, condition=demographic_condition)
        if cnv_tokens is not None:
            latent_tokens = self.latent_cnv_fusion(latent_tokens, cnv_tokens, condition=demographic_condition)
        if methylation_tokens is not None:
            latent_tokens = self.latent_methylation_fusion(latent_tokens, methylation_tokens, condition=demographic_condition)

        if self.branch_fusion is not None and residual_tokens is not None:
            known_residual_tokens, residual_known_tokens = self.branch_fusion(
                signature_tokens,
                residual_tokens,
                condition=demographic_condition,
            )
        else:
            known_residual_tokens = signature_tokens
            residual_known_tokens = residual_tokens if residual_tokens is not None else mutation_tokens.new_zeros(
                batch_size, self.num_channels, signature_tokens.shape[-1]
            )

        latent_pool = latent_tokens.mean(dim=1)
        known_residual_pool = known_residual_tokens.mean(dim=1)
        residual_known_pool = residual_known_tokens.mean(dim=1)
        combined = torch.cat([latent_pool, known_residual_pool, residual_known_pool], dim=-1)

        signature_context = self.combined_to_signature(combined).unsqueeze(1)
        refined_signature_tokens = signature_tokens + signature_context
        exposure_logits = self.refined_exposure_head(refined_signature_tokens).squeeze(-1)
        exposures = torch.softmax(exposure_logits, dim=-1)
        reconstruction = torch.bmm(exposures.unsqueeze(1), reference_signatures).squeeze(1)
        reconstruction = reconstruction / torch.clamp(reconstruction.sum(dim=1, keepdim=True), min=1e-12)

        refined_reconstruction_logits = self.reconstruction_head(combined)
        refined_reconstruction = torch.softmax(refined_reconstruction_logits, dim=-1)
        residual_prediction = self.residual_head(combined)
        candidate_embeddings = self.candidate_projection(latent_tokens[:, : self.num_candidate_embeddings, :])

        residual_score = 1.0 - F.cosine_similarity(reconstruction, normalized_input, dim=1, eps=1e-8)
        confidence_features = torch.cat(
            [
                refined_signature_tokens.detach(),
                torch.log(depth).unsqueeze(1).expand(-1, refined_signature_tokens.shape[1], -1),
                residual_score.unsqueeze(1).unsqueeze(-1).expand(-1, refined_signature_tokens.shape[1], -1),
            ],
            dim=-1,
        )
        confidence = torch.sigmoid(self.confidence_head(confidence_features).squeeze(-1))
        active_logits = None
        active_probabilities = None
        if self.active_head is not None:
            active_logits = self.active_head(confidence_features).squeeze(-1)
            active_probabilities = torch.sigmoid(active_logits)

        outputs = {
            "baseline_exposure_logits": baseline_logits,
            "baseline_exposures": baseline_exposures,
            "baseline_reconstruction": baseline_reconstruction,
            "residual_input": residual_profile,
            "candidate_embeddings": candidate_embeddings,
            "refined_reconstruction": refined_reconstruction,
            "residual_prediction": residual_prediction,
            "exposure_logits": exposure_logits,
            "exposures": exposures,
            "reconstruction": reconstruction,
            "confidence": confidence,
            "depth": depth.squeeze(-1),
        }
        if active_logits is not None:
            outputs["active_logits"] = active_logits
            outputs["active_probabilities"] = active_probabilities
        return outputs


class ConcatFusionMultimodalRefitter(nn.Module):
    def __init__(
        self,
        num_channels: int,
        d_model: int,
        num_heads: int,
        num_sample_layers: int,
        num_reference_layers: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        rna_dim: int = 0,
        cnv_dim: int = 0,
        methylation_dim: int = 0,
        demographic_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.rna_dim = int(rna_dim)
        self.cnv_dim = int(cnv_dim)
        self.methylation_dim = int(methylation_dim)
        self.demographic_dim = int(demographic_dim)
        self.channel_embedding = nn.Embedding(self.num_channels, d_model)
        self.value_projection = nn.Linear(1, d_model)
        self.signature_projection = nn.Linear(self.num_channels, d_model)
        self.signature_exposure_projection = nn.Linear(1, d_model)
        self.signature_index_embedding = nn.Embedding(512, d_model)
        self.mutation_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, num_sample_layers)
        self.signature_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, num_reference_layers)
        self.rna_projection = (
            nn.Sequential(
                nn.LayerNorm(self.rna_dim),
                nn.Linear(self.rna_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.rna_dim > 0
            else None
        )
        self.demographic_projection = (
            nn.Sequential(
                nn.LayerNorm(self.demographic_dim),
                nn.Linear(self.demographic_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.demographic_dim > 0
            else None
        )
        self.cnv_projection = (
            nn.Sequential(
                nn.LayerNorm(self.cnv_dim),
                nn.Linear(self.cnv_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.cnv_dim > 0
            else None
        )
        self.methylation_projection = (
            nn.Sequential(
                nn.LayerNorm(self.methylation_dim),
                nn.Linear(self.methylation_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.methylation_dim > 0
            else None
        )
        fusion_input_dim = d_model * (
            2 + int(self.rna_dim > 0) + int(self.cnv_dim > 0) + int(self.methylation_dim > 0) + int(self.demographic_dim > 0)
        )
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.exposure_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.confidence_head = nn.Sequential(nn.LayerNorm(d_model + 2), nn.Linear(d_model + 2, 1))

    def forward(
        self,
        sample_profile: torch.Tensor,
        reference_signatures: torch.Tensor,
        rna_embedding: torch.Tensor = None,
        cnv_embedding: torch.Tensor = None,
        methylation_embedding: torch.Tensor = None,
        demographic_vector: torch.Tensor = None,
        known_exposures: torch.Tensor = None,
    ):
        batch_size, num_channels = sample_profile.shape
        if num_channels != self.num_channels:
            raise ValueError("Expected {0} channels, got {1}".format(self.num_channels, num_channels))
        depth = torch.clamp(sample_profile.sum(dim=1, keepdim=True), min=1.0)
        normalized_input = sample_profile / depth

        channel_ids = torch.arange(self.num_channels, device=sample_profile.device).unsqueeze(0).expand(batch_size, -1)
        mutation_tokens = self.value_projection(normalized_input.unsqueeze(-1)) + self.channel_embedding(channel_ids)
        mutation_tokens = self.mutation_encoder(mutation_tokens)

        batch_size, num_signatures, _ = reference_signatures.shape
        signature_ids = torch.arange(num_signatures, device=sample_profile.device).unsqueeze(0).expand(batch_size, -1)
        signature_tokens = self.signature_projection(reference_signatures) + self.signature_index_embedding(signature_ids)
        if known_exposures is not None:
            signature_tokens = signature_tokens + self.signature_exposure_projection(known_exposures.unsqueeze(-1))
        signature_tokens = self.signature_encoder(signature_tokens)

        pooled_features = [mutation_tokens.mean(dim=1), signature_tokens.mean(dim=1)]
        if self.rna_projection is not None and rna_embedding is not None:
            pooled_features.append(self.rna_projection(rna_embedding))
        if self.cnv_projection is not None and cnv_embedding is not None:
            pooled_features.append(self.cnv_projection(cnv_embedding))
        if self.methylation_projection is not None and methylation_embedding is not None:
            pooled_features.append(self.methylation_projection(methylation_embedding))
        if self.demographic_projection is not None and demographic_vector is not None:
            pooled_features.append(self.demographic_projection(demographic_vector))
        fused_context = self.fusion_mlp(torch.cat(pooled_features, dim=-1)).unsqueeze(1)

        refined_signature_tokens = signature_tokens + fused_context
        exposure_logits = self.exposure_head(refined_signature_tokens).squeeze(-1)
        exposures = torch.softmax(exposure_logits, dim=-1)
        reconstruction = torch.bmm(exposures.unsqueeze(1), reference_signatures).squeeze(1)
        reconstruction = reconstruction / torch.clamp(reconstruction.sum(dim=1, keepdim=True), min=1e-12)

        residual_score = 1.0 - F.cosine_similarity(reconstruction, normalized_input, dim=1, eps=1e-8)
        confidence_features = torch.cat(
            [
                refined_signature_tokens.detach(),
                torch.log(depth).unsqueeze(1).expand(-1, refined_signature_tokens.shape[1], -1),
                residual_score.unsqueeze(1).unsqueeze(-1).expand(-1, refined_signature_tokens.shape[1], -1),
            ],
            dim=-1,
        )
        confidence = torch.sigmoid(self.confidence_head(confidence_features).squeeze(-1))
        return {
            "exposure_logits": exposure_logits,
            "exposures": exposures,
            "reconstruction": reconstruction,
            "confidence": confidence,
            "depth": depth.squeeze(-1),
        }


class LateFusionMultimodalRefitter(nn.Module):
    def __init__(
        self,
        num_channels: int,
        d_model: int,
        num_heads: int,
        num_sample_layers: int,
        num_reference_layers: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        rna_dim: int = 0,
        cnv_dim: int = 0,
        methylation_dim: int = 0,
        demographic_dim: int = 0,
        metadata_token_count: int = 2,
        rna_token_count: int = 4,
        cnv_token_count: int = 4,
        methylation_token_count: int = 4,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.rna_dim = int(rna_dim)
        self.cnv_dim = int(cnv_dim)
        self.methylation_dim = int(methylation_dim)
        self.demographic_dim = int(demographic_dim)
        self.metadata_token_count = int(metadata_token_count)
        self.rna_token_count = int(rna_token_count)
        self.cnv_token_count = int(cnv_token_count)
        self.methylation_token_count = int(methylation_token_count)

        self.channel_embedding = nn.Embedding(self.num_channels, d_model)
        self.value_projection = nn.Linear(1, d_model)
        self.signature_projection = nn.Linear(self.num_channels, d_model)
        self.signature_exposure_projection = nn.Linear(1, d_model)
        self.signature_index_embedding = nn.Embedding(512, d_model)
        self.mutation_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, num_sample_layers)
        self.signature_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, num_reference_layers)

        if self.rna_dim > 0:
            self.rna_projection = nn.Sequential(
                nn.LayerNorm(self.rna_dim),
                nn.Linear(self.rna_dim, d_model * self.rna_token_count),
                nn.GELU(),
                nn.Linear(d_model * self.rna_token_count, d_model * self.rna_token_count),
            )
            self.rna_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, 1)
        else:
            self.rna_projection = None
            self.rna_encoder = None

        if self.cnv_dim > 0:
            self.cnv_projection = nn.Sequential(
                nn.LayerNorm(self.cnv_dim),
                nn.Linear(self.cnv_dim, d_model * self.cnv_token_count),
                nn.GELU(),
                nn.Linear(d_model * self.cnv_token_count, d_model * self.cnv_token_count),
            )
            self.cnv_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, 1)
        else:
            self.cnv_projection = None
            self.cnv_encoder = None

        if self.methylation_dim > 0:
            self.methylation_projection = nn.Sequential(
                nn.LayerNorm(self.methylation_dim),
                nn.Linear(self.methylation_dim, d_model * self.methylation_token_count),
                nn.GELU(),
                nn.Linear(d_model * self.methylation_token_count, d_model * self.methylation_token_count),
            )
            self.methylation_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, 1)
        else:
            self.methylation_projection = None
            self.methylation_encoder = None

        if self.demographic_dim > 0:
            self.demographic_projection = nn.Sequential(
                nn.LayerNorm(self.demographic_dim),
                nn.Linear(self.demographic_dim, d_model * self.metadata_token_count),
                nn.GELU(),
                nn.Linear(d_model * self.metadata_token_count, d_model * self.metadata_token_count),
            )
            self.demographic_encoder = ModalityEncoder(d_model, num_heads, dropout, ff_multiplier, 1)
        else:
            self.demographic_projection = None
            self.demographic_encoder = None

        fusion_input_dim = d_model * (
            2 + int(self.rna_dim > 0) + int(self.cnv_dim > 0) + int(self.methylation_dim > 0) + int(self.demographic_dim > 0)
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.exposure_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.confidence_head = nn.Sequential(nn.LayerNorm(d_model + 2), nn.Linear(d_model + 2, 1))

    def forward(
        self,
        sample_profile: torch.Tensor,
        reference_signatures: torch.Tensor,
        rna_embedding: torch.Tensor = None,
        cnv_embedding: torch.Tensor = None,
        methylation_embedding: torch.Tensor = None,
        demographic_vector: torch.Tensor = None,
        known_exposures: torch.Tensor = None,
    ):
        batch_size, num_channels = sample_profile.shape
        if num_channels != self.num_channels:
            raise ValueError("Expected {0} channels, got {1}".format(self.num_channels, num_channels))
        depth = torch.clamp(sample_profile.sum(dim=1, keepdim=True), min=1.0)
        normalized_input = sample_profile / depth

        channel_ids = torch.arange(self.num_channels, device=sample_profile.device).unsqueeze(0).expand(batch_size, -1)
        mutation_tokens = self.value_projection(normalized_input.unsqueeze(-1)) + self.channel_embedding(channel_ids)
        mutation_tokens = self.mutation_encoder(mutation_tokens)

        _, num_signatures, _ = reference_signatures.shape
        signature_ids = torch.arange(num_signatures, device=sample_profile.device).unsqueeze(0).expand(batch_size, -1)
        signature_tokens = self.signature_projection(reference_signatures) + self.signature_index_embedding(signature_ids)
        if known_exposures is not None:
            signature_tokens = signature_tokens + self.signature_exposure_projection(known_exposures.unsqueeze(-1))
        signature_tokens = self.signature_encoder(signature_tokens)

        pooled_features = [mutation_tokens.mean(dim=1), signature_tokens.mean(dim=1)]
        if self.rna_projection is not None and rna_embedding is not None:
            rna_tokens = self.rna_projection(rna_embedding).reshape(batch_size, self.rna_token_count, -1)
            rna_tokens = self.rna_encoder(rna_tokens)
            pooled_features.append(rna_tokens.mean(dim=1))
        if self.cnv_projection is not None and cnv_embedding is not None:
            cnv_tokens = self.cnv_projection(cnv_embedding).reshape(batch_size, self.cnv_token_count, -1)
            cnv_tokens = self.cnv_encoder(cnv_tokens)
            pooled_features.append(cnv_tokens.mean(dim=1))
        if self.methylation_projection is not None and methylation_embedding is not None:
            methylation_tokens = self.methylation_projection(methylation_embedding).reshape(
                batch_size, self.methylation_token_count, -1
            )
            methylation_tokens = self.methylation_encoder(methylation_tokens)
            pooled_features.append(methylation_tokens.mean(dim=1))
        if self.demographic_projection is not None and demographic_vector is not None:
            demographic_tokens = self.demographic_projection(demographic_vector).reshape(
                batch_size, self.metadata_token_count, -1
            )
            demographic_tokens = self.demographic_encoder(demographic_tokens)
            pooled_features.append(demographic_tokens.mean(dim=1))

        context = self.context_projection(torch.cat(pooled_features, dim=-1)).unsqueeze(1).expand(-1, num_signatures, -1)
        exposure_logits = self.exposure_head(torch.cat([signature_tokens, context], dim=-1)).squeeze(-1)
        exposures = torch.softmax(exposure_logits, dim=-1)
        reconstruction = torch.bmm(exposures.unsqueeze(1), reference_signatures).squeeze(1)
        reconstruction = reconstruction / torch.clamp(reconstruction.sum(dim=1, keepdim=True), min=1e-12)

        residual_score = 1.0 - F.cosine_similarity(reconstruction, normalized_input, dim=1, eps=1e-8)
        confidence_features = torch.cat(
            [
                (signature_tokens + context).detach(),
                torch.log(depth).unsqueeze(1).expand(-1, num_signatures, -1),
                residual_score.unsqueeze(1).unsqueeze(-1).expand(-1, num_signatures, -1),
            ],
            dim=-1,
        )
        confidence = torch.sigmoid(self.confidence_head(confidence_features).squeeze(-1))
        return {
            "exposure_logits": exposure_logits,
            "exposures": exposures,
            "reconstruction": reconstruction,
            "confidence": confidence,
            "depth": depth.squeeze(-1),
        }


class MLPExposureRefitter(nn.Module):
    def __init__(
        self,
        num_channels: int,
        d_model: int,
        num_heads: int,
        num_sample_layers: int,
        num_reference_layers: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        rna_dim: int = 0,
        cnv_dim: int = 0,
        methylation_dim: int = 0,
        demographic_dim: int = 0,
    ) -> None:
        del num_heads, num_sample_layers, num_reference_layers
        super().__init__()
        self.num_channels = int(num_channels)
        self.rna_dim = int(rna_dim)
        self.cnv_dim = int(cnv_dim)
        self.methylation_dim = int(methylation_dim)
        self.demographic_dim = int(demographic_dim)

        feature_dim = self.num_channels + self.rna_dim + self.cnv_dim + self.methylation_dim + self.demographic_dim
        self.sample_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
            nn.GELU(),
        )
        self.signature_projection = nn.Linear(self.num_channels, d_model)
        self.signature_exposure_projection = nn.Linear(1, d_model)
        self.exposure_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.confidence_head = nn.Sequential(nn.LayerNorm(d_model + 2), nn.Linear(d_model + 2, 1))

    def forward(
        self,
        sample_profile: torch.Tensor,
        reference_signatures: torch.Tensor,
        rna_embedding: torch.Tensor = None,
        cnv_embedding: torch.Tensor = None,
        methylation_embedding: torch.Tensor = None,
        demographic_vector: torch.Tensor = None,
        known_exposures: torch.Tensor = None,
    ):
        batch_size, num_channels = sample_profile.shape
        if num_channels != self.num_channels:
            raise ValueError("Expected {0} channels, got {1}".format(self.num_channels, num_channels))
        depth = torch.clamp(sample_profile.sum(dim=1, keepdim=True), min=1.0)
        normalized_input = sample_profile / depth

        feature_parts = [normalized_input]
        if self.rna_dim > 0:
            if rna_embedding is None:
                raise ValueError("MLPExposureRefitter expected RNA embeddings because rna_dim > 0.")
            feature_parts.append(rna_embedding)
        if self.cnv_dim > 0:
            if cnv_embedding is None:
                raise ValueError("MLPExposureRefitter expected CNV embeddings because cnv_dim > 0.")
            feature_parts.append(cnv_embedding)
        if self.methylation_dim > 0:
            if methylation_embedding is None:
                raise ValueError("MLPExposureRefitter expected methylation embeddings because methylation_dim > 0.")
            feature_parts.append(methylation_embedding)
        if self.demographic_dim > 0:
            if demographic_vector is None:
                raise ValueError("MLPExposureRefitter expected demographic vectors because demographic_dim > 0.")
            feature_parts.append(demographic_vector)
        sample_context = self.sample_mlp(torch.cat(feature_parts, dim=-1))

        signature_tokens = self.signature_projection(reference_signatures)
        if known_exposures is not None:
            signature_tokens = signature_tokens + self.signature_exposure_projection(known_exposures.unsqueeze(-1))
        num_signatures = signature_tokens.shape[1]
        expanded_context = sample_context.unsqueeze(1).expand(-1, num_signatures, -1)
        exposure_logits = self.exposure_head(torch.cat([signature_tokens, expanded_context], dim=-1)).squeeze(-1)
        exposures = torch.softmax(exposure_logits, dim=-1)
        reconstruction = torch.bmm(exposures.unsqueeze(1), reference_signatures).squeeze(1)
        reconstruction = reconstruction / torch.clamp(reconstruction.sum(dim=1, keepdim=True), min=1e-12)

        residual_score = 1.0 - F.cosine_similarity(reconstruction, normalized_input, dim=1, eps=1e-8)
        confidence_features = torch.cat(
            [
                expanded_context.detach(),
                torch.log(depth).unsqueeze(1).expand(-1, num_signatures, -1),
                residual_score.unsqueeze(1).unsqueeze(-1).expand(-1, num_signatures, -1),
            ],
            dim=-1,
        )
        confidence = torch.sigmoid(self.confidence_head(confidence_features).squeeze(-1))
        return {
            "exposure_logits": exposure_logits,
            "exposures": exposures,
            "reconstruction": reconstruction,
            "confidence": confidence,
            "depth": depth.squeeze(-1),
        }


def build_model_from_config(
    config,
    num_channels: int,
    rna_dim: int = 0,
    cnv_dim: int = 0,
    methylation_dim: int = 0,
    demographic_dim: int = 0,
):
    model_cfg = config.get("model", {})
    architecture = str(model_cfg.get("architecture", "sigformer")).strip().lower()
    demographic_mode = _resolve_demographic_mode(model_cfg.get("demographic_conditioning", "both"))
    common_kwargs = {
        "num_channels": int(num_channels),
        "d_model": int(model_cfg["d_model"]),
        "num_heads": int(model_cfg["num_heads"]),
        "num_sample_layers": int(model_cfg.get("num_sample_layers", 2)),
        "num_reference_layers": int(model_cfg.get("num_reference_layers", 4)),
        "ff_multiplier": int(model_cfg.get("ff_multiplier", 4)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "rna_dim": int(rna_dim),
        "cnv_dim": int(cnv_dim),
        "methylation_dim": int(methylation_dim),
        "demographic_dim": int(demographic_dim),
    }
    if architecture in {"sigformer", "legacy_sigformer"}:
        return SigFormerRefitter(**common_kwargs)
    if architecture in {"concat_multimodal_transformer", "concat_fusion_multimodal_transformer"}:
        return ConcatFusionMultimodalRefitter(**common_kwargs)
    if architecture in {"late_fusion_multimodal_transformer", "late_fusion_transformer"}:
        return LateFusionMultimodalRefitter(
            **common_kwargs,
            metadata_token_count=int(model_cfg.get("metadata_token_count", 2)),
            rna_token_count=int(model_cfg.get("rna_token_count", 4)),
            cnv_token_count=int(model_cfg.get("cnv_token_count", 4)),
            methylation_token_count=int(model_cfg.get("methylation_token_count", 4)),
        )
    if architecture in {"mlp_refitter", "mlp_exposure_refitter"}:
        return MLPExposureRefitter(**common_kwargs)
    if architecture in {"residual_multimodal_transformer", "residual_aware_multimodal_transformer"}:
        return ResidualAwareMultimodalTransformer(
            **common_kwargs,
            latent_token_count=int(model_cfg.get("latent_token_count", 16)),
            num_candidate_embeddings=int(model_cfg.get("num_candidate_embeddings", 4)),
            metadata_token_count=int(model_cfg.get("metadata_token_count", 2)),
            rna_token_count=int(model_cfg.get("rna_token_count", 4)),
            cnv_token_count=int(model_cfg.get("cnv_token_count", 4)),
            methylation_token_count=int(model_cfg.get("methylation_token_count", 4)),
            use_residual_branch=_config_bool(model_cfg.get("use_residual_branch"), True),
            use_cross_branch_fusion=_config_bool(model_cfg.get("use_cross_branch_fusion"), True),
            use_demographic_tokens=_config_bool(model_cfg.get("demographic_tokens"), True),
            use_demographic_conditioning=demographic_mode in {"adaln", "both"},
            use_active_head=_config_bool(model_cfg.get("use_active_head"), False),
        )
    raise ValueError(
        "Unsupported model.architecture value '{0}'. Use 'sigformer', 'concat_multimodal_transformer', 'late_fusion_multimodal_transformer', 'mlp_refitter', or 'residual_multimodal_transformer'.".format(
            architecture
        )
    )
