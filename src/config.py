from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    last_key_by_indent: dict[int, str] = {}

    lines = text.splitlines()
    for line_idx, raw_line in enumerate(lines):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            item_text = line[2:].strip()
            if not isinstance(parent, list):
                raise ValueError("Fallback YAML parser expected a list parent.")
            if ":" in item_text:
                key, value = item_text.split(":", 1)
                item: Dict[str, Any] = {key.strip(): _parse_scalar(value)}
                parent.append(item)
                stack.append((indent, item))
            else:
                parent.append(_parse_scalar(item_text))
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_container: Any = {}
        parent[key] = next_container
        last_key_by_indent[indent] = key
        stack.append((indent, next_container))

        # Convert empty mapping to list lazily when the next non-comment child is "-".
        # This small parser is intended only for repo configs used in lightweight smoke runs.
        for upcoming in lines[line_idx + 1 :]:
            if not upcoming.strip() or upcoming.lstrip().startswith("#"):
                continue
            upcoming_indent = len(upcoming) - len(upcoming.lstrip(" "))
            if upcoming_indent <= indent:
                break
            if upcoming.strip().startswith("- "):
                parent[key] = []
                stack[-1] = (indent, parent[key])
            break
    return root


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _format_string_templates(value: Any, timestamp: str) -> Any:
    if isinstance(value, dict):
        return {key: _format_string_templates(item, timestamp) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_string_templates(item, timestamp) for item in value]
    if isinstance(value, str) and "{timestamp}" in value:
        return value.format(timestamp=timestamp)
    return value


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        config = yaml.safe_load(text) or {}
    else:
        config = _simple_yaml_load(text)

    base_path = config.get("base_config")
    if base_path:
        parent_path = (config_path.parent / base_path).resolve()
        base_config = load_config(str(parent_path))
        config = _deep_update(base_config, config)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = _format_string_templates(config, run_timestamp)

    config["run_timestamp"] = run_timestamp
    config["config_path"] = str(config_path.resolve())
    return config
