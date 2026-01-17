"""Configuration file loading utilities."""

from pathlib import Path
from typing import Dict, Any, Type, TypeVar
import json
import yaml


ConfigType = TypeVar('ConfigType')


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file.
    
    Args:
        path: Path to configuration file
    
    Returns:
        Dictionary containing configuration parameters
    
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If file format is not supported
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open() as f:
        if path.suffix == ".json":
            return json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {path.suffix}. "
                f"Supported formats: .json, .yaml, .yml"
            )


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Values in `override` take precedence over values in `base`.
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
    
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override value
            result[key] = value
    
    return result


def config_from_dict(
    data: Dict[str, Any],
    config_class: Type[ConfigType]
) -> ConfigType:
    """Create configuration dataclass from dictionary.
    
    Only includes fields that are defined in the dataclass.
    Converts string paths to Path objects where appropriate.
    
    Args:
        data: Dictionary with configuration parameters
        config_class: Dataclass type to instantiate
    
    Returns:
        Instance of config_class
    """
    # Get valid field names from dataclass
    if not hasattr(config_class, '__dataclass_fields__'):
        raise ValueError(f"{config_class} is not a dataclass")
    
    valid_fields = {f.name for f in config_class.__dataclass_fields__.values()}
    
    # Filter to only include valid fields
    filtered_data = {}
    for key, value in data.items():
        if key in valid_fields:
            # Convert string paths to Path objects for specific fields
            field_type = config_class.__dataclass_fields__[key].type
            if 'Path' in str(field_type) and isinstance(value, str):
                filtered_data[key] = Path(value)
            elif 'List[Path]' in str(field_type) and isinstance(value, list):
                filtered_data[key] = [Path(v) if isinstance(v, str) else v for v in value]
            else:
                filtered_data[key] = value
    
    return config_class(**filtered_data)


def save_config(config: Any, path: Path) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary or dataclass instance
        path: Output path for JSON file
    """
    from dataclasses import asdict, is_dataclass
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict if needed
    if is_dataclass(config) and not isinstance(config, type):
        config_dict = asdict(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise TypeError(f"Expected dict or dataclass, got {type(config)}")
    
    # Convert Path objects to strings for JSON serialization
    config_serializable = _make_serializable(config_dict)
    
    with path.open('w') as f:
        json.dump(config_serializable, f, indent=2)


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format.
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON-serializable version of object
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    else:
        return obj
