import yaml
import os

def load_yaml(file_path: str) -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found at {file_path}")
    
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_yaml(config: dict, file_path: str):
    """Save configuration to a YAML file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
