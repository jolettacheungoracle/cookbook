import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class ModelConfig:
    """Configuration class for model settings"""
    
    def __init__(self, config_dict: dict):
        self.name = config_dict["name"]
        self.display_name = config_dict["display_name"]
        self.type = config_dict["type"]  # "anthropic" or "openai_compatible"
        self.model = config_dict["model"]
        self.api_key_env = config_dict["api_key_env"]
        self.base_url = config_dict.get("base_url")
        self.max_tokens = config_dict.get("max_tokens", 1024)
        self.temperature = config_dict.get("temperature", 0.7)
        self.streaming = config_dict.get("streaming", True)
        self.supports_tools = config_dict.get("supports_tools", True)
        self.description = config_dict.get("description", "")
    
    def get_api_key(self) -> Optional[str]:
        """Get the API key from environment variables"""
        return os.getenv(self.api_key_env)
    
    def __str__(self):
        return f"{self.display_name} ({self.name})"

class ConfigLoader:
    """Loads and manages model configurations"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(__file__).parent / config_dir
        self.configs: Dict[str, ModelConfig] = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all JSON configuration files from the config directory"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                config = ModelConfig(config_data)
                self.configs[config.name] = config
            
            except Exception as e:
                print(f"Error loading config from {config_file}: {e}")
    
    def get_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.configs.get(model_name)

# Global instance for easy access
config_loader = ConfigLoader()

def get_current_model() -> str:
    """Get the currently selected model from environment variable"""
    return os.getenv("CURRENT_MODEL", "anthropic")

def get_model_config(model_name: str = None) -> Optional[ModelConfig]:
    """Get the configuration for the specified model or current model"""
    if model_name is None:
        model_name = get_current_model()
    return config_loader.get_config(model_name)
