import os
import json
from pathlib import Path

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            # Look for config in parent directory
            config_path = Path(__file__).parent.parent / 'config' / 'config.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    @property
    def model_path(self):
        model_path = self.config['model']['path']
        # Convert relative path to absolute
        if not os.path.isabs(model_path):
            return Path(__file__).parent.parent / model_path
        return Path(model_path)
    
    @property
    def class_names(self):
        return self.config['model']['classes']
    
    @property
    def input_shape(self):
        return tuple(self.config['model']['input_shape'])
    
    @property
    def target_size(self):
        return tuple(self.config['preprocessing']['target_size'])
    
    @property
    def rescale_factor(self):
        return self.config['preprocessing']['rescale']
    
    @property
    def api_host(self):
        return self.config['api']['host']
    
    @property
    def api_port(self):
        return self.config['api']['port']
    
    @property
    def debug(self):
        return self.config['api']['debug']
