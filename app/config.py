import os
from pathlib import Path

class Settings:
    # Base directory of the project
    BASE_DIR = Path(__file__).parent.parent.resolve()
    
    # Path configuration
    MODEL_PATH = str(BASE_DIR / "app" / "model" / "best (2).pt")
    DATA_PATH = str(BASE_DIR / "data")
    
    # Ultralytics configuration
    ULTRALYTICS_CONFIG = {
        "runs_dir": str(BASE_DIR / "runs"),
    }

settings = Settings()
