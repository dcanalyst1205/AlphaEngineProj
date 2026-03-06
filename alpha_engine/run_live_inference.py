"""
alpha_engine/run_live_inference.py — Daily signal loop for March 2026.
"""

import logging
import yaml
import time
from pathlib import Path
import pandas as pd
from alpha_engine.main import run_pipeline

def main():
    config_path = Path("alpha_engine/config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("LiveInference")

    logger.info("Starting live inference loop for March 2026...")
    
    try:
        # Run the full pipeline to generate latest signals
        run_pipeline(config, str(config_path))
        logger.info("Daily signal cycle completed successfully.")
    except Exception as e:
        logger.error(f"Live inference loop failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
