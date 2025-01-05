from pathlib import Path

from loguru import logger
import yaml


def load_config(config_path):
    try:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        logger.warning(
            f"Config file not found at {config_path}. Using default configuration."
        )
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        return {}


# Determine the project root directory
PROJECT_DIR = Path(__file__).resolve().parents[1].resolve().parents[0]
logger.info(f"PROJECT_DIR: {PROJECT_DIR}")

SCRIPT_DIR = PROJECT_DIR / "LLMEyesim" / "eyesim" / "scripts"
WORLD_DIR = PROJECT_DIR / "LLMEyesim" / "eyesim" / "worlds"



EXP_DIR = PROJECT_DIR / "experiment"
DATA_DIR = PROJECT_DIR / "experiment" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EYESIM_DIR = "/opt/eyesim/eyesimX"
# Load configuration
config = load_config(PROJECT_DIR / "config.yml")


OPENAI_API_KEY = config["openai"]["api_key"]
