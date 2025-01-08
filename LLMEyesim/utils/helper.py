import argparse
from typing import List, Union

from loguru import logger

from LLMEyesim.utils.constants import DATA_DIR


def str2bool(value: Union[str, bool]) -> bool:
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected')


def float_in_list(choices: List[float]) -> callable:
    """Create a validator for float values within a set of choices."""

    def check_float(value: str) -> float:
        try:
            float_val = float(value)
            if float_val in choices:
                return float_val
            raise argparse.ArgumentTypeError(
                f'Value must be one of {choices}, got {float_val}'
            )
        except ValueError:
            raise argparse.ArgumentTypeError(
                f'Value must be a float, got {value}'
            )

    return check_float


def set_task_name(task: str) -> str:
    """Generate numbered task folder name."""
    try:
        max_num = max(
            (int(folder.name.split("_")[-1])
             for folder in DATA_DIR.iterdir()
             if folder.name.startswith(task) and
             folder.name.split("_")[-1].isdigit()),
            default=0
        )
        return f"{task}_{max_num + 1}"
    except Exception as e:
        logger.error(f"Error generating task name: {e}")
        return f"{task}_1"