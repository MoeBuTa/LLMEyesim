import csv
from pathlib import Path
import shutil
from typing import Any, Dict

from loguru import logger
import pandas as pd

from LLMEyesim.eyesim.utils.models import TaskPaths
from LLMEyesim.utils.constants import DATA_DIR


class TaskManager:
    def __init__(self, task_name: str):
        self.paths = self._init_directory(task_name)
        self.task_name = self.paths.task_name
        self.img_path = self.paths.img_path
        self.task_path = self.paths.task_path
        self.state_path = self.paths.state_path
        self.llm_reasoning_record_path = self.paths.llm_reasoning_record_path
        self.llm_action_record_path = self.paths.llm_action_record_path



    @staticmethod
    def _init_directory(task_name: str) -> TaskPaths:
        """Initialize directory structure for the task."""
        task_name = task_name
        task_path = DATA_DIR / task_name
        img_path = task_path / "images"

        # Create directories
        img_path.mkdir(parents=True, exist_ok=True)

        return TaskPaths(
            task_name=task_name,
            img_path=img_path,
            task_path=task_path,
            state_path=task_path / "robot_state.csv",
            llm_reasoning_record_path=task_path / "llm_reasoning_record.csv",
            llm_action_record_path=task_path / "llm_action_record.csv"
        )

    def data_collection(self, current_state: Dict[str, Any]) -> None:
        """Collect and save robot operation data."""
        logger.info("Data collection started!")
        try:
            self.save_item_to_csv(item=current_state, file_path=str(self.state_path))
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            raise



    @staticmethod
    def save_item_to_csv(item: Dict[str, Any], file_path: str) -> None:
        """Save dictionary item to CSV file."""
        try:
            with Path(file_path).open('a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=item.keys())
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(item)
        except IOError as e:
            logger.error(f"Error writing to CSV {file_path}: {e}")
            raise

    @staticmethod
    def move_directory_contents(src: Path | str, dst: Path | str) -> None:
        """Move directory contents from source to destination."""
        src_path = Path(src)
        dst_path = Path(dst)

        try:
            dst_path.mkdir(parents=True, exist_ok=True)

            for item in src_path.iterdir():
                target = dst_path / item.name
                shutil.move(str(item), str(target))

            src_path.rmdir()
        except OSError as e:
            logger.error(f"Error moving directory contents: {e}")
            raise


    def load_data_from_csv(self) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            csv_path = DATA_DIR / f"{self.task_name}.csv"
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise

    def robot_state_path(self, step: int) -> Dict[str, str]:
        paths = {
            "img": f"{self.img_path}/{step}.png",
            "lidar": f"{self.img_path}/{step}_lidar.png"
        }
        return paths


