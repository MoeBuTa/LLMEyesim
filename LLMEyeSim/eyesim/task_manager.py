import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Dict, List

from loguru import logger
import pandas as pd

from LLMEyeSim.eyesim.image_process import ImageProcess
from LLMEyeSim.utils.constants import DATA_DIR


@dataclass
class TaskPaths:
    task_name: str
    img_path: Path
    task_path: Path
    state_path: Path
    llm_reasoning_record_path: Path
    llm_action_record_path: Path


class TaskManager:
    def __init__(self, task_name: str):
        self.paths = self._init_directory(task_name)
        self.task_name = self.paths.task_name
        self.img_path = self.paths.img_path
        self.task_path = self.paths.task_path
        self.state_path = self.paths.state_path
        self.llm_reasoning_record_path = self.paths.llm_reasoning_record_path
        self.llm_action_record_path = self.paths.llm_action_record_path
        self.image_process = ImageProcess()

    def _init_directory(self, task_name: str) -> TaskPaths:
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

    def data_collection(self, current_state: Dict[str, Any], img: Any, scan: List[int]) -> None:
        """Collect and save robot operation data."""
        logger.info("Data collection started!")
        try:
            self.image_process.cam2image(img).save(current_state["img_path"])
            self.image_process.lidar2image(scan=list(scan), save_path=current_state["lidar_path"])
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