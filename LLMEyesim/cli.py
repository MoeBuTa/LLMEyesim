import argparse
import subprocess
import sys
import time
from typing import Any, Dict, List, Union

from loguru import logger

from LLMEyesim.eyesim.world_generator.manager import WorldManager
from LLMEyesim.simulation.simulator import Simulator
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


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description='Run the robot with different prompts.')

    # World environment selection
    parser.add_argument(
        'world',
        type=str,
        default="demo",
        choices=['free', 'static', 'dynamic', 'mixed', 'demo'],
        help='Select the world environment type'
    )

    # Model selection
    parser.add_argument(
        'model',
        type=str,
        default="gpt-4o-mini",
        choices=['gpt-4o', 'gpt-4o-mini'],
        help='Select the model to use'
    )

    # Attack type
    parser.add_argument(
        "attack",
        type=str,
        default="none",
        choices=['none', 'ghi', 'omi'],
        help='Select the type of attack to use'
    )

    # Defence flag
    parser.add_argument(
        "defence",
        type=str2bool,
        default=False,
        help='Enable defence mode (true/false/yes/no/1/0)'
    )

    # Attack rate
    parser.add_argument(
        "attack_rate",
        type=float_in_list([0.1, 0.3, 0.5, 0.7, 1.0]),
        default=0.5,
        help='Set the attack rate (0.1, 0.3, 0.5, 0.7, or 1.0)'
    )

    return parser


def launch_eyesim() -> int:
    """Launch the eyesim simulator."""
    try:
        # Run the eyesim command
        process = subprocess.Popen("eyesim", shell=True)
        # Wait for the process to complete
        process.wait()
        return process.returncode

    except subprocess.CalledProcessError as e:
        print(f"Error launching eyesim: {e}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print("Error: 'eyesim' command not found. Make sure it's installed and in your PATH", file=sys.stderr)
        return 1



def setup_simulation(args: Dict[str, Any]) -> Simulator:
    """Initialize and configure the simulation."""
    world = args.get("world", "demo")
    attack = args.get("attack", "none")
    model = args.get("model", "gpt-4o-mini")
    defence = args.get("defence", False)
    attack_rate = args.get("attack_rate", 0.5)

    try:
        world_manager = WorldManager(world)
        world_manager.generate_sim()
    except Exception as e:
        logger.error(f"Failed to generate world: {e}")
        raise
    launch_eyesim()
    time.sleep(5)
    task_name = set_task_name(f"{world}_{model}_{attack}_{defence}_{attack_rate}")
    simulator = Simulator(task_name=task_name,
                          attack=attack,
                          agent_name=model,
                          agent_type="cloud",
                          attack_rate=attack_rate,
                          enable_defence=defence)
    return simulator


def main() -> None:
    """Main entry point for the simulation."""
    try:
        parser = create_parser()
        args = parser.parse_args()
        simulator = setup_simulation(vars(args))
        simulator.run()
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise
