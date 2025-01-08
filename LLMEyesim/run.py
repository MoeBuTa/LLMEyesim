import argparse
import subprocess
import sys
import time
from typing import Any, Dict

from loguru import logger

from LLMEyesim.eyesim.generator.manager import WorldManager
from LLMEyesim.simulation.simulator import Simulator
from LLMEyesim.simulation.simulator_v2 import SimulatorV2
from LLMEyesim.utils.constants import LOG_DIR
from LLMEyesim.utils.helper import float_in_list, set_task_name, str2bool

DEFAULT_CONFIG = {
    "mode": "2",
    "world": "demo",
    "model": "gpt-4o-mini",
    "attack": "none",
    "defence": False,
    "attack_rate": 0.5
}

def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description='Run the robot with different prompts.')

    parser.add_argument(
        '--mode',
        type=str,
        default=DEFAULT_CONFIG["mode"],
        choices=['1', '2'],
        help='Select the mode to run the simulation'
    )

    parser.add_argument(
        '--world',
        type=str,
        default=DEFAULT_CONFIG["world"],
        choices=['free', 'static', 'dynamic', 'mixed', 'demo'],
        help='Select the world environment type'
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_CONFIG["model"],
        choices=['gpt-4o', 'gpt-4o-mini'],
        help='Select the model to use'
    )

    # Attack type
    parser.add_argument(
        "--attack",
        type=str,
        default=DEFAULT_CONFIG["attack"],
        choices=['none', 'ghi', 'omi'],
        help='Select the type of attack to use'
    )

    # Defence flag
    parser.add_argument(
        "--defence",
        type=str2bool,
        default=DEFAULT_CONFIG["defence"],
        help='Enable defence mode (true/false/yes/no/1/0)'
    )

    # Attack rate
    parser.add_argument(
        "--attack_rate",
        type=float_in_list([0.1, 0.3, 0.5, 0.7, 1.0]),
        default=DEFAULT_CONFIG["attack_rate"],
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
    """Initialize and configure the simulation.

    Args:
        args: Dictionary of simulation parameters

    Returns:
        Simulator: Configured simulator instance
    """
    mode = args.get("mode", DEFAULT_CONFIG["mode"])
    world = args.get("world", DEFAULT_CONFIG["world"])
    attack = args.get("attack", DEFAULT_CONFIG["attack"])
    model = args.get("model", DEFAULT_CONFIG["model"])
    defence = args.get("defence", DEFAULT_CONFIG["defence"])
    attack_rate = args.get("attack_rate", DEFAULT_CONFIG["attack_rate"])

    try:
        world_manager = WorldManager(world)
        world_manager.init_sim()
    except Exception as e:
        logger.error(f"Failed to generate world: {e}")
        raise

    launch_eyesim()
    time.sleep(5)  # Wait for eyesim to launch, Adjust it as needed

    if mode == "1":
        task_name = set_task_name(f"{world}_{model}_{attack}")
        simulator = Simulator(
            task_name=task_name,
            attack=attack,
            llm_name=model,
            llm_type="cloud",
            enable_defence=defence,
            attack_rate=attack_rate,
            world_items=world_manager.world.items
        )

    else:
        simulator = SimulatorV2(
            mission_name=set_task_name(f"{world}_{model}_{attack}"),
            world_name=world,
            llm_name=model,
            llm_type="cloud",
            world_items=world_manager.world.items
        )
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
