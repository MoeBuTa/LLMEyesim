import argparse

from LLMEyeSim.utils.constants import DATA_DIR
from LLMEyeSim.simulation.simulator import Simulator

from loguru import logger

def get_numbered_task_name(task: str) -> str:
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the robot with different prompts.')
    parser.add_argument('model', type=str, default="gpt-4o-mini", choices=['gpt-4o', 'gpt-4o-mini'])
    parser.add_argument("attack", type=str, default="none", choices=['none', 'naive', 'image', 'noise'])
    parser.add_argument("defence", type=str, default="none", choices=['none', 'agent', 'self'])
    parser.add_argument("attack_rate", type=float, default=0.5, choices=[0.1, 0.3, 0.5, 0.7, 1])

    args = parser.parse_args()
    attack = args.attack
    model = args.model
    defence = args.defence
    attack_rate = args.attack_rate

    if defence == "none":
        defence = False
    else:
        defence = True

    task_name = get_numbered_task_name(f"{model}_{attack}_{defence}_{attack_rate}")

    simulator = Simulator(task_name=task_name,
                          attack=attack,
                          agent_name=model,
                          agent_type="cloud",
                          attack_rate=attack_rate,
                          enable_security=defence)
    simulator.run()
