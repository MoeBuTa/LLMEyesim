#!/Users/wenxiao/miniconda3/envs/llmeyesim/bin/python

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.llm.agents.executive_agent import ExecutiveAgent
from LLMEyesim.integration.embodied_agent import EmbodiedAgent

if __name__ == '__main__':
    
    world_items = [WorldItem(item_id=1, item_name='S4', item_type='robot', x=432, y=1659, angle=0), WorldItem(item_id=2, item_name='Can', item_type='target', x=1663, y=274, angle=90), WorldItem(item_id=3, item_name='Soccer', item_type='obstacle', x=229, y=1391, angle=90), WorldItem(item_id=4, item_name='Soccer', item_type='obstacle', x=1679, y=1525, angle=90), WorldItem(item_id=5, item_name='Soccer', item_type='obstacle', x=1600, y=700, angle=0)]
    agent = ExecutiveAgent(llm_name='gpt-4o-mini', llm_type="cloud")
    actuator = RobotActuator(robot_id=1, robot_name='S4')
    embodied_agent = EmbodiedAgent(agent, actuator, world_items)
    embodied_agent.run_agent()
