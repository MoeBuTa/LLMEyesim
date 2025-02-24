#!/Users/wenxiao/miniconda3/envs/llmeyesim/bin/python

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.llm.agents.agent import ExecutiveAgent
from LLMEyesim.integration.agent import EmbodiedAgent

if __name__ == '__main__':
    
    world_items = [WorldItem(item_id=1, item_name='S4', item_type='robot', x=200, y=267, angle=0), WorldItem(item_id=2, item_name='Can', item_type='target', x=2000, y=3667, angle=90), WorldItem(item_id=3, item_name='Can', item_type='target', x=333, y=2000, angle=90), WorldItem(item_id=4, item_name='Can', item_type='target', x=3667, y=2000, angle=90), WorldItem(item_id=5, item_name='Can', item_type='target', x=2000, y=333, angle=90), WorldItem(item_id=6, item_name='Crate1', item_type='obstacle', x=2520, y=3720, angle=90), WorldItem(item_id=7, item_name='Crate1', item_type='obstacle', x=1080, y=3000, angle=90), WorldItem(item_id=8, item_name='Crate1', item_type='obstacle', x=3480, y=3000, angle=90), WorldItem(item_id=9, item_name='Crate1', item_type='obstacle', x=2760, y=2280, angle=90), WorldItem(item_id=10, item_name='Crate1', item_type='obstacle', x=3240, y=1800, angle=90), WorldItem(item_id=11, item_name='Crate1', item_type='obstacle', x=1320, y=1560, angle=90), WorldItem(item_id=12, item_name='Crate1', item_type='obstacle', x=2760, y=1320, angle=90), WorldItem(item_id=13, item_name='Crate1', item_type='obstacle', x=600, y=840, angle=90), WorldItem(item_id=14, item_name='Crate1', item_type='obstacle', x=2280, y=600, angle=90)]
    agent = ExecutiveAgent(llm_name='gpt-4o', llm_type="cloud")
    actuator = RobotActuator(robot_id=1, robot_name='S4')
    embodied_agent = EmbodiedAgent(agent, actuator, world_items)
    embodied_agent.run_agent()
