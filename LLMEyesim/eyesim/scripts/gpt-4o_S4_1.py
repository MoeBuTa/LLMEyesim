#!/Users/wenxiao/miniconda3/envs/llmeyesim/bin/python

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.llm.agents.agent import ExecutiveAgent
from LLMEyesim.integration.agent import EmbodiedAgent

if __name__ == '__main__':
    
    world_items = [WorldItem(item_id=1, item_name='S4', item_type='robot', x=300, y=400, angle=0), WorldItem(item_id=2, item_name='Can', item_type='target', x=3000, y=5500, angle=90), WorldItem(item_id=3, item_name='Can', item_type='target', x=500, y=3000, angle=90), WorldItem(item_id=4, item_name='Can', item_type='target', x=5500, y=3000, angle=90), WorldItem(item_id=5, item_name='Can', item_type='target', x=3000, y=500, angle=90), WorldItem(item_id=6, item_name='Crate1', item_type='obstacle', x=3780, y=5580, angle=90), WorldItem(item_id=7, item_name='Crate1', item_type='obstacle', x=4140, y=5220, angle=90), WorldItem(item_id=8, item_name='Crate1', item_type='obstacle', x=3420, y=4860, angle=90), WorldItem(item_id=9, item_name='Crate1', item_type='obstacle', x=1620, y=4500, angle=90), WorldItem(item_id=10, item_name='Crate1', item_type='obstacle', x=5220, y=4500, angle=90), WorldItem(item_id=11, item_name='Crate1', item_type='obstacle', x=1980, y=4140, angle=90), WorldItem(item_id=12, item_name='Crate1', item_type='obstacle', x=4860, y=4140, angle=90), WorldItem(item_id=13, item_name='Crate1', item_type='obstacle', x=4140, y=3420, angle=90), WorldItem(item_id=14, item_name='Crate1', item_type='obstacle', x=180, y=3060, angle=90), WorldItem(item_id=15, item_name='Crate1', item_type='obstacle', x=3780, y=3060, angle=90), WorldItem(item_id=16, item_name='Crate1', item_type='obstacle', x=1620, y=2700, angle=90), WorldItem(item_id=17, item_name='Crate1', item_type='obstacle', x=4860, y=2700, angle=90), WorldItem(item_id=18, item_name='Crate1', item_type='obstacle', x=180, y=2340, angle=90), WorldItem(item_id=19, item_name='Crate1', item_type='obstacle', x=1980, y=2340, angle=90), WorldItem(item_id=20, item_name='Crate1', item_type='obstacle', x=4500, y=2340, angle=90), WorldItem(item_id=21, item_name='Crate1', item_type='obstacle', x=180, y=1980, angle=90), WorldItem(item_id=22, item_name='Crate1', item_type='obstacle', x=4140, y=1980, angle=90), WorldItem(item_id=23, item_name='Crate1', item_type='obstacle', x=2340, y=1620, angle=90), WorldItem(item_id=24, item_name='Crate1', item_type='obstacle', x=900, y=1260, angle=90), WorldItem(item_id=25, item_name='Crate1', item_type='obstacle', x=3780, y=1260, angle=90), WorldItem(item_id=26, item_name='Crate1', item_type='obstacle', x=3420, y=900, angle=90), WorldItem(item_id=27, item_name='Crate1', item_type='obstacle', x=2340, y=540, angle=90)]
    agent = ExecutiveAgent(llm_name='gpt-4o', llm_type="cloud")
    actuator = RobotActuator(robot_id=1, robot_name='S4')
    embodied_agent = EmbodiedAgent(agent, actuator, world_items)
    embodied_agent.run_agent()
