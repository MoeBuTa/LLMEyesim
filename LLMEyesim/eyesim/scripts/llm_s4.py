#!/Users/wenxiao/miniconda3/envs/llmeyesim/bin/python

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.llm.agents.executive_agent import ExecutiveAgent
from LLMEyesim.integration.embodied_agent import EmbodiedAgent

if __name__ == "__main__":
        agent = ExecutiveAgent(llm_name='gpt-4o-mini', llm_type="cloud")
        actuator = RobotActuator(robot_id=2, robot_name='S4')

        embodied_agent = EmbodiedAgent(agent, actuator)
        embodied_agent.run_agent()
