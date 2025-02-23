#!/Users/wenxiao/miniconda3/envs/llmeyesim/bin/python


from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.integration.agent import EmbodiedAgent
from LLMEyesim.llm.agents.agent import ExecutiveAgent

if __name__ == '__main__':

    agent = ExecutiveAgent(llm_name='gpt-4o-mini', llm_type="cloud")
    actuator = RobotActuator(robot_id=1, robot_name='LabBot')
    embodied_agent = EmbodiedAgent(agent, actuator)
    embodied_agent.run_agent()