from LLMEyeSim.eyesim.environ_generator.dynamic_environ import DynamicEnviron
from LLMEyeSim.eyesim.environ_generator.free_environ import FreeEnviron
from LLMEyeSim.eyesim.environ_generator.static_dynamic_environ import (
    StaticDynamicEnviron,
)
from LLMEyeSim.eyesim.environ_generator.static_environ import StaticEnviron
from loguru import logger

if __name__ == '__main__':
    free_environ = FreeEnviron()
    free_environ.generate_random_sim()
    dynamic_environ = DynamicEnviron()
    dynamic_environ.generate_random_sim()
    static_environ = StaticEnviron()
    static_environ.generate_random_sim()
    static_dynamic_environ = StaticDynamicEnviron()
    static_dynamic_environ.generate_random_sim()
    logger.info("New Environ generated successfully!")
