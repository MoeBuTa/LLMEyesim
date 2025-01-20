from loguru import logger

from LLMEyesim.eyesim.generator.base import WorldGenerator


class GridWorld(WorldGenerator):
    def __init__(self, world_name: str, llm_name: str = "gpt-4o-mini"):
        super().__init__(world_name=world_name, llm_name=llm_name)

    def init_sim(self, **kwargs):
        # TODO: Try implementing a grid world if demo world is failed

        logger.info("GridWorld simulation generation complete")
