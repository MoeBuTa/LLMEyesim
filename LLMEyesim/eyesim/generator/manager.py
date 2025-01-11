from LLMEyesim.eyesim.generator.base import WorldGenerator
from LLMEyesim.eyesim.generator.demo import DemoWorld
from LLMEyesim.eyesim.generator.dynamic import DynamicWorld
from LLMEyesim.eyesim.generator.free import FreeWorld
from LLMEyesim.eyesim.generator.mixed import MixedWorld
from LLMEyesim.eyesim.generator.static import StaticWorld


class WorldManager:
    def __init__(self, world_name: str, llm_name: str = "gpt-4o-mini"):
        self.world_type = {
            "dynamic": DynamicWorld,
            "free": FreeWorld,
            "static": StaticWorld,
            "mixed": MixedWorld,
            "demo": DemoWorld
        }
        self.world = self._init_world(world_name, llm_name)

    def _init_world(self, world_name: str, llm_name) -> WorldGenerator:
        world_name = world_name.lower()
        llm_name = llm_name.lower()
        if world_name not in self.world_type:
            raise NotImplementedError(
                f"Invalid world type. Must be one of: {', '.join(self.world_type.keys())}"
            )
        return self.world_type[world_name](world_name, llm_name)

    def init_sim(self, **kwargs):
        return self.world.init_sim(**kwargs)

    def get_world_info(self):
        return {
            "name": self.world.world_name,
            "type": type(self.world).__name__,
        }