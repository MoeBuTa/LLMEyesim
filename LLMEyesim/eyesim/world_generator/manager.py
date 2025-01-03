from LLMEyesim.eyesim.world_generator.base import WorldGenerator
from LLMEyesim.eyesim.world_generator.demo import DemoWorld
from LLMEyesim.eyesim.world_generator.dynamic import DynamicWorld
from LLMEyesim.eyesim.world_generator.free import FreeWorld
from LLMEyesim.eyesim.world_generator.mixed import MixedWorld
from LLMEyesim.eyesim.world_generator.static import StaticWorld


class WorldManager:
    def __init__(self, world_name: str):
        self.world_type = {
            "dynamic": DynamicWorld,
            "free": FreeWorld,
            "static": StaticWorld,
            "mixed": MixedWorld,
            "demo": DemoWorld
        }
        self.world = self._init_world(world_name)

    def _init_world(self, world_name: str) -> WorldGenerator:
        world_name = world_name.lower()
        if world_name not in self.world_type:
            raise NotImplementedError(
                f"Invalid world type. Must be one of: {', '.join(self.world_type.keys())}"
            )
        return self.world_type[world_name](world_name)

    def generate_sim(self):
        return self.world.generate_sim()