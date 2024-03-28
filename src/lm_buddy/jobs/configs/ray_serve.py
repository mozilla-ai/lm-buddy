
from lm_buddy.types import BaseLMBuddyConfig
from typing import List


class RayServeActorConfig(BaseLMBuddyConfig):
    num_cpus: float
    num_gpus: float
class RayServeDeployConfig(BaseLMBuddyConfig):
    name: str
    num_replicas: int
    ray_actor_options: List[RayServeActorConfig]
class RayServeRuntimeConfig(BaseLMBuddyConfig):
    working_dir: str
    pip: list
class RayServeConfig(BaseLMBuddyConfig):
    name:str
    route_prefix: str
    import_path: str
    runtime_env: List[RayServeRuntimeConfig]
    deployments: List[RayServeDeployConfig]



