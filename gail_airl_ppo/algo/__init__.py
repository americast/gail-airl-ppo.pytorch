from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL
from .multi import MULTI

ALGOS = {
    'gail': GAIL,
    'airl': AIRL,
    'multi': MULTI,
}
