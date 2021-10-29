import os
import argparse
from datetime import datetime
import torch
import pudb
from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer_multi
import pudb
import gym
import numpy as np

class BasicEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]), high=np.array([np.inf, np.inf, np.inf, np.inf]))
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]), high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))

def run(args):
    env_1 = make_env("Hopper-v3")
    env_2 = make_env("InvertedPendulum-v2")
    env_multi = BasicEnv()
    env_test_1 = make_env("Hopper-v3")
    env_test_2 = make_env("InvertedPendulum-v2")

    buffer_exp_hopper = SerializedBuffer(
        path="buffers/Hopper-v3/size1000000_std0.01_prand0.0.pth",
        device=torch.device("cuda" if args.cuda else "cpu")
    )

    buffer_exp_pendulum = SerializedBuffer(
        path="buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth",
        device=torch.device("cuda" if args.cuda else "cpu")
    )

    algo = ALGOS["multi"](
        buffer_exp_1=buffer_exp_hopper,
        buffer_exp_2=buffer_exp_pendulum,
        state_shape_1=env_1.observation_space.shape,
        state_shape_2=env_2.observation_space.shape,
        state_shape_multi=env_multi.observation_space.shape,
        action_shape_1=env_1.action_space.shape,
        action_shape_2=env_2.action_space.shape,
        action_shape_multi=env_multi.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', "multi", args.algo, f'seed{args.seed}-{time}')

    # pu.db
    trainer = Trainer_multi(
        env_1=env_1,
        env_2=env_2,
        env_multi=env_multi,
        env_test_1=env_test_1,
        env_test_2=env_test_2,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--eval_interval', type=int, default=10**5)
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
