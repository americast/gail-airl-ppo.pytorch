import os
from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Trainer:

    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()

        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_return = 0.0
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))


class Trainer_multi:

    def __init__(self, env_1, env_2, env_multi, env_test_1, env_test_2, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env_1 = env_1
        self.env_2 = env_2
        self.env_multi = env_multi
        self.env_1.seed(seed)
        self.env_2.seed(seed)
        self.env_multi.seed(seed)

        # Env for evaluation.
        self.env_test_1 = env_test_1
        self.env_test_1.seed(2**31-seed)
        self.env_test_2 = env_test_2
        self.env_test_2.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state_1 = self.env_1.reset()
        state_2 = self.env_2.reset()

        for step in range(1, self.num_steps + 1):
            print(step, end="\r")
            # Pass to the algorithm to update state and episode timestep.
            state_1, state_2, t = self.algo.step(self.env_1, state_1, self.env_2, state_2, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                print("\n\n")
                self.evaluate(step)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            state_1 = self.env_test_1.reset()
            state_2 = self.env_test_2.reset()
            episode_return = 0.0
            done_1 = done_2 = done = False

            while (not done):
                state_multi = np.concatenate([state_1, state_2], axis=-1)
                action_multi = self.algo.exploit(state_multi)
                action_1 = action_multi[:3]
                action_2 = action_multi[3:]
                reward_1 = reward_2 = 0
                if not done_1:
                    state_1, reward_1, done_1, _ = self.env_test_1.step(action_1)
                    episode_return += 0.5 * reward_1
                if not done_2:
                    state_2, reward_2, done_2, _ = self.env_test_2.step(action_2)
                    episode_return += 0.5 * reward_2
                if done_1 and done_2: done = True

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
