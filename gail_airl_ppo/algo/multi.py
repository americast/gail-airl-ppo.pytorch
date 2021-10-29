import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pudb
from .ppo import PPO_Multi
from gail_airl_ppo.network import MULTIDiscrim
import numpy as np

class MULTI(PPO_Multi):

    def __init__(self, buffer_exp_1, buffer_exp_2, state_shape_1, state_shape_2, state_shape_multi, action_shape_1, action_shape_2, action_shape_multi, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc_r=(100, 100), units_disc_v=(100, 100),
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape_1, state_shape_2, state_shape_multi, action_shape_1, action_shape_2, action_shape_multi, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp_1 = buffer_exp_1
        self.buffer_exp_2 = buffer_exp_2

        # Discriminator.
        self.disc = MULTIDiscrim(
            state_shape=state_shape_multi,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states_1, _, _, dones_1, log_pis_1, next_states_1 = \
                self.buffer_1.sample(self.batch_size)
            states_2, _, _, dones_2, log_pis_2, next_states_2 = \
                self.buffer_2.sample(self.batch_size)
            states_multi = torch.cat([states_1, states_2], axis=-1)
            next_states_multi = torch.cat([next_states_1, next_states_2], axis=-1)
            dones_multi = torch.bitwise_and(dones_1.int(),dones_2.int()).float()
            log_pis_multi = (log_pis_1 + log_pis_2)/2 #Verify
            # Samples from expert's demonstrations.
            states_exp_1, actions_exp_1, _, dones_exp_1, next_states_exp_1 = \
                self.buffer_exp_1.sample(self.batch_size)
            states_exp_2, actions_exp_2, _, dones_exp_2, next_states_exp_2 = \
                self.buffer_exp_2.sample(self.batch_size)
            states_exp_multi = torch.cat([states_exp_1, states_exp_2], axis=-1)
            actions_exp_multi = torch.cat([actions_exp_1, actions_exp_2], axis=-1)
            next_states_exp_multi = torch.cat([next_states_exp_1, next_states_exp_2], axis=-1)
            dones_exp_multi = torch.bitwise_and(dones_exp_1.int(),dones_exp_2.int()).float()
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(
                    states_exp_multi, actions_exp_multi)
            # Update discriminator.
            self.update_disc(
                states_multi, dones_multi, log_pis_multi, next_states_multi, states_exp_multi,
                dones_exp_multi, log_pis_exp, next_states_exp_multi, writer
            )

        # We don't use reward signals here,
        states_1, actions_1, _, dones_1, log_pis_1, next_states_1 = self.buffer_1.get()
        states_2, actions_2, _, dones_2, log_pis_2, next_states_2 = self.buffer_2.get()

        states_multi = torch.cat([states_1, states_2], axis=-1)
        actions_multi = torch.cat([actions_1, actions_2], axis=-1)
        next_states_multi = torch.cat([next_states_1, next_states_2], axis=-1)
        dones_multi = torch.bitwise_and(dones_1.int(),dones_2.int()).float()
        log_pis_multi = (log_pis_1 + log_pis_2)/2 #Verify
        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states_multi, dones_multi, log_pis_multi, next_states_multi)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states_multi, actions_multi, rewards, dones_multi, log_pis_multi, next_states_multi, writer)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
