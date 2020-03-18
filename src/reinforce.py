import torch
import numpy as np
import gym

from src.models.policy import GaussianPolicy, CategoricalPolicy
from torch.distributions import Normal, Categorical, OneHotCategorical
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

to_torch = lambda arr: torch.from_numpy(arr).float().to(device)
to_npy = lambda arr: arr.detach().double().cpu().numpy()


class Reinforce:

    def __init__(self, env, n_runs, gamma, horizon, epochs, lr=1e-3, continuous_policy=True, **kwargs):
        self.policy = GaussianPolicy(**kwargs) if continuous_policy else CategoricalPolicy(**kwargs)
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.env = env
        self.horizon = horizon
        self.act_dim = (env.action_space.n if len(env.action_space.shape) == 0 else env.action_space.shape[0]) if continuous_policy else 1
        self.obs_dim = env.observation_space.shape[0]
        self.gamma = gamma
        self.n_runs = n_runs
        self.epochs = epochs
        self.dist = Normal if continuous_policy else Categorical
        self.cont_p = continuous_policy
        self.reward_flag = self.check_reward()

    def check_reward(self):
        """ Check if positive or negative rewards """
        self.env.reset()
        action = self.env.action_space.sample()
        _, reward, _, _ = self.env.step(action)

        return 1 if reward > 0 else -1


    def perform_rollout(self, obs):
        dist_params = self.policy(to_torch(obs))

        # TODO: CHECK arg
        dist = self.dist(*dist_params)
        action = dist.sample(torch.Size((self.act_dim,)))
        # action = dist.sample()
        # .sample(torch.Size((1,)))
        # TODO: Fix action element
        # nxt_obs, reward, done, _ = self.env.step(to_npy(action).squeeze()) if self.cont_p else self.env.step(int(to_npy(action)[0]))
        nxt_obs, reward, done, _ = self.env.step(to_npy(action)) if self.cont_p else self.env.step(
            int(to_npy(action)[0]))

        return nxt_obs.squeeze(), reward, done, action, dist.log_prob(action)

    def perform_rollouts(self):

        actions = torch.empty((self.horizon, self.act_dim))
        observations = torch.empty((self.horizon, self.obs_dim))
        rewards = torch.empty((self.horizon, 1))
        log_probs = torch.empty((self.horizon, 1))
        masks = torch.empty((self.horizon, 1))
        obs = self.env.reset()

        for i in range(self.horizon):
            nxt_obs, reward, done, action, log_prob =  self.perform_rollout(obs)

            actions[i] = action
            observations[i] = to_torch(nxt_obs)
            # rewards[i] = to_torch(reward) if self.cont_p else to_torch(np.array([reward]))
            rewards[i] = to_torch(reward) if isinstance(reward, np.ndarray) else to_torch(np.array([reward]))
            masks[i] = 1 - done
            log_probs[i] = log_prob.sum()

            if done:
                nxt_obs = self.env.reset()

            obs = nxt_obs


        return actions, observations, rewards, masks, log_probs.detach()


    def discount_rewards(self, rewards, masks, gamma=0.99):
        disc_r = torch.zeros_like(rewards)
        run_add = 0

        for t in reversed(range(0, rewards.shape[0])):
            run_add = (run_add * masks[t] * gamma + rewards[t])
            disc_r[t] = run_add

        return disc_r

    def compute_baseline(self):
        pass


    def reinforce_update(self, actions, obs, rewards, log_probs, epochs, batch_size=32, shuffle=True):

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        dataset = TensorDataset(actions, obs, rewards, log_probs)
        loader = DataLoader(dataset, batch_size, shuffle=shuffle)

        for epoch in range(epochs):
            for _act, _obs, _rewards, _log_prob in loader:

                dist_params = self.policy(_obs)
                dist = self.dist(*dist_params)

                log_prob = dist.log_prob(_act)

                # Reward flag to check if reward is positive or negative
                # loss = self.reward_flag * torch.mean(-log_prob * _rewards)
                # loss = -torch.mean(log_prob * _rewards)
                loss = self.reward_flag * torch.mean(log_prob * _rewards)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def run(self):
        writer = SummaryWriter()

        for run in range(self.n_runs):

            act, obs, reward, masks, log_probs = self.perform_rollouts()
            cum_reward = torch.sum(reward)
            disc_rewards = self.discount_rewards(reward, masks, self.gamma)
            cum_disc_reward = torch.sum(disc_rewards)

            self.reinforce_update(act, obs, disc_rewards, log_probs, self.epochs)


            eval_reward = self.eval_policy()
            print("Cumulative discounted reward: {} # Eval reward: {} in run: {}/{}".format(cum_disc_reward, eval_reward, run+1, self.n_runs))

            writer.add_scalars('reinforce', {'train_reward': to_npy(cum_disc_reward).item(),
                                'eval_reward': eval_reward}, global_step=run)


    def eval_policy(self):
        obs = self.env.reset()
        done = False
        r = 0
        while not done:
            dist_params = self.policy(to_torch(obs))

            obs, reward, done, _ = self.env.step(to_npy(dist_params[0])) if self.cont_p else self.env.step(np.argmax(to_npy(*dist_params)))
            self.env.render()
            r += reward
        return r


# if __name__ == '__main__':
#     env = gym.make('CartPole-v1')
#     # env = gym.make('MountainCar-v0')
#     # env = gym.make('LunarLander-v2')
#     # env = gym.make('LunarLanderContinuous-v2')
#     # env = gym.make('Pendulum-v0')
#
#     act_dim = env.action_space.n if len(env.action_space.shape) == 0 else env.action_space.shape[0]
#     obs_dim = env.observation_space.shape[0]
#     continous_policy = False if len(env.action_space.shape) == 0 else True
#
#     policy_kwargs = {'in_features': obs_dim, 'out_features': act_dim, 'n_hidden': [128, 128],
#                      'nonlin': 'tanh', 'layer_norm': True}
#
#     env.seed(420)
#     torch.manual_seed(420)
#
#     model = Reinforce(env, n_runs=1000, gamma=0.99, horizon=2000, epochs=1,
#                       lr=1e-3, continuous_policy=continous_policy, **policy_kwargs)
#     model.run()