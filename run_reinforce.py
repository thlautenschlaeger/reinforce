import gym
import torch
from src.reinforce import Reinforce
import argparse
import sys

def run(args=None):

    parser = argparse.ArgumentParser(description='Running reinforce on gym environments')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Choose your gym environment')
    parser.add_argument('--hidden_neurons', type=list, default=[64, 64], help='Hidden neurons in a list')
    parser.add_argument('--gamma', type=int, default=0.99, help='Discount factor for advantage')
    parser.add_argument('--horizon', type=int, default=2000, help='Total rollout horizon')
    parser.add_argument('--n_runs', type=int, default=1000, help='Total number of reinforce steps')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learn rate')
    parser.add_argument('--nonlin', type=str, default='tanh', help='Non linearities for neural networks')
    parser.add_argument('--layer_norm', type=bool, default=True, help='Layer normalization')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batches for each optimizer step')




    args = parser.parse_known_args(args)[0]

    env = gym.make(args.env)
    # env = gym.make('MountainCar-v0')
    # env = gym.make('LunarLander-v2')
    # env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('Pendulum-v0')

    act_dim = env.action_space.n if len(env.action_space.shape) == 0 else env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    continuous_policy = False if len(env.action_space.shape) == 0 else True

    policy_kwargs = {'in_features': obs_dim, 'out_features': act_dim, 'n_hidden': args.hidden_neurons,
                     'nonlin': args.nonlin, 'layer_norm': args.layer_norm}

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Reinforce(env, n_runs=args.n_runs, gamma=args.gamma, horizon=args.horizon, epochs=1,
                      lr=args.lr, continuous_policy=continuous_policy, batch_size=args.batch_size,
                      **policy_kwargs)
    model.run()


if __name__ == '__main__':
    run(sys.argv)
