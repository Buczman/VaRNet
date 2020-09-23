from rl import ValueAtRiskEnvironment
# import random
#
# env = ValueAtRiskEnvironment()
# obs = env.reset()
# for i in range(2000):
#   action = random.uniform(-1, 1)
#   obs, rewards, done, info = env.step(action)
#   env.render()


from all.experiments import run_experiment, plot_returns_100
from preset import ppo
from var_gym import GymEnvironment


def main():
  device = 'cuda'
  timesteps = 100000
  run_experiment(
    [ppo(device)],
    [GymEnvironment(ValueAtRiskEnvironment(), 'VaR', device)],
    timesteps,
  )
  plot_returns_100('runs', timesteps=timesteps)


if __name__ == "__main__":
  main()