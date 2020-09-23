from rl import ValueAtRiskEnvironment
import random

env = ValueAtRiskEnvironment()
obs = env.reset()
for i in range(2000):
  action = random.uniform(-1, 1)
  obs, rewards, done, info = env.step(action)
  env.render()