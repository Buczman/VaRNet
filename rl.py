import gym
import numpy as np
import yfinance as yf
import random


class ValueAtRiskEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.frame = 0
        self.done = False
        self.name = 'VaREnv'
        self.symbols = [
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOG",
            "FB",
            "BABA",
            "TSLA",
            "WMT",
            "NVDA",
            "DIS",
            "CRM",
            "NFLX",
            "INTC",
            "KO",
            "ABT",
            "AMD",
            "ZM",
            "SQ",
            "WDAY",
            "NIO",
            "ROKU",
            "PTON",
            "FSLY",
            "BYND",
            "QDEL",
            "NTNX",
            "BL"
        ]
        self.symbol = random.choice(self.symbols)
        self.data = self._gather_data(self.symbol)

        self.action_space = gym.spaces.Box(
            low=-1, high=0, shape=(1,), dtype=np.float16
        )
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, dtype=np.float16, shape=(32, 1)
        )

    def step(self, action):
        if action[0] is None:
            a = 1
        self.frame += 1
        if self.observation_space.shape[0]+self.frame == self.data.shape[0] - 1:
            self.done = True
        print(self.data.index[self.observation_space.shape[0]+self.frame],
              self.data[self.observation_space.shape[0]+self.frame], action, self._calculate_loss(action))
        return self.data[self.frame:(self.observation_space.shape[0]+self.frame)].values, \
               self._calculate_loss(action), \
               self.done, \
               []

    def reset(self):
        self.frame = 0
        self.done = False
        self.symbol = random.choice(self.symbols)
        self.data = self._gather_data(self.symbol)
        return self.data[self.frame:(self.observation_space.shape[0]+self.frame)].values

    def render(self, mode='human'):
        print(self.data.index[self.frame])

    def _calculate_loss(self, action):
        if self.data[(self.observation_space.shape[0]+self.frame)] > 0:
            return -1*np.abs(action)
        elif self.data[(self.observation_space.shape[0]+self.frame)] > action:
            return -1*np.abs(self.data[(self.observation_space.shape[0]+self.frame)] - action)
        else:
            return -1*np.abs([self.data[(self.observation_space.shape[0]+self.frame)]])

    @staticmethod
    def _gather_data(symbol):
        data = yf.download(symbol,
                           start='2000-01-01',
                           end='2019-12-31',
                           progress=False)
        return (np.log(data.Close) - np.log(data.Close.shift(1))).iloc[1:]
