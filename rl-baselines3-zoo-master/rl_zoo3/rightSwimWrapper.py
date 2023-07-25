# right_swim_wrapper.py
import gym
from gym import spaces

class RightSwimWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # swimmer의 x 좌표를 가져옵니다.
        x_position = obs[0]
        # 오른쪽으로 이동할 때 추가 보상을 제공합니다.
        reward += x_position
        return obs, reward, done, info
