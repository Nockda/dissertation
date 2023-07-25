from gym import Wrapper

class ReverseSwimmerWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = -reward
        return obs, reward, done, info