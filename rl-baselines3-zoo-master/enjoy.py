import time
from rl_zoo3.enjoy import enjoy

if __name__ == "__main__":
    for i in range(1000):
        seed = i
        time.sleep(1)
        enjoy(seed)
