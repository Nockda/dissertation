import time
from rl_zoo3.enjoy import enjoy
from tqdm import tqdm

if __name__ == "__main__":
    for i in tqdm(range(100000)):
        seed = i
        enjoy(seed)
    # enjoy(0)
