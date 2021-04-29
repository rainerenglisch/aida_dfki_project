from multiprocessing import set_start_method
#set_start_method("spawn")
from multiprocessing import get_context
from multiprocessing import Pool
from os import getpid

def double(i):
    print(f"I'm process {getpid()}\n")
    return i * 2

if __name__ == '__main__':
    with get_context("spawn").Pool() as pool:
        result = pool.map(double, [1, 2, 3, 4, 5])
        print(result)