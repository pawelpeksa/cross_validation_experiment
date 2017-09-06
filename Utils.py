import time

def get_seed():
    t = time.time() - int(time.time())
    t *= 1000000
    return int(t)

