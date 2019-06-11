import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    print(f"[{name}] start")
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")
