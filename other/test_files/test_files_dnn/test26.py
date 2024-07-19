import time

from gardenpy.utils import (
    ansi_formats,
    progress,
    convert_time
)
from gardenpy.utils.helper_functions import print_credits

ANSI = ansi_formats()

status = True
max_iter = 10000

print()
print_credits()
print()

time.sleep(1.0)

print("Test 1")
for i in range(int(max_iter / 10)):
    time.sleep(0.001)
    if status:
        progress(i, int(max_iter / 10))
print()

print("Test 2")
start = time.time()
for i in range(max_iter):
    time.sleep(0.001)
    if status:
        desc = (
            f"{str(i + 1).zfill(len(str(max_iter)))}{ANSI['white']}it{ANSI['reset']}/{max_iter}{ANSI['white']}it{ANSI['reset']}  "
            f"{(100 * (i + 1) / max_iter):05.1f}{ANSI['white']}%{ANSI['reset']}  "
            f"{convert_time(time.time() - start)}{ANSI['white']}et{ANSI['reset']}  "
            f"{convert_time((time.time() - start) * max_iter / (i + 1) - (time.time() - start))}{ANSI['white']}eta{ANSI['reset']}  "
            f"{round((i + 1) / (time.time() - start), 1)}{ANSI['white']}it/s{ANSI['reset']}"
        )
        progress(i, max_iter, desc=desc)
print()
