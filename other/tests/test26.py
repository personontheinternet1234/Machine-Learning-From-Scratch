import time

from gardenpy.utils import (
    ansi,
    progress,
    convert_time
)
from gardenpy.utils.helpers import print_contributors

status = True
max_iter = 10000

print()
print_contributors()
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
            f"{str(i + 1).zfill(len(str(max_iter)))}{ansi['white']}it{ansi['reset']}/{max_iter}{ansi['white']}it{ansi['reset']}  "
            f"{(100 * (i + 1) / max_iter):05.1f}{ansi['white']}%{ansi['reset']}  "
            f"{convert_time(time.time() - start)}{ansi['white']}et{ansi['reset']}  "
            f"{convert_time((time.time() - start) * max_iter / (i + 1) - (time.time() - start))}{ansi['white']}eta{ansi['reset']}  "
            f"{round((i + 1) / (time.time() - start), 1)}{ansi['white']}it/s{ansi['reset']}"
        )
        progress(i, max_iter, desc=desc)
print()
