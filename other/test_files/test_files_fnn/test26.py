import time

from gardenpy.utils import progress, convert_time
from gardenpy.utils.helper_functions import print_credits

DEFAULT = '\033[0m'
LIGHT_GRAY = '\033[37m'

status = True
max_iter = 1000

print()
print_credits()
print()

time.sleep(1.0)

print("Test 1")
for i in range(int(max_iter / 10)):
    time.sleep(0.01)
    if status:
        progress(i, int(max_iter / 10))
print()

print("Test 2")
start = time.time()
for i in range(max_iter):
    time.sleep(0.01)
    if status:
        desc = (
            f"{str(i + 1).zfill(len(str(max_iter)))}/{max_iter}  "
            f"{convert_time(time.time() - start)}{LIGHT_GRAY}et{DEFAULT}  "
            f"{convert_time((time.time() - start) * max_iter / (i + 1) - (time.time() - start))}{LIGHT_GRAY}eta{DEFAULT}  "
            f"{round((i + 1) / (time.time() - start), 1)}{LIGHT_GRAY}it/s{DEFAULT}"
        )
        progress(i, max_iter, desc=desc)
print()
