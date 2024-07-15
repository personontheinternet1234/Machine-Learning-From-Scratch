import time

from gardenpy.utils import progress
from gardenpy.utils.helper_functions import convert_time, print_credits

status = True
max_iter = 10000

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
            f"et-{convert_time(time.time() - start)}  "
            f"eta-{convert_time((time.time() - start) * max_iter / (i + 1) - (time.time() - start))}  "
            f"{round((i + 1) / (time.time() - start), 1)}it/s"
        )
        progress(i, max_iter, desc=desc)
print()
