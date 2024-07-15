import time

from gardenpy.utils import progress
from gardenpy.utils.helper_functions import print_credits

status = True
max_iter = 1000

print('')
print_credits()
print('')

time.sleep(1.0)

print(f"Testing")
start = time.time()
for i in range(max_iter):
    time.sleep(0.01)
    if status:
        current = time.time()
        desc = (
            f"{round(current - start, 1)}s  "
            f"{round((i + 1) / (current - start), 1)}it/s"
        )
        progress(i, max_iter, desc=desc)
print('')
