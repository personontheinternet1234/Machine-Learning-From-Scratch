import time
from tqdm import tqdm

GREEN = '\033[32m'
RED = '\033[31m'
DEFAULT = '\033[0m'


def bar(idx, max_idx, start, desc=None, b_len=50, color='\033[0m'):
    if not isinstance(b_len, int):
        raise ValueError(f"'b_len' is not an integer: {b_len}")
    current = time.time()
    completed = (idx + 1) / max_idx
    p_bar = (
        f"\r{GREEN}{'—' * int(b_len * completed)}"
        f"{RED}{'—' * (b_len - int(b_len * completed))}"
        f"{color}  {round(current - start, 1)}s  "
        f"{round((idx + 1) / (current - start), 1)}it/s{DEFAULT}"
    )
    if desc is None:
        print(p_bar, end='')
    else:
        p_desc = (
            f"{color}  {desc}{DEFAULT}"
        )
        print(p_bar, p_desc, end='')


num = 500
start_t = time.time()
status = True
simulate = True

print(f"Training model - {num} iterations")
for i in range(num):
    if simulate:
        time.sleep(0.01)
    if status:
        bar(i, num, start_t, desc=f'{(i + 1)}/{num}')
print('')

for i in tqdm(range(num), ncols=100, disable=not status):
    if simulate:
        time.sleep(0.01)
