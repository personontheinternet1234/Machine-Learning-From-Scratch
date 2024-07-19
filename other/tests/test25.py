import time

GREEN = '\033[32m'
RED = '\033[31m'
DEFAULT = '\033[0m'


def bar(idx, max_idx, desc=None, b_len=50, color='\033[0m'):
    if not isinstance(b_len, int):
        raise ValueError(f"'b_len' is not an integer: {b_len}")
    completed = (idx + 1) / max_idx
    p_bar = (
        f"\r{GREEN}{'—' * int(b_len * completed)}"
        f"{RED}{'—' * (b_len - int(b_len * completed))}{DEFAULT}"
    )
    print(p_bar, end='')
    if desc:
        p_desc = (
            f"{color}  {desc}{DEFAULT}"
        )
        print(p_desc, end='')


num = 500
start = time.time()
status = True
simulate = True

print(f"Training model  {num} iterations")
for i in range(num):
    if simulate:
        time.sleep(0.01)
    if status:
        current = time.time()
        description = (
            f"{round(current - start, 1)}s  "
            f"{round((i + 1) / (current - start), 1)}it/s"
        )
        bar(i, num, desc=description)
print('')
