# progress bar test

import time


def simple_progress_bar(total):
    # s/o ChatGPT!
    for i in range(total):
        time.sleep(0.1)  # Simulate work
        progress = (i + 1) / total
        bar_length = 40
        block = int(bar_length * progress)
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {int(progress * 100)}%"
        print(text, end="")

    print("\nDone!")


simple_progress_bar(50)
