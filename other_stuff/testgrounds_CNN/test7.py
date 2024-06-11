from tqdm import tqdm
from colorama import Fore, Style

# Initialize colorama

# Example list to iterate over
items = list(range(0, 100))

# Custom format with white color
bar_format = (
    f"{Fore.GREEN}{{l_bar}}{{bar}}{{r_bar}}{Style.RESET_ALL}"
)

for item in tqdm(items, bar_format=bar_format, desc="Processing"):
    # Simulate some work
    pass