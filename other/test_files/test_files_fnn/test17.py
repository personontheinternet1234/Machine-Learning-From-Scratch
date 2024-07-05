import time

from tqdm import tqdm

# Specify the exact width you want for the progress bar
pbar = tqdm(total=100, ncols=80, desc='a')
pbar2 = tqdm(total=100, ncols=80, desc='abc')

# Update the progress bar in your loop
for i in range(100):
    # Do some work here
    pbar.update(1)
    time.sleep(0.05)

for i in range(100):
    # Do some work here
    pbar2.update(1)
    time.sleep(0.05)

# Close the progress bar
pbar.close()
pbar2.close()