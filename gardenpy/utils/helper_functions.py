r"""
'helper_functions' includes minor utility functions for GardenPy.

'helper_functions' includes:
    'ansi_formats': Common ANSI formats.
    'progress': Progress bar.
    'convert_time': Converts time to hours:minutes:seconds.
    'print_credits': Print credits for GardenPy.

Refer to 'todo' for in-depth documentation on these functions.
"""


def ansi_formats() -> dict:
    r"""
        'ansi_formats' is a function that returns a dictionary of common ansi formats.

        Arguments:
            None.

        Returns:
            A dictionary of ANSI formats.
        """
    # ANSI formats
    formats = {
        'reset': '\033[0m',
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        'blinking': '\033[5m',
        'reverse': '\033[7m',
        'hidden': '\033[8m',
        'strikethrough': '\033[9m'
    }
    # return ANSI formats
    return formats


def progress(idx: int, max_idx: int, desc=None, b_len: int = 50) -> None:
    r"""
    'progress' is a function that prints a progress bar.

    Arguments:
        idx: The current iteration.
        max_idx: The maximum amount of iterations.
        desc: The progress bar description.
        b_len: The length of the progress bar.

    Returns:
        None.
    """
    # get ansi formats
    progress_formats = ansi_formats()
    if not isinstance(b_len, int):
        # invalid datatype
        raise ValueError(f"'b_len' is not an integer: '{b_len}'")
    # completed progress
    completed = (idx + 1) / max_idx
    # make progress bar
    p_bar = (
        f"\r{progress_formats['green']}{'—' * int(b_len * completed)}"
        f"{progress_formats['red']}{'—' * (b_len - int(b_len * completed))}{progress_formats['reset']}"
    )
    # print progress bar
    print(p_bar, end='')

    if desc:
        # set description
        p_desc = (
            f"  {desc}"
        )
        # print description
        print(p_desc, end='')


def convert_time(seconds: float, number_colors: str = None, separators_color: str = None) -> str:
    r"""
    'convert_time' is a function that converts seconds to hours:minutes:seconds.

    Arguments:
        seconds: The elapsed seconds.
        number_colors: The color of the numbers.
        separators_color: The color of the time separators.

    Returns:
        Converted time.
    """
    # get ansi formats
    time_formats = ansi_formats()
    if not number_colors:
        number_colors = time_formats['reset']
    if not separators_color:
        separators_color = time_formats['reset']
    # round seconds
    seconds = int(seconds)
    # find minutes and hours
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    # adjust times
    minutes -= hours * 60
    seconds -= minutes * 60
    # return time
    return f"{number_colors}{hours:01}{separators_color}:{number_colors}{minutes:02}{separators_color}:{number_colors}{seconds:02}{time_formats['reset']}"


def print_credits() -> None:
    r"""
    'print_credits' is a function that prints the credits for GardenPy.

    Arguments:
        None.

    Returns:
        None.
    """
    # get ansi formats
    credit_formats = ansi_formats()
    bold_green = '\033[1;32m'
    # print credits in alphabetical order
    print(f"{bold_green}GardenPy{credit_formats['reset']}")
    print(f"    {credit_formats['bold']}Contributors:{credit_formats['reset']}")
    print(f"    Christian SW Host-Madsen", end='')
    print(f"  {credit_formats['white']}CO '25{credit_formats['reset']}", end='')
    print(f"  {credit_formats['bright_black']}<chost-madsen25@punahou.edu>{credit_formats['reset']}",)
    print(f"    Mason Morales", end='')
    print(f"  {credit_formats['white']}CO '25{credit_formats['reset']}", end='')
    print(f"  {credit_formats['bright_black']}<mmorales25@punahou.edu>{credit_formats['reset']}")
    print(f"    Isaac P Verbrugge", end='')
    print(f"  {credit_formats['white']}CO '25{credit_formats['reset']}", end='')
    print(f"  {credit_formats['bright_black']}<isaacverbrugge@gmail.com>{credit_formats['reset']}")
    print(f"    Derek Yee CO", end='')
    print(f"  {credit_formats['white']}CO '25{credit_formats['reset']}", end='')
    print(f"  {credit_formats['bright_black']}<dyee25@punahou.edu>{credit_formats['reset']}")
