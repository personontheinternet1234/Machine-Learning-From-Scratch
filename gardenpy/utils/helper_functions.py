r"""
'helper_functions' includes minor utility functions for GardenPy.

'helper_functions' includes:
    'ansi': Common ANSI formats.
    'progress': Progress bar.
    'convert_time': Converts time to hours:minutes:seconds.
    'print_credits': Print credits for GardenPy.

Refer to 'todo' for in-depth documentation on these functions.
"""

ansi = {
        # 'ansi' is a variable that contains all the commonly used ANSI formats.
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
    if not isinstance(b_len, int):
        # invalid datatype
        raise ValueError(f"'b_len' is not an integer: '{b_len}'")
    # completed progress
    completed = (idx + 1) / max_idx
    # make progress bar
    p_bar = (
        f"\r{ansi['green']}{'—' * int(b_len * completed)}"
        f"{ansi['red']}{'—' * (b_len - int(b_len * completed))}{ansi['reset']}"
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
    if not number_colors:
        number_colors = ansi['reset']
    if not separators_color:
        separators_color = ansi['reset']
    # round seconds
    seconds = int(seconds)
    # find minutes and hours
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    # adjust times
    minutes -= hours * 60
    seconds -= minutes * 60
    # return time
    return f"{number_colors}{hours:01}{separators_color}:{number_colors}{minutes:02}{separators_color}:{number_colors}{seconds:02}{ansi['reset']}"


def print_credits() -> None:
    r"""
    'print_credits' is a function that prints the credits for GardenPy.

    Arguments:
        None.

    Returns:
        None.
    """
    # print credits in alphabetical order
    print(f"{ansi['bold']}{ansi['green']}GardenPy{ansi['reset']}")
    print(f"    {ansi['bold']}Contributors{ansi['reset']}")
    print(f"    Christian SW Host-Madsen", end='')
    print(f"    {ansi['white']}Punahou School CO '25{ansi['reset']}", end='')
    print(f"    {ansi['bright_black']}<chost-madsen25@punahou.edu>{ansi['reset']}",)
    print(f"    Mason YY Morales", end='')
    print(f"            {ansi['white']}Punahou School CO '25{ansi['reset']}", end='')
    print(f"    {ansi['bright_black']}<mmorales25@punahou.edu>{ansi['reset']}")
    print(f"    Isaac P Verbrugge", end='')
    print(f"           {ansi['white']}Punahou School CO '25{ansi['reset']}", end='')
    print(f"    {ansi['bright_black']}<isaacverbrugge@gmail.com>{ansi['reset']}")
    print(f"    Derek Yee", end='')
    print(f"                   {ansi['white']}Punahou School CO '25{ansi['reset']}", end='')
    print(f"    {ansi['bright_black']}<dyee25@punahou.edu>{ansi['reset']}")
