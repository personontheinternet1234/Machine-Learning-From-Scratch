"""
'helper_functions' includes minor utility functions for GardenPy.

'helper_functions' includes:
    'print_color': Print text in color.
    'input_color': Input text in color.
    'progress': Progress bar.
    'convert_time': Converts time to hours:minutes:seconds.
    'print_credits': Print credits for GardenPy.

refer to 'todo' for in-depth documentation on these functions.
"""

import math

DEFAULT = '\033[0m'
LIGHT_GRAY = '\033[37m'
GRAY = '\033[90m'
GREEN = '\033[32m'
RED = '\033[31m'


def print_color(text, color=DEFAULT, end='\n') -> None:
    """
    'print_color' is a function that prints text in color.

    Arguments:
        text: The text that will be printed.
        color: The color that the text will be printed in.
        end: The end-line argument.

    Returns:
        None.
    """
    # print text
    print(f"{color}{text}{DEFAULT}", end=end)


def input_color(text, color=DEFAULT) -> str:
    """
    'input_color' is a function that prompts the user for an input in color.

    Arguments:
        text: The text that will be printed for input.
        color: The color that the text will be printed in for input.

    Returns:
        A string of the user's input.
    """
    # get and return input
    return input(f"{color}{text}{DEFAULT}")


def progress(idx: int, max_idx: int, desc=None, b_len: int = 50) -> None:
    """
    'progress' is a function that prints a progress bar.

    Arguments:
        idx: The current iteration.
        max_idx: The maximum amount of iterations.
        desc: The progress bar description.
        b_len: The length of the progress bar.

    Returns:
        None.
    """
    if not isinstance(b_len, int):
        # invalid datatype
        raise ValueError(f"'b_len' is not an integer: {b_len}")
    # completed progress
    completed = (idx + 1) / max_idx
    # make progress bar
    p_bar = (
        f"\r{GREEN}{'—' * int(b_len * completed)}"
        f"{RED}{'—' * (b_len - int(b_len * completed))}{DEFAULT}"
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


def convert_time(seconds: float, number_colors: str = DEFAULT, separators_color: str = DEFAULT) -> str:
    """
    'convert_time' is a function that converts seconds to hours:minutes:seconds.

    Arguments:
        seconds: The elapsed seconds.
        number_colors: The color of the numbers.
        separators_color: The color of the time separators.

    Returns:
        Converted time.
    """
    # round seconds
    seconds = int(seconds)
    # find minutes and hours
    minutes = math.floor(seconds / 60)
    hours = math.floor(minutes / 60)
    # adjust times
    minutes -= hours * 60
    seconds -= minutes * 60
    # return time
    return f"{number_colors}{hours:01}{separators_color}:{number_colors}{minutes:02}{separators_color}:{number_colors}{seconds:02}{DEFAULT}"


def print_credits() -> None:
    """
    'print_credits' is a function that prints the credits for GardenPy.

    Returns:
        None.
    """
    # print credits in alphabetical order
    print_color("GardenPy", color=GREEN)
    print_color("   Contributors:", color=LIGHT_GRAY)
    print_color("   Christian SW Host-Madsen CO '25 ", end='')
    print_color("   <chost-madsen25@punahou.edu>", color=GRAY)
    print_color("   Mason Morales CO '25 ", end='')
    print_color("              <mmorales25@punahou.edu>", color=GRAY)
    print_color("   Isaac Park Verbrugge CO '25 ", end='')
    print_color("       <iverbrugge25@punahou.edu>", color=GRAY)
    print_color("   Derek Yee CO '25 ", end='')
    print_color("                  <dyee25@punahou.edu>", color=GRAY)
