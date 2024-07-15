"""
'helper_functions' includes minor utility functions for GardenPy.

'helper_functions' includes:
    'print_color': Print text in color (default = green).
    'input_color': Input text in color (default = green).
    'print_credits': Print credits for GardenPy.

refer to 'todo' for in-depth documentation on these functions.
"""

GREEN = '\033[32m'
DEFAULT = '\033[0m'


def print_color(text, color=GREEN, end='\n'):
    """
    'print_color' is a function that prints text in color.

    Arguments:
        text: The text that will be printed.
        color: The color that the text will be printed in. (default of green)
        end: The end-line argument.
    """
    # print text
    print(f'{color}{text}{DEFAULT}', end=end)


def input_color(text, color=GREEN):
    """
    'input_color' is a function that prompts the user for an input in color.

    Arguments:
        text: The text that will be printed for input.
        color: The color that the text will be printed in for input. (default of green)

    Returns:
        A string of the user's input.
    """
    # get and return input
    return input(f'{color}{text}{DEFAULT}')


def print_credits(color=GREEN):
    """
    'print_credits' is a function that prints the credits for GardenPy.

    Arguments:
        color: The color that the text will be printed in. (default of green)
    """
    # print credits in alphabetical order
    print_color('"""', color=color)
    print_color('GardenPy', color=color)
    print_color("   Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>", color=color)
    print_color("   Mason Morales CO '25 <mmorales25@punahou.edu>", color=color)
    print_color("   Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>", color=color)
    print_color("   Derek Yee CO '25 <dyee@punahou.edu>", color=color)
    print_color('"""', color=color)
