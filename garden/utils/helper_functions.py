"""
helper functions
"""

GREEN = '\033[32m'


def print_color(text, color_code=GREEN, end='\n'):
    """ print text in color """
    print(f'{color_code}{text}{GREEN}', end=end)


def input_color(text, color_code=GREEN):
    """ print input statement in color """
    return input(f'{color_code}{text}{GREEN}')


def print_credits():
    """ print garden credits """
    # print credits in alphabetical order
    print_color('"""')
    print_color('The Garden')
    print_color("Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>")
    print_color("Mason Morales CO '25 <mmorales25@punahou.edu>")
    print_color("Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>")
    print_color("Derek Yee CO '25 <dyee@punahou.edu>")
    print_color('"""')
