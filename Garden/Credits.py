"""
Garden credits
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""


def print_credits():
    """ print garden credits """
    # define color printing
    def print_color(text, color_code):
        print(f'{color_code}{text}\033[0m')
    # print credits
    print_color('"""', '\033[32m')
    print_color('The Garden', '\033[32m')
    print_color("Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>", '\033[32m')
    print_color("Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>", '\033[32m')
    print_color('"""', '\033[32m')