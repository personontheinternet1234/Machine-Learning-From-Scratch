r"""
helper_functions.py

Includes basic helper functions for GardenPy.
"""

import sys
from typing import Optional, Union
import time

ansi = {
        # common ANSI formats
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


def progress(idx: int, max_idx: int, *, desc: Optional[str] = None, b_len: int = 50) -> None:
    r"""
    Prints a customizable progress bar for any sort of loop.

    Args:
        idx (int):
            Current loop index.
        max_idx (int):
            Maximum loop index.
        desc (str, optional):
            Progress bar description.
            Defaults to no description
        b_len (int, optional):
            Bar length.
            Defaults to 50 characters.

    Returns:
        None

    Raises:
        TypeError: If parameters are of the wrong type.
    """
    # check for errors
    if not isinstance(idx, int):
        raise TypeError("'idx' must be an integer")
    if not (isinstance(max_idx, int) and 0 < max_idx):
        raise TypeError("'max_idx' must be a positive integer")
    if not (isinstance(b_len, int) and 0 < b_len):
        raise TypeError("'b_len' must be a positive integer")
    # completed progress
    completed = (idx + 1) / max_idx
    # make progress bar
    sys.stdout.write(f"\r{ansi['reset']}[{ansi['green']}{'—' * int(b_len * completed)}{ansi['red']}{'—' * (b_len - int(b_len * completed))}{ansi['reset']}]  {desc or ''}")
    sys.stdout.flush()
    if completed == 1:
        sys.stdout.write("\n")
    return None


def convert_time(seconds: Union[float, int]) -> str:
    r"""
    Converts seconds to hours:minutes:seconds.

    Args:
        seconds (float | int):
            Number of seconds.

    Returns:
        str:
            Time in hours:minutes:seconds format.

    Raises:
        TypeError: If parameters are of the wrong type.
    """
    # check for errors
    if not (isinstance(seconds, (float, int)) and 0 < seconds):
        raise TypeError("'seconds' must be a positive float or integer")
    # calculate hours and minutes
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    # return time
    return f"{hours:01}:{minutes:02}:{seconds:02}"


def slow_print(text: str, *, delay: Union[float, int] = 0.05) -> None:
    r"""
    Prints text with delay.

    Args:
        text (str):
            Text to print.

        delay (float | int):
            Delay between characters in seconds.
            Defaults to 0.05

    Returns:
        None

    Raises:
        TypeError: If parameters are of the wrong type.
    """
    # check for errors
    if not isinstance(text, str):
        raise TypeError("'text' must be a str")
    if not (isinstance(delay, (float, int)) and 0 < delay):
        raise TypeError("'delay' must be a positive float or integer")
    # print text
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    return None


def print_contributors(*, who: Optional[list] = None, cinematic: bool = False) -> None:
    r"""
    Prints GardenPy contributors.

    Args:
        who (list, optional):
            Type of contributors to print.
        cinematic (bool):
            Cinematic-style printing of contributors.
            Defaults to False.

    Returns:
        None

    Raises:
        TypeError: If parameters are of the wrong type.
        ValueError: If invalid contributors were requested.
    """
    # contributors
    contributors = {
        'programmers': [
            ["Christian SW Host-Madsen", "Punahou School CO '25", "<c.host.madsen25@gmail.com>"],
            ["Mason YY Morales", "Punahou School CO '25", "<mmorales25@punahou.edu>"],
            ["Isaac P Verbrugge", "Punahou School CO '25", "<isaacverbrugge@gmail.com>"],
            ["Derek S Yee", "Punahou School CO '25", "<dyee25@punahou.edu>"]
        ],
        'artists': [
            ["Kamalau Kimata", "Punahou School CO '25", "<kkimata25@punahou.edu>"]
        ],
        'thanks': [
            ['Justin Johnson', 'The University of Michigan', 'his thorough video lectures'],
            ['Josh Starmer', 'StatQuest', 'his informative videos'],
            ['The PyTorch Team', 'PyTorch', 'their detailed documentation and resources'],
            ['Grant Sanderson', '3Blue1Brown', 'his inspirational videos']
        ]
    }

    # clean who list
    contributor_types = ['programmers', 'artists', 'thanks']
    if isinstance(who, list):
        who = list(set(who))
    cinematic = bool(cinematic)
    if not (isinstance(who, list) or who is None):
        raise TypeError("'who' must be a list")
    if who is not None and not all([(pers in contributor_types) for pers in who]):
        raise ValueError(
            f"Invalid contributor type detected in: {who}\n"
            f"Choose from: {contributor_types}"
        )
    who = who or contributor_types

    # print contributors
    if cinematic:
        print(f"{ansi['reset']}", end='')
        slow_print("The <", delay=0.05)
        print(f"{ansi['bold']}", end='')
        slow_print("MACHINE LEARNING ", delay=0.05)
        print(f"{ansi['reset']}{ansi['white']}{ansi['italic']}", end='')
        slow_print("from scratch", delay=0.05)
        print(f"{ansi['reset']}", end='')
        slow_print("> team presents", delay=0.05)
        print(f"{ansi['reset']}")
        time.sleep(0.5)
        print(f"{ansi['bold']}{ansi['green']}", end='')
        slow_print("GardenPy", delay=0.25)
        print(f"{ansi['reset']}", end='\n')
        if 'programmers' in who:
            time.sleep(0.5)
            print(f"{ansi['bold']}", end='')
            slow_print("Programmers", delay=0.05)
            print(f"{ansi['reset']}", end='\n')
            for row in contributors['programmers']:
                time.sleep(0.5)
                slow_print("    {reset}{:<30} {white}{:<25} {reset}{bright_black}{:<20}".format(row[0], row[1], row[2], **ansi), delay=0.05)
                print(f"{ansi['reset']}", end='\n')
        if 'artists' in who:
            time.sleep(0.5)
            print(f"{ansi['bold']}", end='')
            slow_print("Artists", delay=0.05)
            print(f"{ansi['reset']}", end='\n')
            for row in contributors['artists']:
                time.sleep(0.5)
                slow_print("    {reset}{:<30} {white}{:<25} {reset}{bright_black}{:<20}".format(row[0], row[1], row[2], **ansi), delay=0.05)
                print(f"{ansi['reset']}", end='\n')
        if 'thanks' in who:
            time.sleep(0.5)
            print(f"{ansi['bold']}", end='')
            slow_print("Special Thanks To", delay=0.05)
            print(f"{ansi['reset']}", end='')
            for row in contributors['thanks']:
                time.sleep(0.5)
                print("\n    ", end='')
                print(f"{ansi['reset']}", end='')
                slow_print(f"{row[0]} ", delay=0.05)
                print(f"{ansi['white']}", end='')
                slow_print(f"from {row[1]} ", delay=0.05)
                print(f"{ansi['bright_black']}", end='')
                slow_print(f"for {row[2]}", delay=0.05)
            print(f"{ansi['reset']}", end='\n')
            time.sleep(0.5)
    else:
        print(f"{ansi['bold']}{ansi['green']}GardenPy{ansi['reset']}")
        if 'programmers' in who:
            print(f"{ansi['bold']}Programmers{ansi['reset']}", end='\n')
            for row in contributors['programmers']:
                print("    {reset}{:<30} {white}{:<25}{reset} {bright_black}{:<20}{reset}".format(row[0], row[1], row[2], **ansi))
        if 'artists' in who:
            print(f"{ansi['bold']}Artists{ansi['reset']}", end='\n')
            for row in contributors['artists']:
                print("    {reset}{:<30} {white}{:<25}{reset} {bright_black}{:<20}{reset}".format(row[0], row[1], row[2], **ansi))
        if 'thanks' in who:
            print(f"{ansi['bold']}Special Thanks To{ansi['reset']}", end='\n')
            for row in contributors['thanks']:
                print(f"    {ansi['reset']}{row[0]} {ansi['white']}from {row[1]} {ansi['bright_black']}for {row[2]}{ansi['reset']}")
    return None
