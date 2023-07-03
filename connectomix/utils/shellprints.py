#!/usr/bin/env python3

# IMPORTS

from datetime import datetime
import os  # to interact with dirs (join paths etc)
import subprocess  # to call bin outside of python
from collections import defaultdict
import sys

# CLASSES


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    ERROR = BOLD + RED
    INFO = BOLD + GREEN
    WARNING = BOLD + YELLOW

# FUNCTIONS

def msg_info(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    print(f"{BColors.INFO}INFO{BColors.ENDC} " + time_stamp + msg, flush=True)


def msg_error(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    print(f"{BColors.ERROR}ERROR{BColors.ENDC} " + time_stamp + msg, flush=True)


def msg_warning(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    print(f"{BColors.WARNING}WARNING{BColors.ENDC} " + time_stamp + msg, flush=True)


def printProgressBar (iteration, total, prefix = '', suffix = 'Complete', decimals = 4, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    complete_prefix = f"{BColors.OKCYAN}Progress {BColors.ENDC}" + prefix
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{complete_prefix} |{bar}| {percent}% {suffix}', end = printEnd, flush="True")
    # Print New Line on Complete
    if iteration == total:
        print()
