#!/usr/bin/env python3
"""BIDS app to compute connectomes from fmri data preprocessed with FMRIPrep

Author: Antonin Rovai

Created: August 2022
"""

from connectomix.core.core import main
from connectomix.core.utils.tools import setup_terminal_colors

setup_terminal_colors()

if __name__ == "__main__":
    main()
