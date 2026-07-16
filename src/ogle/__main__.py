"""Enable `python -m ogle …` as an alias for the `ogle` console script."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
