import sys
import pytest
import numpy as np
from datetime import timedelta
from pathlib import Path
import parcels

root_dir = Path(__file__).resolve().parents[1]
sys.path.append((root_dir / 'scripts').as_posix())
from example_diffusion import main

def test_main():
    main()