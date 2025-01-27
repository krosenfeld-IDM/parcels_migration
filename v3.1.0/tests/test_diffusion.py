import sys
import pytest
import numpy as np
from datetime import timedelta
from pathlib import Path
import parcels

root_dir = Path(__file__).resolve().parents[1]
sys.path.append((root_dir / 'scripts').as_posix())
from example_diffusion import basic_diffusion_example, smagdiff_example

def test_basic_diffusion_example():
    basic_diffusion_example()

def test_smagdiff_example():
    smagdiff_example()