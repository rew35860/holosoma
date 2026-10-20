"""Run the current signed-contact solver in a new regression directory."""
import sys
from config import *

implementation = module('numerics_signed_solver', SIGNED_CODE / 'solver.py')
load, target_at = implementation.load, implementation.target_at

if __name__ == '__main__':
    implementation.main(*sys.argv[1:])
