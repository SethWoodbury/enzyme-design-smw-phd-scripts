#!/usr/bin/env python3
"""
Step 1: Coordinate Transformation - Standalone Runner

Run directly from command line:
    python run_step1.py --input_pdb input.pdb --ref_pdb ref.pdb --output_pdb output.pdb

Or make executable and run:
    chmod +x run_step1.py
    ./run_step1.py --input_pdb input.pdb --ref_pdb ref.pdb --output_pdb output.pdb
"""

import sys
from pathlib import Path

# Add the package directory to Python path so imports work without pip install
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Now import and run the CLI
from remastered_fastmpnn.step1_coordinate_transform.cli import main

if __name__ == "__main__":
    sys.exit(main())
