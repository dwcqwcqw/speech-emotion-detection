#!/usr/bin/env python3

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now imports should work
from app.evaluate_model import main

if __name__ == "__main__":
    main()
