#!/usr/bin/env python3

import os
import sys
import shutil

def main():
    """
    Fix Python module import issues to ensure the app package can be properly imported.
    The solution is to create convenience scripts with the correct import paths.
    """
    # Ensure we're running from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print("Fixing module import issues...")
    
    # Check if app directory is a package
    app_init = os.path.join(project_root, "app", "__init__.py")
    if not os.path.exists(app_init):
        print(f"Error: {app_init} doesn't exist. Make sure app directory has an __init__.py file.")
        return
    
    # Check if utils directory is a package
    utils_init = os.path.join(project_root, "app", "utils", "__init__.py")
    if not os.path.exists(utils_init):
        print(f"Error: {utils_init} doesn't exist. Make sure app/utils directory has an __init__.py file.")
        return
    
    # Solution 1: Add project root to PYTHONPATH (temporary solution)
    print("Creating run scripts...")
    
    with open("run_train.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now imports should work
from app.train_model import main

if __name__ == "__main__":
    main()
""")
    
    # Set executable permissions
    os.chmod("run_train.py", 0o755)
    
    with open("run_evaluate.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now imports should work
from app.evaluate_model import main

if __name__ == "__main__":
    main()
""")
    
    # Set executable permissions
    os.chmod("run_evaluate.py", 0o755)
    
    with open("run_app.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys
import subprocess

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Run Streamlit app with subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"])
""")
    
    # Set executable permissions
    os.chmod("run_app.py", 0o755)
    
    print("Fix completed!")
    print("You can now train the model with:")
    print("  python run_train.py")
    print("Or evaluate the model with:")
    print("  python run_evaluate.py")
    print("Or run the web application with:")
    print("  python run_app.py")

if __name__ == "__main__":
    main() 