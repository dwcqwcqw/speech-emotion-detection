#!/usr/bin/env python3

import os
import sys
import shutil

def main():
    """
    Fix Python module import issues for Google Colab environment.
    This modifies the source files directly to use absolute imports.
    """
    # Ensure we're running from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print("Fixing import issues for Google Colab...")
    
    # Update train_model.py
    with open("app/train_model.py", "r") as f:
        content = f.read()
    
    # Replace relative imports with absolute imports
    content = content.replace("from utils.", "from app.utils.")
    
    with open("app/train_model.py", "w") as f:
        f.write(content)
    print("Updated app/train_model.py")
    
    # Update app.py
    with open("app/app.py", "r") as f:
        content = f.read()
    
    # Replace relative imports with absolute imports
    content = content.replace("from utils.", "from app.utils.")
    
    with open("app/app.py", "w") as f:
        f.write(content)
    print("Updated app/app.py")
    
    # Update evaluate_model.py
    with open("app/evaluate_model.py", "r") as f:
        content = f.read()
    
    # Replace relative imports with absolute imports
    content = content.replace("from utils.", "from app.utils.")
    
    with open("app/evaluate_model.py", "w") as f:
        f.write(content)
    print("Updated app/evaluate_model.py")
    
    # Create direct run scripts for Colab
    with open("colab_train.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys

# Add absolute path to the project
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import main function directly
from app.train_model import main

if __name__ == "__main__":
    main()
""")
    print("Created colab_train.py")
    
    with open("colab_evaluate.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys

# Add absolute path to the project
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import main function directly
from app.evaluate_model import main

if __name__ == "__main__":
    main()
""")
    print("Created colab_evaluate.py")
    
    with open("colab_app.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys
import subprocess

# Add absolute path to the project
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Run Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", 
                   os.path.join(project_root, "app", "app.py")])
""")
    print("Created colab_app.py")
    
    print("\nFix completed for Google Colab!")
    print("In Colab, run the following commands:")
    print("  python colab_train.py      # To train the model")
    print("  python colab_evaluate.py   # To evaluate the model")
    print("  python colab_app.py        # To run the web app")

if __name__ == "__main__":
    main() 