#!/usr/bin/env python3

import os
import sys
import subprocess

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # 使用subprocess运行Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"])
