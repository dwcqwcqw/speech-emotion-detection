#!/usr/bin/env python3

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 现在导入应该能工作了
from app.train_model import main

if __name__ == "__main__":
    main()
