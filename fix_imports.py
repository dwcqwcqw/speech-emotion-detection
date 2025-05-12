#!/usr/bin/env python3

import os
import sys
import shutil

def main():
    """
    修复Python模块导入问题，确保app包可以正确导入。
    解决方案是在项目根目录创建一个包导入文件。
    """
    # 确保在项目根目录运行
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print("正在修复模块导入问题...")
    
    # 检查app目录是否是一个包
    app_init = os.path.join(project_root, "app", "__init__.py")
    if not os.path.exists(app_init):
        print(f"错误：{app_init} 不存在，请确保app目录中有__init__.py文件。")
        return
    
    # 检查utils目录是否是一个包
    utils_init = os.path.join(project_root, "app", "utils", "__init__.py")
    if not os.path.exists(utils_init):
        print(f"错误：{utils_init} 不存在，请确保app/utils目录中有__init__.py文件。")
        return
    
    # 解决方案1: 在PYTHONPATH中添加项目根目录（临时解决方案）
    print("创建运行脚本...")
    
    with open("run_train.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 现在导入应该能工作了
from app.train_model import main

if __name__ == "__main__":
    main()
""")
    
    # 设置可执行权限
    os.chmod("run_train.py", 0o755)
    
    with open("run_app.py", "w") as f:
        f.write("""#!/usr/bin/env python3

import os
import sys
import subprocess

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # 使用subprocess运行Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"])
""")
    
    # 设置可执行权限
    os.chmod("run_app.py", 0o755)
    
    print("修复完成！")
    print("现在可以通过以下命令训练模型：")
    print("  python run_train.py")
    print("或者运行Web应用：")
    print("  python run_app.py")

if __name__ == "__main__":
    main() 