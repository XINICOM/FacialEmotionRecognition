import sys
from flask_api import app
import os

# 解决 Windows 上 OpenMP 重复加载问题
if "__main__" == __name__:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if len(sys.argv) < 2:
        print("请传入端口号，例如: py main.py 5000")
        sys.exit(1)

    # 获取第二个参数（索引 1）
    port = int(sys.argv[1])
    # port = 5000
    print(f"收到端口参数: {port}")
    app.run(host='127.0.0.1', port=port, threaded=False)