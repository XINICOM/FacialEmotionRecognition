from flask_api import app
import os

# 解决 Windows 上 OpenMP 重复加载问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app.run(host='127.0.0.1', port=5000, debug=True, threaded=False)