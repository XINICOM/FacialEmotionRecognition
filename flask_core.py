from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/api")
def hello():
    return "Hello World5！"

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'message': 'API服务正常'})

@app.route('/api/file/<path:filepath>')
def injection(filepath):
    return f'已接受文件路径: {filepath}'

@app.route('/api/json', methods=['POST'])
def json_injection():
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON data'}), 400
    arg1 = data.get('arg1')
    arg2 = data.get('arg2')
    arg3 = data.get('arg3')
    return jsonify({
        'message': '已获得JSON',
        'data': data
    }), 201

import time
import torch
@app.route("/api/cuda")
def cuda():
    start = time.time()
    # print("PyTorch 版本:", torch.__version__)
    # print("CUDA 是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        # print("CUDA 版本:", torch.version.cuda)
        # print("cuDNN 版本:", torch.backends.cudnn.version())
        # print("GPU 数量:", torch.cuda.device_count())
        # print("GPU 名称:", torch.cuda.get_device_name(0))
        a = torch.rand(1000, 1000).cuda()
        b = torch.rand(1000, 1000).cuda()
        c = torch.matmul(a, b)
        # print("GPU 矩阵乘法测试通过！")
        end = time.time()
        return jsonify({'result': f'{torch.__version__}','time': f'{end - start}'})
    else:
        end = time.time()
        # print("⚠️ CUDA 不可用，请检查 NVIDIA 驱动和 PyTorch 安装。")
        return jsonify({'result': 0,'time': f'{end - start}'})

if __name__ == "__main__":
    app.run(debug=True)