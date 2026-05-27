from flask import Flask, jsonify, Response

from test_func import *
import test_config as cfg

app = Flask(__name__)
# ========== 控制接口 ==========

@app.route('/test<arg>')
def test(arg):
    cfg.I = arg
    print(f"调用了{test.__name__}")
    ans = func()
    return jsonify(ans)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=False)