from flask import Flask, jsonify
import threading
import json

import src.config as cfg

from src.load_data import load_data as load_data_s
from src.preprocessing import preprocessing as preprocessing_s
from src.train import do_train as train_s
from src.predict import predict_image as predict_s
from src.share_state import get_controller
app = Flask(__name__)




# ========== 控制接口 ==========

@app.route('/pause')
def pause():
    print(f"调用了{pause.__name__}")
    ctrl = get_controller()
    ctrl.pause()
    return jsonify({"successful": "0"})


@app.route('/resume')
def resume():
    print(f"调用了{resume.__name__}")
    ctrl = get_controller()
    ctrl.resume()
    return jsonify({"successful": "0"})

@app.route('/terminate')
def terminte():
    print(f"调用了{terminte.__name__}")
    ctrl = get_controller()
    ctrl.terminate()
    return jsonify({"successful": "0"})


# ========== 启动与查询接口 ==========
train_x, train_y, val_x, val_y, test_x, test_y = 0, 0, 0, 0, 0, 0
@app.route('/load_data/<arg>')
def load_data(arg):
    try:
        params = json.loads(arg)
        global train_x, train_y, val_x, val_y, test_x, test_y
        train_x, train_y, val_x, val_y, test_x, test_y = load_data_s(params.get("load_path").replace(",,", "/"))
        return jsonify({"successful": "0"})
    except Exception as e:
        return jsonify({"successful": str(e)})

train_loader, val_loader, test_loader = 0, 0, 0
@app.route('/preprocessing/<arg>')
def preprocessing(arg):
    try:
        params = json.loads(arg)
        cfg.IMG_SIZE, cfg.BATCH_SIZE = params["IMG_SIZE"], params["BATCH_SIZE"]
        global train_loader, val_loader, test_loader
        train_loader, val_loader, test_loader = preprocessing_s(train_x, train_y, val_x, val_y, test_x, test_y)
        return jsonify({"successful": "0"})
    except Exception as e:
        return jsonify({"successful": str(e)})

@app.route('/train_stream/<arg>', methods=['GET', 'POST'])
def train_stream(arg):
    try:
        params = json.loads(arg)
        (cfg.NUM_EPOCHS,
         cfg.LEARNING_RATE,
         cfg.EARLY_STOP_PATIENCE,
         cfg.MODEL_SAVE_PATH,) = (params.get("NUM_EPOCHS"),
                                params.get("LEARNING_RATE"),
                                params.get("EARLY_STOP_PATIENCE"),
                                params.get("MODEL_SAVE_PATH").replace(",,", "/"),)
        print(f"调用了{train_stream.__name__}")
        global train_loader, val_loader
        t = threading.Thread(
            target=train_s,
            kwargs={"train_loader": train_loader, "val_loader": val_loader},
            daemon=True
        )
        t.start()
        return jsonify({"successful": "0"})
    except Exception as e:
        return jsonify({"successful": str(e)})

@app.route('/predict/<arg>')
def predict(arg):
    try:
        params = json.loads(arg)
        (cfg.IMG_SIZE, cfg.DEVICE,
         cfg.MODEL_SAVE_PATH, cfg.EMOTION_LABELS) = (params.get("IMG_SIZE"), params.get("DEVICE"),
                                                     params.get("MODEL_SAVE_PATH").replace(",,", "/"), params.get("EMOTION_LABELS"))

        ans = predict_s(image_path=params.get("image_path").replace(",,", "/"),)
        return jsonify({"successful": "0", "predict": ans})
    except Exception as e:
        return jsonify({"successful": str(e)})

