from flask import Flask, jsonify, Response
import json

import src.config as cfg

from src.load_data import load_data as load_data_s
from src.preprocessing import preprocessing as preprocessing_s
from src.train import train as train_s
from src.predict import predict_image as predict_s
from src.control import pause as pause_s
from src.control import resume as resume_s
from src.control import terminate as terminate_s
app = Flask(__name__)




# ========== 控制接口 ==========

@app.route('/pause')
def pause():
    pause_s()
    print(f"调用了{pause.__name__}")
    return jsonify({"successful": "0"})


@app.route('/resume')
def resume():
    resume_s()
    print(f"调用了{resume.__name__}")
    return jsonify({"successful": "0"})

@app.route('/terminate')
def terminte():
    terminate_s()
    print(f"调用了{terminte.__name__}")
    return jsonify({"successful": "0"})


# ========== 启动与查询接口 ==========
train_x, train_y, val_x, val_y, test_x, test_y = 0, 0, 0, 0, 0, 0
@app.route('/load_data<arg>')
def load_data(arg):
    try:
        params = json.loads(arg)
        train_x, train_y, val_x, val_y, test_x, test_y, successful = load_data_s(params.get("load_path"))
    except Exception as e:
        successful = str(e)
    print(f"调用了{load_data.__name__}")
    return jsonify({"successful": successful})

train_loader, val_loader, test_loader = 0, 0, 0
@app.route('/preprocessing<arg>')
def preprocessing(arg):
    try:
        params = json.loads(arg)
        cfg.IMG_SIZE, cfg.BATCH_SIZE = params["IMG_SIZE"], params["BATCH_SIZE"]
        train_loader, val_loader, test_loader = preprocessing_s(train_x, train_y, val_x, val_y, test_x, test_y)
        successful = "0"
    except Exception as e:
        successful = str(e)
    return jsonify({"successful": successful})

@app.route('/train_stream<arg>', methods=['GET', 'POST'])
def train_stream(arg):
    try:
        params = json.loads(arg)
        (cfg.DEVICE,
         cfg.NUM_EPOCHS,
         cfg.LEARNING_RATE,
         cfg.EARLY_STOP_PATIENCE,
         cfg.MODEL_SAVE_PATH,
         cfg.EMOTION_LABELS) = (params.get("DEVICE"),
                                params.get("NUM_EPOCHS"),
                                params.get("LEARNING_RATE"),
                                params.get("EARLY_STOP_PATIENCE"),
                                params.get("MODEL_SAVE_PATH"),
                                params.get("EMOTION_LABELS"))
        print(f"调用了{train_stream.__name__}")
        return Response(
            train_s(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
    except Exception as e:
        return jsonify({"successful": str(e)})

@app.route('/predict<arg>')
def predict(arg):
    try:
        params = json.loads(arg)
        (cfg.IMG_SIZE, cfg.DEVICE,
         cfg.MODEL_SAVE_PATH, cfg.EMOTION_LABELS) = (params.get("IMG_SIZE"), params.get("DEVICE"),
                                                     params.get("MODEL_SAVE_PATH"), params.get("EMOTION_LABELS"))

        ans = predict_s(image_path=params.get("image_path"),)
        successful = "0"
    except Exception as e:
        successful = str(e)
        ans = 0
    return jsonify({"successful": successful, "predict": ans})

