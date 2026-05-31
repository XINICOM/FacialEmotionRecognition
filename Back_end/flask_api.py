from flask import Flask, jsonify
import threading
import json
import pandas as pd

from shared_state import get_controller
from model_src.src.load_config_from_str import handle_str_config
from model_src.src.load_data import load_fer2013 as load_data_s
from model_src.src.preprocessing import preprocessing as preprocessing_s
from model_src.src.train import train_stream_packer_console as train_s
from model_src.src.predict import predict as predict_s


app = Flask(__name__)


@app.route('/pause')
def pause():
    print(f"invoke {pause.__name__}")
    ctrl = get_controller()
    ctrl.pause()
    return jsonify({"successful": "0"})
    return "pause successful"


@app.route('/resume')
def resume():
    print(f"invoke {resume.__name__}")
    ctrl = get_controller()
    ctrl.resume()
    return jsonify({"successful": "0"})
    return "resume successful"


@app.route('/terminate')
def terminate():
    print(f"invoke {terminate.__name__}")
    ctrl = get_controller()
    ctrl.terminate()
    return jsonify({"successful": "0"})
    return "terminate successful"

x_train, x_val, y_train, y_val, x_test, y_test = 0,0,0,0,0,0
@app.route('/load_data/<arg>')
def load_data(arg):
    try:
        print(f"invoke {load_data.__name__}")
        cfg = json.loads(arg)
        global x_train, x_val, y_train, y_val, x_test, y_test
        x_train, x_val, y_train, y_val, x_test, y_test = load_data_s(cfg=cfg)
        return jsonify({"successful": "0"})
    except Exception as e:
        return jsonify({"successful": str(e)})

train_loader, val_loader = 0, 0
@app.route('/preprocessing/<arg>')
def preprocessing(arg):
    try:
        print(f"invoke {preprocessing.__name__}")
        cfg = json.loads(arg)
        global train_loader, val_loader
        train_loader, val_loader = preprocessing_s(cfg=cfg, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        return jsonify({"successful": "0"})
    except Exception as e:
        return jsonify({"successful": str(e)})


@app.route('/predict/<arg>')
def predict(arg):
    try:
        print(f"invoke {predict.__name__}")
        cfg = json.loads(arg)
        pre = predict_s(cfg=cfg)
        return jsonify({"successful": "0", "prediction": str(pre)})
    except Exception as e:
        return jsonify({"successful": str(e)})

model_layers = {"features":[],"classifier":[]}
@app.route('/add_conv_layer/<arg>')
def add_conv_layer(arg):
    try:
        print(f"invoke {add_conv_layer.__name__}")
        cfg = json.loads(arg)
        list = ["conv"]
        list_index = ["out_channels", "kernel_size", "padding", "stride", "act", "bn_lr_factor", "drop2d", "lr"]
        for index in list_index:
            list.append(cfg[index])
        model_layers["features"].append(list)
        return jsonify({"successful": "0"})
    except Exception as e:
        return jsonify({"successful": str(e)})

@app.route('/add_pool_layer/<arg>')
def add_pool_layer(arg):
    try:
        print(f"invoke {add_pool_layer.__name__}")
        cfg = json.loads(arg)
        list = ["pool"]
        list_index = ["mode", "kernel_size", "stride"]
        for index in list_index:
            list.append(cfg[index])
        model_layers["features"].append(list)
        return jsonify({"successful": "0"})
    except Exception as e:
        return jsonify({"successful": str(e)})


@app.route('/add_linear_layer/<arg>')
def add_linear_layer(arg):
    try:
        print(f"invoke {add_linear_layer.__name__}")
        cfg = json.loads(arg)
        list = ["linear"]
        list_index = ["h_dim", "act", "drop", "lr"]
        for index in list_index:
            list.append(cfg[index])
        model_layers["classifier"].append(list)
        return jsonify({"successful": "0"})
    except Exception as e:
        return jsonify({"successful": str(e)})


@app.route('/train_stream/<arg>', methods=['GET', 'POST'])
def train_stream(arg):
    try:
        print(f"invoke {train_stream.__name__}")
        cfg = json.loads(arg)
        cfg["model_layers"] = model_layers
        t = threading.Thread(
            target=train_s,
            kwargs={"cfg": cfg, "train_loader": train_loader, "val_loader": val_loader},
            daemon=True
        )
        t.start()
        return jsonify({"successful": "0"})
    except Exception as e:
        return jsonify({"successful": str(e)})


# @app.route('/train')
# def train():
#     cfg_or_err = handle_str_config()
#     if not isinstance(cfg_or_err, dict):
#         return cfg_or_err
#     cfg = cfg_or_err
#     ans = train_s(cfg=cfg, train_loader=train_loader, val_loader=val_loader)
#     return ans