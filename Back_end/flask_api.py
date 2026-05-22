from flask import Flask, jsonify, request, Response
import pandas as pd
from shared_state import get_controller
from Back_end.model_src.src.load_config_from_str import handle_str_config
from Back_end.model_src.src.load_data import load_fer2013 as load_data_s
from Back_end.model_src.src.preprocessing import preprocessing as preprocessing_s
from Back_end.model_src.src.train import train_stream_packer as train_s
from Back_end.model_src.src.predict import predict as predict_s


app = Flask(__name__)


@app.route('/pause')
def pause():
    ctrl = get_controller
    ctrl.pause()
    return "0"
    return "pause successful"


@app.route('/resume')
def resume():
    ctrl = get_controller
    ctrl.resume()
    return "0"
    return "resume successful"


@app.route('/stop')
def stop():
    ctrl = get_controller
    ctrl.terminate()
    return "0"
    return "terminate successful"

x_train, x_val, y_train, y_val = 0, 0, 0, 0
@app.route('/load_data<arg>')
def load_data(arg):
    cfg_or_err = handle_str_config(arg)
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err
    try:
        df = pd.read_csv(cfg["load_path"])
    except FileNotFoundError:
        return f"csv文件未找到,搜索路径 {cfg['load_path']}"
    x_train, x_val, y_train, y_val, ans = load_data_s(df=df, cfg=cfg)
    return ans

train_loader, val_loader = 0, 0
@app.route('/preprocessing<arg>')
def preprocessing(arg):
    cfg_or_err = handle_str_config(arg)
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err
    train_loader, val_loader, ans = preprocessing_s(cfg=cfg, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    return ans


@app.route('/predict<arg>')
def predict(arg):
    cfg_or_err = handle_str_config(arg)
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err
    return predict_s(cfg=cfg)

model_layers = {"features":[],"classifier":[]}
@app.route('/add_conv_layer<arg>')
def add_conv_layer(arg):
    cfg_or_err = handle_str_config(arg)
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err
    list = ["conv"]
    list_index = ["out_channels", "kernel_size", "padding", "stride", "act", "bn_lr_factor", "drop2d", "lr"]
    for index in list_index:
        list.append(cfg[index])
    model_layers["features"].append(list)
    return jsonify("0")

@app.route('/add_pool_layer<arg>')
def add_pool_layer(arg):
    cfg_or_err = handle_str_config(arg)
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err
    list = ["pool"]
    list_index = ["mode", "kernel_size", "stride"]
    for index in list_index:
        list.append(cfg[index])
    model_layers["features"].append(list)
    return jsonify("0")


@app.route('/add_linear_layer<arg>')
def add_linear_layer(arg):
    cfg_or_err = handle_str_config(arg)
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err
    list = ["linear"]
    list_index = ["h_dim", "act", "drop", "lr"]
    for index in list_index:
        list.append(cfg[index])
    model_layers["classifier"].append(list)
    return jsonify("0")


@app.route('/train_stream<arg>', methods=['GET', 'POST'])
def train_stream(arg):
    cfg_or_err = handle_str_config(arg)
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err
    cfg["model_layers"] = model_layers

    return Response(
        train_s(cfg=cfg, train_loader=train_loader, val_loader=val_loader),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


# @app.route('/train')
# def train():
#     cfg_or_err = handle_str_config()
#     if not isinstance(cfg_or_err, dict):
#         return cfg_or_err
#     cfg = cfg_or_err
#     ans = train_s(cfg=cfg, train_loader=train_loader, val_loader=val_loader)
#     return ans