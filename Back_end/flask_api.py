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
@app.route('/load_data')
def load_data():
    cfg_or_err = handle_str_config()
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
@app.route('/preprocessing')
def preprocessing():
    cfg_or_err = handle_str_config()
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err
    train_loader, val_loader, ans = preprocessing_s(cfg=cfg, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    return ans


@app.route('/predict')
def predict():
    cfg_or_err = handle_str_config()
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err
    return predict_s(cfg=cfg)


@app.route('/train_stream', methods=['GET', 'POST'])
def train_stream():
    cfg_or_err = handle_str_config()
    if not isinstance(cfg_or_err, dict):
        return cfg_or_err
    cfg = cfg_or_err


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