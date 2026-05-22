import json5
from flask import request


def handle_str_config(cfg_str):
    if not cfg_str:
        return "接收到的配置字符串为空"
    if not isinstance(cfg_str, str):
        return f"期待收到 str 类型的配置，但收到了 {type(cfg_str)}"
    try:
        cfg = json5.loads(cfg_str)
        return cfg
    except Exception as e:
        return f"JSON5 语法解析失败，请检查前端传来的字符串格式是否正确。错误信息: {e}"
