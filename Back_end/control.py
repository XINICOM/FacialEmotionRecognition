"""
训练控制模块 - 实现 pause / resume / terminate 功能
通过文件标志位控制训练流程，可在训练过程中手动操作
"""
import os
import time
from config import PAUSE_FLAG, RESUME_FLAG, TERMINATE_FLAG


def pause():
    """暂停训练: 创建暂停标志文件"""
    _clear_flag(RESUME_FLAG)
    _set_flag(PAUSE_FLAG)
    print("[control] 暂停信号已发送，训练将在当前epoch结束后暂停")


def resume():
    """恢复训练: 创建恢复标志文件并清除暂停标志"""
    _clear_flag(PAUSE_FLAG)
    _set_flag(RESUME_FLAG)
    print("[control] 恢复信号已发送，训练将继续")


def terminate():
    """终止训练: 创建终止标志文件"""
    _set_flag(TERMINATE_FLAG)
    print("[control] 终止信号已发送，训练将在当前epoch结束后终止")


def check_pause():
    """检查是否需要暂停，如果暂停则等待恢复信号"""
    while os.path.exists(PAUSE_FLAG):
        print("[control] 训练已暂停，等待恢复... (删除 pause.flag 或创建 resume.flag 恢复)")
        time.sleep(2)
        if os.path.exists(RESUME_FLAG):
            _clear_flag(PAUSE_FLAG)
            _clear_flag(RESUME_FLAG)
            print("[control] 训练已恢复!")
            break


def check_terminate():
    """检查是否需要终止，返回True表示需要终止"""
    if os.path.exists(TERMINATE_FLAG):
        _clear_flag(TERMINATE_FLAG)
        _clear_flag(PAUSE_FLAG)
        _clear_flag(RESUME_FLAG)
        return True
    return False


def clear_all_flags():
    """清除所有控制标志"""
    _clear_flag(PAUSE_FLAG)
    _clear_flag(RESUME_FLAG)
    _clear_flag(TERMINATE_FLAG)


def _set_flag(path):
    with open(path, "w") as f:
        f.write("1")


def _clear_flag(path):
    if os.path.exists(path):
        os.remove(path)
