import threading

class TaskController:
    """单个任务的控制状态（暂停/继续/终止）"""
    def __init__(self):
        self._state = 0          # 0=运行, 1=暂停, -1=终止
        self._cond = threading.Condition()

    def pause(self):
        with self._cond:
            self._state = 1

    def resume(self):
        with self._cond:
            self._state = 0
            self._cond.notify()

    def terminate(self):
        with self._cond:
            self._state = -1
            self._cond.notify_all()

    def wait_if_paused_or_terminated(self):
        """在任务循环中调用，返回 True 表示需要终止"""
        with self._cond:
            while self._state == 1:
                self._cond.wait()
            return self._state == -1

    def get_state(self):
        """返回当前状态：0=运行, 1=暂停, -1=终止"""
        with self._cond:
            return self._state


controller = []
_ctrl_lock = threading.Lock()

def get_controller() -> TaskController:
    with _ctrl_lock:
        if not controller:
            controller.append(TaskController())
        return controller[0]
