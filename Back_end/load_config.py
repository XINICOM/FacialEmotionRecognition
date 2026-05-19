import json5

def load_config(config_path='Back_end/config.json5'):
    """
    加载 JSON5 配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json5.load(f)
    return config