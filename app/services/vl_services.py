import os
import toml
import datetime
import logging
import pathlib
from dotenv import load_dotenv

def get_debug_flag():
    config_path = "config/config.toml"
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("debug = true\n")
        return True
    try:
        import toml
        config = toml.load(config_path)
        return bool(config.get("debug", True))
    except Exception:
        return True
    return True

DEBUG_MODE = get_debug_flag()

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("llm_VL")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(log_dir, "llm_VL.log"), encoding="utf-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | llm_VL | %(levelname)s ‖ %(message)s', "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
if DEBUG_MODE:
    logger.addHandler(ch)

if os.path.exists(".env"):
    load_dotenv(".env", override=True)
    if DEBUG_MODE:
        logger.info("成功加载环境变量配置")
else:
    logger.error("未找到.env文件，请确保程序所需的环境变量被正确设置")
    raise FileNotFoundError(".env 文件不存在，请创建并配置所需的环境变量")

# 初始化版本信息和API
version = "0.0.1.dev"
component = "llm_VL"

def get_provider_config(provider_name):
    """获取指定提供商的配置"""
    provider_configs = {
        "openai": {
            "url": os.getenv('openai_api_url'),
            "key": os.getenv('openai_api_key'),
            "headers": lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
        },
        "anthropic": {
            "url": os.getenv('anthropic_api_url'),
            "key": os.getenv('anthropic_api_key'),
            "headers": lambda key: {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
        },
        "azure": {
            "url": os.getenv('azure_api_url'),
            "key": os.getenv('azure_api_key'),
            "headers": lambda key: {
                "api-key": key,
                "Content-Type": "application/json"
            }
        },
        "local": {
            "url": os.getenv('local_api_url'),
            "key": os.getenv('local_api_key'),
            "headers": lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
        },
        "siliconflow": {
            "url": os.getenv('siliconflow_api_url', 'https://api.siliconflow.cn/v1/chat/completions'),
            "key": os.getenv('siliconflow_api_key'),
            "headers": lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
        }
    }
    return provider_configs.get(provider_name, provider_configs["siliconflow"])

def printf(text, level):
    if level == 1:
        logger.info(text)
    elif level == 2:
        logger.debug(text)
        if DEBUG_MODE:
            for h in logger.handlers:
                if isinstance(h, logging.StreamHandler):
                    h.emit(logging.LogRecord(logger.name, logging.DEBUG, "", 0, text, None, None))
    elif level == 3:
        logger.warning(text)
    elif level == 4:
        logger.error(text)
    else:
        logger.info(text)

printf("成功接入LLM/VL", 2)

def getLlmConfig(file_path):
    """读取整个TOML配置文件，不进行过滤"""
    try:
        # 读取TOML文件
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = toml.load(f)  # 将TOML文件内容解析为Python字典
        
        printf(f"成功读取配置文件，包含以下键: {list(config_data.keys())}", 2)
        return config_data
        
    except FileNotFoundError:
        printf(f"配置文件未找到: {file_path}", 4)
        return {}
    except Exception as e:
        printf(f"读取配置文件时出错: {e}", 4)
        return {}

# 读取配置
llm_config = getLlmConfig("config/llm.toml")

def get_llm_config(config_key="llm.chat", default_config=None):
    """
    获取指定的LLM配置
    
    Args:
        config_key: 配置键名，如 "llm.chat", "llm.embedding" 等
        default_config: 如果配置不存在时返回的默认配置
    
    Returns:
        配置字典，如果不存在则返回默认配置或空字典
    """
    if default_config is None:
        default_config = {}
    
    # 读取配置
    llm_config = getLlmConfig("config/llm.toml")
    
    # 第一种方式：直接访问 llm.chat 格式
    if config_key in llm_config:
        config_data = llm_config[config_key]
        printf(f"找到 {config_key} 配置: {config_data}", 2)
        return config_data
    
    # 第二种方式：访问 llm['chat'] 格式
    elif "." in config_key:
        main_key, sub_key = config_key.split(".", 1)
        if (main_key in llm_config and 
            isinstance(llm_config[main_key], dict) and 
            sub_key in llm_config[main_key]):
            config_data = llm_config[main_key][sub_key]
            printf(f"找到 {main_key}['{sub_key}'] 配置: {config_data}", 2)
            return config_data
    
    # 未找到配置
    printf(f"未找到 {config_key} 配置", 3)
    printf(f"可用的配置键: {list(llm_config.keys())}", 2)
    return default_config

def get_llm_config_value(config_key, value_key, default_value=None):
    """
    获取LLM配置中的特定值
    
    Args:
        config_key: 配置键名，如 "llm.chat"
        value_key: 值键名，如 "name", "model"
        default_value: 默认值
    
    Returns:
        配置值，如果不存在则返回默认值
    """
    config = get_llm_config(config_key, {})
    return config.get(value_key, default_value)

printf("加载成功",1)