import os
import requests
import traceback
import sys
import datetime
import colorama
from colorama import Fore, Style
import toml
import json
import logging
import pathlib
from dotenv import load_dotenv
from .vl_services import *

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

DEBUG_MODE = get_debug_flag()

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("CutTool")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(log_dir, "CutTool.log"), encoding="utf-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | CutTool | %(levelname)s ‖ %(message)s', "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
if DEBUG_MODE:
    logger.addHandler(ch)

def create_default_env():
    """创建默认的.env文件"""
    default_env = """# API配置
url=http://localhost:8000/v1/chat/completions
key=your-api-key-here

# 角色配置
name=Vix
nickname=小V
age=18
personality_core=友善、活泼、富有同理心
lang_style=活泼、可爱、但不失礼貌

# 系统配置
cut_num=8"""
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(default_env)
    return True

if os.path.exists(".env"):
    load_dotenv(".env", override=True)
    if DEBUG_MODE:
        logger.info("成功加载环境变量配置")
else:
    logger.warning("未找到.env文件，正在创建默认配置...")
    if create_default_env():
        logger.info("已创建默认.env文件，请修改其中的配置后重新运行程序")
        sys.exit(0)
    else:
        logger.error("创建.env文件失败")
        raise FileNotFoundError(".env 文件创建失败")

# 初始化colorama
colorama.init()

# 初始化版本信息和API
version = "0.0.1.dev"
component = "CutTool"
api_url = os.getenv('url')
api_key = os.getenv('key')

# 全局对话历史存储
conversation_history = {}

# 日程文件路径
SCHEDULE_FILE = "daily_schedule.json"

# 定义函数
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

def requestsFellingllm(messages, cut_num=8):
    # 获取当前使用的模型配置
    cut_config = get_llm_config("llm.cut")
    provider = cut_config.get("provider", "local")
    
    # 获取提供商配置
    provider_config = get_provider_config(provider)
    url = provider_config["url"]
    api_key = provider_config["key"]
    headers = provider_config["headers"](api_key)

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = f'''
    请你将以下消息分割成QQ的聊天消息，最多分{cut_num}段（或者更低，只要分段了并且不超过{cut_num}就可以）。中间用15条“-”分割，比如：
    Message 1
    ---------------
    Message 2
    ---------------
（注意！！！！！！！！！！只分割传入的消息，不要做补充（段落不够{cut_num}可以分成更小的！！！也不要重复！！！）
不输出多余的内容或者注释等。必须严格按照15个“-”分割
    现在，分割这些消息
    {messages}
    '''

    printf(system_prompt,2)

    payload = {
        "model": get_llm_config_value("llm.cut", "model", "unknown/model"),
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
        ],
        "enable_thinking": get_llm_config_value("llm.cut", "enable_thinking", None),
        "max_tokens": get_llm_config_value("llm.cut", "max_token", 512),
        "temperature": get_llm_config_value("llm.cut", "temp", 0.7)
    }

    if provider == "azure":
        payload["deployment_id"] = os.getenv('azure_deployment')

    try:
        printf(f"使用提供商: {provider}", 2)
        printf(f"请求URL: {url}", 2)
        printf(f"请求头: {headers}", 2)
        printf(f"请求体: {payload}", 2)
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        printf(f"响应状态码: {response.status_code}", 2)
        
        if response.status_code != 200:
            printf(f"响应内容: {response.text}", 3)
            return "无法分割文段"
        
        response_data = response.json()
        
        # 根据不同提供商处理响应
        if provider == "anthropic":
            content = response_data.get("content", [{"text": "No response"}])[0]["text"]
        elif provider == "azure":
            content = response_data["choices"][0]["message"]["content"] if response_data.get("choices") else "No response"
        else:
            content = response_data["choices"][0]["message"]["content"] if response_data.get("choices") else "No response"
        
        return content

    except requests.exceptions.RequestException as e:
        printf(f"网络请求错误: {e}", 4)
        return "文段生成失败"
    except ValueError as e:
        printf(f"JSON解析错误: {e}", 4)
        return "文段生成失败"
    except Exception as e:
        printf(f"未知错误: {e}", 4)
        traceback.print_exc()
        return "文段生成失败"