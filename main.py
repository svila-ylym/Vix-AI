import os
import datetime
import logging
import pathlib
import sys
import time
from dotenv import load_dotenv
# 获取调试标志
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
logger = logging.getLogger("Main")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(log_dir, "Main.log"), encoding="utf-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | Main | %(levelname)s ‖ %(message)s', "%Y-%m-%d %H:%M:%S")
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
    else:
        logger.error("创建.env文件失败")
    sys.exit(0)

import colorama
from colorama import Fore, Style
from app.services import llm_services
colorama.init()
import platform

# 引入路由模块
from app.services.router_service import app as router_app
from fastapi import FastAPI
from app.api import app as api_app  # 修改为绝对导入
from app.models import create_db  # 修改为绝对导入

app = FastAPI()

app.mount("/api", api_app)

@app.on_event("startup")
async def startup_event():
    create_db()

# 定义
version = "0.0.1.main"
component = "Main"

def printf(text, level):
    if level == 1:
        logger.info(text)
    elif level == 2:
        logger.debug(text)
    elif level == 3:
        logger.warning(text)
    elif level == 4:
        logger.error(text)
    else:
        logger.info(text)

def load_init():
    printf("正在检查您的版本...",2)
    system = platform.system()      # Windows, Linux, Darwin (macOS)
    release = platform.release()    # 系统版本号
    version = platform.version()    # 详细版本信息

    # 获取架构信息
    architecture = platform.architecture()  # ('64bit', 'ELF')
    machine = platform.machine()            # x86_64, AMD64, arm64
    processor = platform.processor()        # 处理器信息
    platforms = platform.platform()

    printf("===============",2)
    printf("查询到以下信息：",2)
    printf(platforms,2)
    printf(architecture,2)
    printf(machine,2)
    printf(processor,2)
    printf("===============",2)

def load_llm_model():
    chat_config = llm_services.get_llm_config("llm.chat")
    if chat_config:
        llm_services.printf(f"聊天主模型配置: {chat_config}", 1)
            # 直接获取特定值
        model_name = llm_services.get_llm_config_value("llm.chat", "name", "默认模型")
        model_path = llm_services.get_llm_config_value("llm.chat", "model", "unknown/model")
        max_tokens = llm_services.get_llm_config_value("llm.chat", "max_token", 512)
        model_temp = llm_services.get_llm_config_value("llm.chat", "temp", 0.7)
        llm_services.printf("===============",1)
        llm_services.printf(f"获取到以下主模型配置：", 1)
        llm_services.printf(f"模型名称: {model_name}", 1)
        llm_services.printf(f"模型路径: {model_path}", 1)
        llm_services.printf(f"最大token数: {max_tokens}", 1)
        llm_services.printf(f"模型温度：{model_temp}", 1)
        llm_services.printf("===============",1)
    
# 获取嵌入配置
    schedule_config = llm_services.get_llm_config("llm.schedule")
    if schedule_config:
        model_name = llm_services.get_llm_config_value("llm.schedule", "name", "默认模型")
        model_path = llm_services.get_llm_config_value("llm.schedule", "model", "unknown/model")
        max_tokens = llm_services.get_llm_config_value("llm.schedule", "max_token", 512)
        model_temp = llm_services.get_llm_config_value("llm.schedule", "temp", 0.7)
        # printf("===============",1)
        llm_services.printf(f"日程模型配置：", 1)
        llm_services.printf(f"模型名称: {model_name}", 1)
        llm_services.printf(f"模型路径: {model_path}", 1)
        llm_services.printf(f"最大token数: {max_tokens}", 1)
        llm_services.printf(f"模型温度：{model_temp}", 1)
        llm_services.printf("===============",1)
    
    # 可以轻松获取其他LLM配置
    api_config = llm_services.get_llm_config("llm.api")
    if api_config:
        llm_services.printf(f"API配置: {api_config}", 1)

def requests_json_chat(messages, conversation_id="default"):
    result = llm_services.requestsChatllm(messages, conversation_id)
    if result["success"]:
        printf(f"对话ID: {result['conversation_id']}", 1)
        printf(f"回复内容: {result['content']}", 1)
        return result["conversation_id"]  # 返回对话ID用于后续对话
    else:
        printf(f"请求失败: {result.get('error', '未知错误')}", 3)
        return None

def version_level():
    
    if int(llm_services.get_llm_config_value("vt.config", "version", 0)) >= 4:
        config_version = int(llm_services.get_llm_config_value("vt.config", "version", 0))
        printf(f"配置版本：{config_version}",2)
        printf(f"非常的新鲜，非常的美味。",1)
    elif int(llm_services.get_llm_config_value("vt.config", "version", 0)) == 0 or int(llm_services.get_llm_config_value("vt.config", "version", 0)) == None or int(llm_services.get_llm_config_value("vt.config", "version", 0)) == "":
        printf("无效的配置文件",4)
    else:
        printf(f"你给我干哪来了？这配置不好吃啊",3)
    if "dev" in llm_services.version:
        llm_services.printf(f"LLM模块当前使用的版本号是一个开发版本，可能含有不稳定性。版本号：{llm_services.version}",3)
    elif "fix" in llm_services.version:
        llm_services.printf(f"LLM模块版本：{llm_services.version}。该版本似乎是一个修复版本。",1)
    elif "RC" in llm_services.version:
        llm_services.printf(f"LLM模块版本：{llm_services.version}这是一个备选版本。",1)
    else:
        llm_services.printf(f"LLM模块版本号：{llm_services.version}",1)

def interactive_chat():
    """交互式聊天模式"""
    printf("进入交互式聊天模式，输入 '退出' 或 'exit' 结束对话", 1)
    printf("输入 '清空' 或 'clear' 清空当前对话历史", 1)
    
    conversation_id = "interactive_chat"
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if user_input.lower() in ['退出', 'exit', 'quit']:
                printf("结束对话", 1)
                break
            elif user_input.lower() in ['清空', 'clear']:
                llm_services.clear_conversation_history(conversation_id)
                printf("已清空当前对话历史", 1)
                continue
            elif not user_input:
                continue
                
            # 发送消息并获取回复
            result = llm_services.requestsChatllm(user_input, conversation_id)
            
            if result["success"]:
                printf(f"AI: {result['content']}", 1)
            else:
                printf(f"请求失败: {result.get('error', '未知错误')}", 3)
                
        except KeyboardInterrupt:
            printf("\n用户中断对话", 2)
            break
        except Exception as e:
            printf(f"发生错误: {e}", 3)

# 主程序
if __name__ == "__main__":
    load_init()
    load_llm_model()
    version_level()
    # interactive_chat()  # 注释掉原有的交互式聊天
    router_app.run(debug=DEBUG_MODE, host='0.0.0.0', port=8000)  # 启动 Flask 应用