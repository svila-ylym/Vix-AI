import os
import time
import logging
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, abort

from .llm_services import requestsChatllm  # 假设 llm_services.py 实现了请求 LLM 的功能

app = Flask(__name__, static_folder='../../webui')
app.static_url_path = '/static'

# 配置日志
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "router.log"), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 身份验证装饰器
DEFAULT_API_KEY = "sk-jwuhnobgkwkppjpnnrjvzutqhkwcjshfbaiehsjaoetcbzkh"

def authenticate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key or api_key != DEFAULT_API_KEY:
            logging.warning(f"Authentication failed for API key: {api_key}")
            return jsonify({"error": "Unauthorized"}), 401
        
        return func(*args, **kwargs)
    return wrapper

@app.route('/', methods=['GET'])
def index():
    """提供 Web UI 界面"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/v1/chat/completions', methods=['POST'])
@authenticate  # 应用身份验证
def chat_completions():
    """
    接收符合 OpenAI API 格式的请求，调用 LLM 服务，并返回符合 OpenAI API 格式的响应。
    """
    try:
        # 1. 提取 OpenAI API 请求中的消息
        openai_data = request.get_json()
        messages = openai_data.get('messages', [])
        
        # 提取 conversation_id，如果没有则使用 "default"
        conversation_id = openai_data.get('conversation_id', 'default')
        
        # 2. 调用 LLM 服务
        user_message = messages[-1]['content'] if messages else ""  # 获取用户消息
        llm_response = requestsChatllm(user_message, conversation_id)
        
        if not llm_response["success"]:
            logging.error(f"LLM service failed: {llm_response.get('error', '未知错误')}")
            return jsonify({"error": llm_response.get("error", "未知错误")}), 500
        
        # 3. 将 LLM 服务的响应转换为 OpenAI API 格式
        openai_response = {
            "id": "chatcmpl-" + conversation_id,  # 生成一个唯一的ID
            "object": "chat.completion",
            "created": int(time.time()),
            "model": llm_response.get("model", "unknown"),
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": llm_response["content"]
                    },
                    "finish_reason": llm_response.get("finish_reason", "stop"),
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # 实际 token 使用量需要根据具体情况计算
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        # 4. 返回 OpenAI API 格式的响应
        logging.info(f"Request processed successfully for conversation_id: {conversation_id}")
        return jsonify(openai_response)
    
    except Exception as e:
        logging.exception("An error occurred during request processing")
        return jsonify({"error": str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    """提供静态文件服务"""
    try:
        logging.info(f"Attempting to serve static file: {path}")
        full_path = os.path.join(app.static_folder, path)
        logging.info(f"Full path: {full_path}")
        return send_from_directory(app.static_folder, path)
    except Exception as e:
        logging.error(f"Static file serving error: {e}")
        abort(404)

@app.errorhandler(404)
def not_found(error):
    """处理404错误，返回index.html"""
    logging.warning(f"404 error: {request.path}")
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)  # 假设在8000端口运行
