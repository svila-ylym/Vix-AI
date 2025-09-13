# VixAI

## 项目简介
VixAI 是一个基于多模型融合的智能对话与日程助手，支持多种大模型（如 Qwen、InternLM、GLM 等），具备情感模拟、角色扮演、日程生成、审核等能力。项目采用 Flask 提供 API 服务，并支持本地及云端模型调用。

## 主要功能
- 智能对话（角色扮演、情感模拟）
- 日程自动生成与管理
- 消息审核与内容安全
- 支持多模型和多提供商（OpenAI、Anthropic、Azure、Siliconflow、本地）
- OpenAI API兼容接口
- 对话历史记录与分段输出

## 快速启动

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量
   - 编辑 `.env` 文件，填写各模型API Key及角色参数。
   - 主要参数见 `.env` 示例。

3. 启动服务
   ```bash
   python main.py
   ```
   默认监听 `8000` 端口。

4.  访问 Web UI
    - 打开浏览器，访问 `http://localhost:8000/` 即可使用 Web UI 界面与 AI 聊天。

## 配置说明

- `.env`：API密钥、角色设定、系统参数等。
- `config/llm.toml`：各模型参数、提供商选择。
- `daily_schedule.json`：自动生成和存储日程。
- `conversation_log.txt`：对话历史记录。

## API接口示例

### 聊天接口

- 路径：`POST /v1/chat/completions`
- 请求格式（兼容OpenAI）：
  ```json
  {
    "messages": [
      {"role": "user", "content": "你好，AI！"}
    ],
    "conversation_id": "test001"
  }
  ```
- Header需包含 `X-API-Key`

- 响应格式：
  ```json
  {
    "id": "chatcmpl-test001",
    "object": "chat.completion",
    "created": 1694080000,
    "model": "Qwen/Qwen3-14B",
    "choices": [
      {
        "message": {
          "role": "assistant",
          "content": "你好，很高兴见到你！"
        },
        "finish_reason": "stop",
        "index": 0
      }
    ],
    "usage": {
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0
    }
  }
  ```

## 目录结构

```
VT/
├── app/
│   └── services/
├── config/
├── log/
├── daily_schedule.json
├── conversation_log.txt
├── main.py
├── .env
├── README.md
```

## 其他说明

- 支持多模型切换，具体参数见 `config/llm.toml`。
- 日程与情感模拟可自定义。
- 日志自动保存于 `log/` 目录。

## 联系方式

如有问题或建议，请联系项目维护者。
