import requests

url = "http://127.0.0.1:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "sk-jwuhnobgkwkppjpnnrjvzutqhkwcjshfbaiehsjaoetcbzkh"
}
data = {
    "messages": [
        {"role": "user", "content": "你好，Vix"}
    ],
    "conversation_id": "test001"
}

response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.json())
