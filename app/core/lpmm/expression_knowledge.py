# app/core/lpmm/expression_knowledge.py
import os
import json
import datetime
import re
import numpy as np
import requests
import hashlib
from typing import List, Dict, Any, Optional

# ==================== 配置加载工具 ====================
def get_llm_config(config_key: str, default_config: dict = None) -> dict:
    if default_config is None:
        default_config = {}
    try:
        import toml
        config = toml.load("config/llm.toml")
        if config_key in config:
            return config[config_key]
        elif "." in config_key:
            main_key, sub_key = config_key.split(".", 1)
            if main_key in config and isinstance(config[main_key], dict) and sub_key in config[main_key]:
                return config[main_key][sub_key]
    except Exception:
        pass
    return default_config

def get_llm_config_value(config_key: str, value_key: str, default_value=None):
    config = get_llm_config(config_key, {})
    return config.get(value_key, default_value)

# ==================== 嵌入模型工具 ====================
def get_embed_provider_config(provider_name: str) -> dict:
    from dotenv import load_dotenv
    load_dotenv(".env", override=True)
    
    embed_configs = {
        "openai": {
            "url": os.getenv('openai_embed_url', 'https://api.openai.com/v1/embeddings'),
            "key": os.getenv('openai_api_key'),
            "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            "model": get_llm_config_value("lpmm.embedmodel", "model", "text-embedding-ada-002")
        },
        "siliconflow": {
            "url": os.getenv('siliconflow_embed_url', 'https://api.siliconflow.cn/v1/embeddings'),
            "key": os.getenv('siliconflow_api_key'),
            "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            "model": get_llm_config_value("lpmm.embedmodel", "model", "BAAI/bge-m3")
        },
        "local": {
            "url": os.getenv('local_embed_url'),
            "key": os.getenv('local_api_key'),
            "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            "model": get_llm_config_value("lpmm.embedmodel", "model", "text-embedding")
        }
    }
    return embed_configs.get(provider_name, embed_configs["siliconflow"])

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        embed_config = get_llm_config("lpmm.embedmodel", {})
        provider = embed_config.get("provider", "siliconflow")
        config = get_embed_provider_config(provider)
        
        url = config["url"]
        api_key = config["key"]
        model = config["model"]
        headers = config["headers"](api_key)
        
        payload = {
            "model": model,
            "input": text
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            if "embedding" in data["data"][0]:
                return data["data"][0]["embedding"]
            elif "embeddings" in data["data"][0]:
                return data["data"][0]["embeddings"]
        return None
    except Exception:
        return None

def cosine_similarity(vec1, vec2) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    a = np.array(vec1)
    b = np.array(vec2)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def generate_file_hash(filepath: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# ==================== 表达方式选择器 ====================
def should_learn_expression(text: str) -> bool:
    """
    判断是否应该学习这个表达方式
    规则：
    1. 长度在 5-100 字符之间
    2. 包含情感词或网络流行语
    3. 不是纯标点或数字
    4. 不包含敏感词
    """
    if len(text) < 5 or len(text) > 100:
        return False
    
    # 检查是否包含情感词或流行语
    emotional_words = [
        "哈哈", "笑死", "绝了", "太可了", "awsl", "蚌埠住了",
        "破防", "泪目", "感动", "暖心", "扎心", "真实", "典中典"
    ]
    has_emotional = any(word in text for word in emotional_words)
    
    # 检查是否包含网络流行语模式
    internet_patterns = [
        r'[a-zA-Z]{2,}了$',  # xx了
        r'^[a-zA-Z]+$',      # 纯字母缩写
        r'[0-9]+[a-zA-Z]+',  # 数字+字母
        r'=.*=',             # 定义式
    ]
    has_pattern = any(re.search(pattern, text) for pattern in internet_patterns)
    
    # 检查是否纯标点/数字
    if re.match(r'^[0-9\s\W]+$', text):
        return False
    
    # 检查敏感词
    sensitive_words = ["死", "杀", "滚", "骂", "操", "fuck", "shit"]
    if any(word in text.lower() for word in sensitive_words):
        return False
    
    return has_emotional or has_pattern

# ==================== 表达方式知识库核心类 ====================
class ExpressionKnowledgeBase:
    def __init__(self, expr_dir: str = "lpmm-ie/expressions", json_dir: str = None):
        self.expr_dir = expr_dir
        self.json_dir = json_dir or os.path.join("lpmm-ie", "json", "expressions")
        self.data: List[Dict[str, Any]] = []  # ✅ 修复：正确定义 data 属性
        
        # 创建目录
        os.makedirs(self.expr_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        # 从配置读取是否加载
        self.load_on_start = get_llm_config_value("lpmm.settings", "load_expressions_on_start", True)
        self.show_progress = get_llm_config_value("lpmm.settings", "show_progress", True)
        
        # 加载表达方式
        if self.load_on_start:
            self.reload()
        else:
            print("[表达方式库] 跳过启动加载（load_expressions_on_start=false）")

    def extract_expressions_from_text(self, text: str) -> List[str]:
        """从文本中提取可能的表达方式"""
        expressions = []
        
        # 按行分割
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 如果是定义式，提取等号后部分
            if "=" in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    expr = parts[1].strip()
                    if expr and should_learn_expression(expr):
                        expressions.append(expr)
                    if parts[0].strip() and should_learn_expression(parts[0].strip()):
                        expressions.append(parts[0].strip())
            else:
                # 按句子分割
                sentences = re.split(r'[。！？；]', line)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and should_learn_expression(sentence):
                        expressions.append(sentence)
        
        return expressions

    def load_from_files(self) -> List[Dict[str, Any]]:
        """从表达方式文件加载"""
        knowledge = []
        expr_files = [f for f in os.listdir(self.expr_dir) if f.endswith(".txt")]
        total_files = len(expr_files)
        if total_files == 0:
            return knowledge

        use_tqdm = self.show_progress
        if use_tqdm:
            try:
                from tqdm import tqdm
            except ImportError:
                use_tqdm = False
                print("[表达方式库] 未安装 tqdm，进度条已禁用")

        file_iter = expr_files
        if use_tqdm:
            file_iter = tqdm(expr_files, desc="💬 加载表达方式", unit="文件")

        for filename in file_iter:
            path = os.path.join(self.expr_dir, filename)
            try:
                # 生成文件哈希
                file_hash = generate_file_hash(path)
                cache_found = False
                cache_file = None

                # 检查缓存
                for cache_filename in os.listdir(self.json_dir):
                    if cache_filename.endswith(".json") and filename.replace(".txt", "") in cache_filename:
                        cache_path = os.path.join(self.json_dir, cache_filename)
                        try:
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                cache_data = json.load(f)
                                if isinstance(cache_data, list) and len(cache_data) > 0:
                                    if cache_data[0].get("source_file_hash") == file_hash:
                                        knowledge.extend(cache_data)
                                        cache_found = True
                                        if use_tqdm:
                                            file_iter.set_postfix({"状态": "缓存命中"})
                                        break
                        except:
                            continue

                if cache_found:
                    continue

                # 重新生成
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                expressions = self.extract_expressions_from_text(content)
                file_knowledge = []
                
                expr_iter = expressions
                if use_tqdm and len(expressions) > 20:
                    from tqdm import tqdm
                    expr_iter = tqdm(expressions, desc=f"   {filename}", unit="条", leave=False)

                for i, expr in enumerate(expr_iter):
                    if not should_learn_expression(expr):  # 双重检查
                        continue
                        
                    embedding = get_embedding(expr)
                    item = {
                        "key": f"expr_{filename}_{i}",
                        "value": expr,
                        "embedding": embedding,
                        "tags": ["expression", "imported"],
                        "source": filename,
                        "source_file_hash": file_hash,
                        "created_at": datetime.datetime.now().isoformat(),
                        "usage_count": 0
                    }
                    file_knowledge.append(item)
                    knowledge.append(item)

                if file_knowledge:
                    self.save_to_json(file_knowledge, "expression", filename)

            except Exception as e:
                print(f"[表达方式库] 加载文件失败 {filename}: {e}")
        return knowledge

    def save_to_json(self, knowledge_list: List[Dict], source_type: str, source_name: str = "") -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{source_type}_{source_name.replace('.txt', '').replace('/', '_')}.json"
        filepath = os.path.join(self.json_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(knowledge_list, f, ensure_ascii=False, indent=2)
            return filepath
        except Exception as e:
            print(f"保存表达方式文件失败: {e}")
            return ""

    def load_all_json(self) -> List[Dict[str, Any]]:
        """加载所有表达方式缓存"""
        knowledge = []
        if not os.path.exists(self.json_dir):
            return knowledge
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith(".json")]
        use_tqdm = self.show_progress
        if use_tqdm:
            try:
                from tqdm import tqdm
            except ImportError:
                use_tqdm = False
        file_iter = json_files
        if use_tqdm:
            file_iter = tqdm(json_files, desc="🧠 加载表达缓存", unit="文件")
        for filename in file_iter:
            path = os.path.join(self.json_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                    if isinstance(items, list):
                        knowledge.extend(items)
                    else:
                        knowledge.append(items)
            except Exception as e:
                print(f"加载表达方式文件失败 {filename}: {e}")
        return knowledge

    def add_expression(self, expression: str, source: str = "auto_learn") -> Optional[Dict[str, Any]]:
        """添加新表达方式"""
        if not should_learn_expression(expression):
            return None
            
        embedding = get_embedding(expression)
        new_item = {
            "key": f"expr_learned_{hash(expression) % 1000000}",
            "value": expression,
            "embedding": embedding,
            "tags": ["expression", "learned"],
            "source": source,
            "created_at": datetime.datetime.now().isoformat(),
            "usage_count": 1
        }
        self.save_to_json([new_item], "learned_expression", source)
        self.data.append(new_item)
        return new_item

    def reload(self):
        print("[表达方式库] 开始加载表达方式...")
        start_time = datetime.datetime.now()
        self.data = []
        json_knowledge = self.load_all_json()
        self.data.extend(json_knowledge)
        expr_knowledge = self.load_from_files()
        self.data.extend(expr_knowledge)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[表达方式库] ✅ 加载完成！共 {len(self.data)} 条表达方式，耗时 {duration:.2f} 秒")

    def search_similar(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索相似表达方式"""
        if not self.data or not text.strip():
            return []
            
        query_embedding = get_embedding(text)
        if query_embedding is None:
            return []
            
        matches = []
        for item in self.data:  # ✅ 修复：self.data
            if item.get("embedding") is not None:
                score = cosine_similarity(query_embedding, item["embedding"])
                if score > 0.4:  # 较高阈值，确保相似度
                    item_copy = item.copy()
                    item_copy["similarity"] = score
                    matches.append(item_copy)
        
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_k]

    def get_random_expressions(self, count: int = 3) -> List[str]:
        """随机获取表达方式"""
        if not self.data:  # ✅ 修复：self.data
            return []
        import random
        selected = random.sample(self.data, min(count, len(self.data)))
        return [item["value"] for item in selected]

    def manual_load(self):
        self.reload()

# ==================== 全局单例 ====================
_expression_kb_instance = None

def get_expression_knowledge_base() -> ExpressionKnowledgeBase:
    global _expression_kb_instance
    if _expression_kb_instance is None:
        _expression_kb_instance = ExpressionKnowledgeBase()
    return _expression_kb_instance

# ==================== 便捷函数 ====================
def search_similar_expressions(text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    kb = get_expression_knowledge_base()
    return kb.search_similar(text, top_k)

def add_expression(expression: str, source: str = "auto_learn") -> Optional[Dict[str, Any]]:
    kb = get_expression_knowledge_base()
    return kb.add_expression(expression, source)

def get_random_expressions(count: int = 3) -> List[str]:
    kb = get_expression_knowledge_base()
    return kb.get_random_expressions(count)

def reload_expressions():
    kb = get_expression_knowledge_base()
    kb.reload()

def manual_load_expressions():
    kb = get_expression_knowledge_base()
    kb.manual_load()

# 初始化
if get_llm_config_value("lpmm.settings", "load_expressions_on_start", True):
    reload_expressions()
else:
    print("[表达方式库] 启动时未加载表达方式（load_expressions_on_start=false）")