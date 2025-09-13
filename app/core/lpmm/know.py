# app/core/lpmm/know.py
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
    """简化版配置加载（供知识库独立使用）"""
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
    """获取嵌入模型提供商配置"""
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
    """调用嵌入模型获取向量"""
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
    """计算余弦相似度"""
    if vec1 is None or vec2 is None:
        return 0.0
    a = np.array(vec1)
    b = np.array(vec2)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def generate_file_hash(filepath: str) -> str:
    """生成文件内容的 MD5 哈希，用于版本检测"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# ==================== 知识库核心类 ====================
class KnowledgeBase:
    def __init__(self, text_dir: str = "lpmm-ie", json_dir: str = None):
        self.text_dir = text_dir
        self.json_dir = json_dir or os.path.join(text_dir, "json")
        self.data : List[Dict[str, Any]] = []  # ✅ 正确定义 data 属性
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        self.load_on_start = get_llm_config_value("lpmm.settings", "load_on_start", True)
        self.show_progress = get_llm_config_value("lpmm.settings", "show_progress", True)
        
        if self.load_on_start:
            self.reload()
        else:
            print("[知识库] 跳过启动加载（load_on_start=false）")

    def split_sentences(self, text: str) -> List[str]:
        """智能分句：优先按换行，其次按标点，保留等号结构"""
        sentences = []
        
        # 如果包含等号且没有标点，整句保留（适配定义式文本）
        if "=" in text and not any(c in text for c in "。！？；"):
            sentences.append(text.strip())
            return sentences
        
        # 按换行符切分
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 如果是定义式（含等号），直接保留
            if "=" in line and len(line.split('=', 1)) == 2:
                sentences.append(line)
            else:
                # 按标点切分
                parts = re.split(r'[。！？；]', line)
                for part in parts:
                    part = part.strip()
                    if len(part) >= 3:  # 放宽到3个字符
                        sentences.append(part)
        
        return sentences

    def load_from_files(self) -> List[Dict[str, Any]]:
        """从文本文件加载知识 - 优化版"""
        knowledge = []
        txt_files = [f for f in os.listdir(self.text_dir) if f.endswith(".txt")]
        total_files = len(txt_files)
        if total_files == 0:
            return knowledge

        use_tqdm = self.show_progress
        if use_tqdm:
            try:
                from tqdm import tqdm
            except ImportError:
                use_tqdm = False
                print("[知识库] 未安装 tqdm，进度条已禁用。运行: pip install tqdm")

        file_iter = txt_files
        if use_tqdm:
            file_iter = tqdm(txt_files, desc="📄 处理知识文件", unit="文件")

        for filename in file_iter:
            path = os.path.join(self.text_dir, filename)
            try:
                # 生成文件哈希
                file_hash = generate_file_hash(path)
                cache_found = False
                cache_file = None

                # 检查是否有对应缓存文件
                for cache_filename in os.listdir(self.json_dir):
                    if cache_filename.endswith(".json") and filename.replace(".txt", "") in cache_filename:
                        cache_path = os.path.join(self.json_dir, cache_filename)
                        try:
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                cache_data = json.load(f)
                                if isinstance(cache_data, list) and len(cache_data) > 0:
                                    # 检查是否包含文件哈希（版本校验）
                                    if cache_data[0].get("source_file_hash") == file_hash:
                                        # ✅ 缓存命中！直接加载
                                        knowledge.extend(cache_data)
                                        cache_found = True
                                        if use_tqdm:
                                            file_iter.set_postfix({"状态": "缓存命中"})
                                        break
                        except:
                            continue

                if cache_found:
                    continue

                # ❌ 缓存未命中，重新生成
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                file_knowledge = []
                line_iter = lines
                if use_tqdm and len(lines) > 50:
                    from tqdm import tqdm
                    line_iter = tqdm(lines, desc=f"   {filename}", unit="行", leave=False)

                for i, line in enumerate(line_iter):
                    line = line.strip()
                    if not line:
                        continue
                    if "=" in line:
                        clean_text = line
                    else:
                        sentences = self.split_sentences(line)
                        if not sentences:
                            sentences = [line]
                        clean_text = sentences[0] if sentences else line
                    if len(clean_text.strip()) < 3:
                        continue
                    
                    # 获取嵌入
                    embedding = get_embedding(clean_text)
                    item = {
                        "key": f"{filename}_{i}",
                        "value": clean_text.strip(),
                        "embedding": embedding,
                        "tags": ["imported", "txt"],
                        "source": filename,
                        "source_file_hash": file_hash,  # ✅ 记录文件哈希用于缓存校验
                        "created_at": datetime.datetime.now().isoformat(),
                        "usage_count": 0
                    }
                    file_knowledge.append(item)
                    knowledge.append(item)

                # 保存新缓存
                if file_knowledge:
                    self.save_to_json(file_knowledge, "imported", filename)

            except Exception as e:
                print(f"[知识库] 加载文件失败 {filename}: {e}")
        return knowledge

    def save_to_json(self, knowledge_list: List[Dict], source_type: str, source_name: str = "") -> str:
        """保存知识到JSON文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{source_type}_{source_name.replace('.txt', '').replace('/', '_')}.json"
        filepath = os.path.join(self.json_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(knowledge_list, f, ensure_ascii=False, indent=2)
            return filepath
        except Exception as e:
            print(f"保存知识文件失败: {e}")
            return ""

    def load_all_json(self) -> List[Dict[str, Any]]:
        """从JSON目录加载所有知识"""
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
            file_iter = tqdm(json_files, desc="🧠 加载知识缓存", unit="文件")
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
                print(f"加载知识文件失败 {filename}: {e}")
        return knowledge

    def add_knowledge(self, key: str, value: str, tags: List[str] = None) -> Dict[str, Any]:
        """添加新知识"""
        if tags is None:
            tags = []
        embedding = get_embedding(value)
        new_item = {
            "key": key.strip(),
            "value": value.strip(),
            "embedding": embedding,
            "tags": [t.strip() for t in tags] + ["learned"],
            "source": "auto_learn",
            "created_at": datetime.datetime.now().isoformat(),
            "usage_count": 1
        }
        self.save_to_json([new_item], "learned", key.replace(" ", "_"))
        self.data.append(new_item)
        return new_item

    def reload(self):
        """重新加载所有知识"""
        print("[知识库] 开始加载知识...")
        start_time = datetime.datetime.now()
        self.data = []
        
        # 先加载独立学习的知识（JSON）
        json_knowledge = self.load_all_json()
        self.data.extend(json_knowledge)
        
        # 再加载TXT文件（带缓存检测）
        txt_knowledge = self.load_from_files()
        self.data.extend(txt_knowledge)
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[知识库] ✅ 加载完成！共 {len(self.data)} 条知识，耗时 {duration:.2f} 秒")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """语义搜索 + 关键词回退"""
        if not self.data or not query.strip():
            return []
        query_embedding = get_embedding(query)
        semantic_matches = []
        if query_embedding is not None:
            for item in self.data:  # ✅ 修复：self.data
                if item.get("embedding") is not None:
                    score = cosine_similarity(query_embedding, item["embedding"])
                    if score > 0.3:
                        item_copy = item.copy()
                        item_copy["score"] = score
                        item_copy["match_type"] = "semantic"
                        semantic_matches.append(item_copy)
            semantic_matches.sort(key=lambda x: x["score"], reverse=True)
            if len(semantic_matches) >= top_k:
                return semantic_matches[:top_k]
        keyword_matches = []
        query_lower = query.lower().strip()
        for item in self.data:  # ✅ 修复：self.data
            score = 0
            item_value_lower = item["value"].lower()
            item_key_lower = item["key"].lower()
            if query_lower in item_key_lower:
                score += 2
            if query_lower in item_value_lower:
                score += 1
            if query_lower == item_value_lower:
                score += 3
            if score > 0:
                item_copy = item.copy()
                item_copy["score"] = score
                item_copy["match_type"] = "keyword"
                keyword_matches.append(item_copy)
        keyword_matches.sort(key=lambda x: x["score"], reverse=True)
        all_matches = semantic_matches + [m for m in keyword_matches if m not in semantic_matches]
        
        # 尝试添加表达方式推荐
        try:
            from .expression_knowledge import search_similar_expressions
            expr_matches = search_similar_expressions(query, top_k=2)
            for expr in expr_matches:
                expr_item = {
                    "key": "expression_recommendation",
                    "value": expr["value"],
                    "score": expr["similarity"] * 0.8,  # 降低权重
                    "match_type": "expression",
                    "source": "expression_knowledge"
                }
                all_matches.append(expr_item)
        except Exception:
            pass
        
        all_matches.sort(key=lambda x: x["score"], reverse=True)
        return all_matches[:top_k]

    def get_all(self) -> List[Dict[str, Any]]:
        return self.data.copy()

    def manual_load(self):
        self.reload()

# ==================== 全局单例 ====================
_knowledge_base_instance = None

def get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base_instance
    if _knowledge_base_instance is None:
        _knowledge_base_instance = KnowledgeBase()
    return _knowledge_base_instance

# ==================== 便捷函数 ====================
def search_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    kb = get_knowledge_base()
    return kb.search(query, top_k)

def add_knowledge(key: str, value: str, tags: List[str] = None) -> Dict[str, Any]:
    kb = get_knowledge_base()
    return kb.add_knowledge(key, value, tags)

def reload_knowledge():
    kb = get_knowledge_base()
    kb.reload()

def manual_load_knowledge():
    kb = get_knowledge_base()
    kb.manual_load()

# 初始化
if get_llm_config_value("lpmm.settings", "load_on_start", True):
    reload_knowledge()
else:
    print("[知识库] 启动时未加载知识库（load_on_start=false）")
    print("[提示] 如需手动加载，请调用 manual_load_knowledge()")