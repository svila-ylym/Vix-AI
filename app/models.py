from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# 数据库配置 (使用 SQLite 作为示例)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vt.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 用户模型
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True)
    create_date = Column(DateTime, default=datetime.datetime.utcnow)

# 对话历史模型
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, index=True)
    user_id = Column(Integer)  # 外键关联到用户
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def create_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_db()
