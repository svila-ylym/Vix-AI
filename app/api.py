from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from . import models, auth
from .models import SessionLocal, Conversation
from typing import List

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: dict, db: Session = Depends(get_db)):
    username = user.get("username")
    password = user.get("password")
    email = user.get("email")

    if not username or not password or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username, password and email are required"
        )
    
    existing_user = auth.get_user(db, username=username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    hashed_password = auth.get_password_hash(password)
    new_user = models.User(username=username, hashed_password=hashed_password, email=email)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = auth.get_user(db, form_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password"
        )
    if not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password"
        )
    access_token_expires = datetime.timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# 修改 /v1/chat/completions 接口
@app.post("/v1/chat/completions")
async def chat_completions(request_data: dict, db: Session = Depends(get_db)):
    messages = request_data.get("messages")
    conversation_id = request_data.get("conversation_id")
    user_id = 1  # 假设当前用户ID为1，需要根据实际情况获取

    # 保存用户消息到对话历史
    for message in messages:
        db_message = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            role=message["role"],
            content=message["content"]
        )
        db.add(db_message)
    db.commit()

    # TODO: 调用LLM模型获取回复 (这里需要你原来的LLM调用逻辑)
    response_content = "这是AI回复"  # 替换为实际的AI回复

    # 保存AI回复到对话历史
    db_response = Conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        role="assistant",
        content=response_content
    )
    db.add(db_response)
    db.commit()

    return {"response": response_content}

@app.get("/conversations/{conversation_id}", response_model=List[dict])
async def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conversations = db.query(Conversation).filter(Conversation.conversation_id == conversation_id).all()
    return [{"role": c.role, "content": c.content} for c in conversations]
