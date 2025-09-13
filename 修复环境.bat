@echo off
setlocal enabledelayedexpansion
set "VENV_DIR=.venv"
set "REQ_FILE=requirements.txt"

echo [1/3] 删除旧虚拟环境（如有）...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
)

echo [2/3] 创建虚拟环境...
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo 创建失败，请确认 python 已加入 PATH
    pause & exit /b 1
)

echo [3/3] 安装依赖...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip
if exist "%REQ_FILE%" (
    pip install -r "%REQ_FILE%"
) else (
    echo 警告：未找到 %REQ_FILE%
)

echo.
echo 完成！已激活虚拟环境，可直接使用。
echo 退出虚拟环境请运行  deactivate
cmd /k