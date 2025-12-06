@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置标题
title Flux Kontext LoRA 训练

echo ======================================================
echo              Flux Kontext LoRA 训练启动器
echo ======================================================
echo.

:: 激活虚拟环境
echo 正在激活虚拟环境...
call .\venv\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo 错误：虚拟环境激活失败！
    echo 请确保当前目录下存在venv文件夹。
    goto :error
)

echo 虚拟环境已激活。
echo.

:: 设置默认配置文件
set CONFIG_FILE=train_lora_flux_kontext_24gb.yaml

:: 检查是否有自定义配置文件参数
if not "%~1"=="" (
    set CONFIG_FILE=%~1
)

:: 检查配置文件是否存在
if not exist %CONFIG_FILE% (
    echo 错误：配置文件 %CONFIG_FILE% 不存在！
    goto :error
)

echo 使用配置文件: %CONFIG_FILE%
echo.

:: 启动训练
echo 开始训练...
echo.
python run_yaml_training.py --config_path %CONFIG_FILE%

if %errorlevel% neq 0 (
    echo.
    echo 训练过程中出现错误！
    goto :error
)

echo.
echo 训练完成！
goto :end

:error
echo.
echo 程序执行出错，请检查以上信息。
pause
exit /b 1

:end
echo.
echo 按任意键退出...
pause >nul
exit /b 0 