#!/usr/bin/env python
# coding=utf-8
"""
YAML配置文件训练启动器
用于解析YAML配置文件并启动train_flux_lora_ui_kontext.py训练脚本
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="使用YAML配置文件执行Flux Kontext训练")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="YAML配置文件的路径",
    )
    return parser.parse_args()

def extract_training_params(config):
    """
    通过递归遍历配置字典来提取所有可传递的训练参数。
    """
    params = {}

    # 参数名映射：将YAML中的参数名映射到训练脚本所需的参数名
    param_mapping = {
        "caption_dropout_rate": "caption_dropout",
    }

    # 忽略的参数：这些参数不会被传递给训练脚本
    ignore_keys = {
        "job", "type", "name", "meta", "training_folder", "device",
        "folder_path", "caption_ext", "cache_latents_to_disk",
        "arch", "quantize", "low_vram", "lora_alpha", "enable_bucketing"
    }

    def walk_and_extract(cfg_node):
        """递归函数，用于遍历配置并提取参数"""
        if isinstance(cfg_node, dict):
            for key, value in cfg_node.items():
                if key in ignore_keys:
                    continue

                if key == "resolution" and isinstance(value, list):
                    # 特殊处理resolution，列表只取第一个值
                    if value:
                        params["resolution"] = str(value[0])
                elif key == "target_modules" and isinstance(value, list):
                    # 将target_modules列表转换为lora_layers字符串
                    params["lora_layers"] = ",".join(value)
                elif isinstance(value, (dict, list)):
                    # 如果值是字典或列表，则继续递归
                    walk_and_extract(value)
                else:
                    # 基础类型的值，直接添加
                    script_key = param_mapping.get(key, key)
                    params[script_key] = value
        elif isinstance(cfg_node, list):
            # 如果是列表，则遍历其所有项
            for item in cfg_node:
                walk_and_extract(item)

    # 从 'config' 根节点开始遍历
    if "config" in config:
        walk_and_extract(config["config"])

    # 检查是否启用bucketing
    if config.get("config", {}).get("enable_bucketing"):
        params["enable_bucketing"] = True

    # 检查必要参数是否存在
    essential_params = [
        "pretrained_model_name_or_path", "train_data_dir", "output_dir", "save_name"
    ]
    missing_params = [p for p in essential_params if p not in params]
    if missing_params:
        print(f"警告: 配置文件中缺少以下必要参数: {', '.join(missing_params)}")
    
    return params

def main():
    args = parse_args()
    config_path = Path(args.config_path)

    if not config_path.is_file():
        print(f"错误: 配置文件 {config_path} 不存在")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"解析YAML文件时出错: {e}")
        sys.exit(1)

    params = extract_training_params(config)

    print("从YAML配置文件中提取的参数:")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")

    # 根据是否启用bucketing选择正确的训练脚本
    use_bucketing = params.get("enable_bucketing", False)
    script_to_run = "train_flux_lora_ui_kontext_bucketing.py" if use_bucketing else "train_flux_lora_ui_kontext.py"
    
    # 构建命令行参数
    cmd = [sys.executable, script_to_run]
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    # 将配置文件路径也作为一个参数传递，便于追溯
    cmd.append("--config_path")
    cmd.append(str(config_path))

    print("\n即将执行命令:")
    # 为了可读性，将命令中的长路径用引号括起来
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))

    # 询问用户是否继续
    try:
        confirmation = input("\n是否继续执行训练? (y/n): ")
        if confirmation.lower() not in ['y', 'yes']:
            print("训练已取消")
            return
    except KeyboardInterrupt:
        print("\n操作已取消")
        return

    # 执行训练脚本
    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"训练过程中出错 (返回码: {e.returncode}):\n{e.stderr}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n训练已被用户中断")

if __name__ == "__main__":
    main() 