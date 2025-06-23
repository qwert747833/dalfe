#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from models.TimeCMA import Dual
import matplotlib.pyplot as plt
import seaborn as sns

# 全局字体设置（SVG矢量文字兼容）
plt.rcParams.update({
    'font.family': 'Gurmukhi MN',  # 确保使用矢量友好的字体
    'font.weight': 'bold',
    'font.size': 26,
    'axes.titlesize': 28,
    'axes.labelsize': 22,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'svg.fonttype': 'none'  # 确保文字在SVG中保存为可编辑文本（非路径）
})

def visualize_attention(model, device, output_dir="./attention_vis"):
    """生成SVG格式的可视化"""
    os.makedirs(output_dir, exist_ok=True)
    x, x_mark, embeddings = generate_valid_input(device)
    
    with torch.no_grad():
        raw_output = model(x, x_mark, embeddings, return_attn=True)
        output, (ts_weights, emb_weights) = raw_output
        ts_weights = ts_weights.squeeze(0).cpu().numpy()
        emb_weights = emb_weights.squeeze(0).cpu().numpy()
        
        # 创建图形
        plt.figure(figsize=(18, 8))
        
        # 时序->嵌入注意力
        plt.subplot(1, 2, 1)
        heatmap = sns.heatmap(
            ts_weights,
            cmap="YlGnBu",
            # annot=True,
            fmt=".2f",
            cbar_kws={"ticks": [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,0.19,0.20]} ,
            annot_kws={"size": 12}
        )
        heatmap.set_yticklabels(heatmap.get_yticklabels(), 
                       rotation=0,  # 0度旋转（正着显示）
                       va='center')  # 垂直居中
        plt.text(0.5, -0.15, "(a) Time → Embedding Attention.", 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=30, weight='bold')

        
        # 嵌入->时序注意力
        plt.subplot(1, 2, 2)
        heatmap = sns.heatmap(
            emb_weights,
            cmap="YlGnBu",
            # annot=True,
            fmt=".2f",
            cbar_kws={"ticks": [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,0.19,0.20]} ,
            annot_kws={"size": 12}
        )
        heatmap.set_yticklabels(heatmap.get_yticklabels(), 
                       rotation=0,  # 0度旋转（正着显示）
                       va='center')  # 垂直居中
        # 在子图下方添加(b)标签
        plt.text(0.5, -0.15, "(b) Embedding → Time Attention.", 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=30, weight='bold')        
        
        plt.tight_layout()
        
        # 明确保存为SVG
        svg_path = os.path.join(output_dir, "attention.svg")
        plt.savefig(svg_path, format="svg", bbox_inches='tight')
        plt.close()
        print(f"矢量图已保存至: {svg_path}")

def load_model(model_path):
    """安全加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Dual(
        device=device,
        channel=64,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.6,
        d_llm=768,
        e_layer=1,
        d_layer=2,
        head=8
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def generate_valid_input(device):
    """生成符合模型要求的输入"""
    batch_size = 1
    seq_len = 96
    num_nodes = 7
    d_llm = 768
    
    dummy_x = torch.randn(batch_size, seq_len, num_nodes).to(device)
    dummy_x_mark = torch.randn(batch_size, seq_len, 4).to(device)
    dummy_embeddings = torch.randn(batch_size, num_nodes, d_llm).to(device)
    dummy_embeddings = dummy_embeddings.transpose(1, 2)  # [B, E, N]
    return dummy_x, dummy_x_mark, dummy_embeddings

if __name__ == "__main__":
    MODEL_PATH = "logs/2025-06-22-17:09:27-/ETTh1/96_64_1_2_0.0001_0.6_2024/best_model.pth"
    
    print("Starting visualization...")
    try:
        model, device = load_model(MODEL_PATH)
        visualize_attention(model, device)
        print("Successfully completed!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")