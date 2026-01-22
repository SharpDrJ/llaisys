#!/usr/bin/env python3
"""
正确的128 token完整对比测试
使用正确的调用方式对比LLAISYS和HuggingFace
"""

import sys
sys.path.insert(0, 'python')

import llaisys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_path = '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'

print("="*80)
print("128 Token完整对比测试（正确的测试方式）")
print("="*80)
print(f"模型路径: {model_path}")
print()

# 初始tokens（完整序列）
tokens = [151646, 151646, 151644]
max_new_tokens = 128

# 加载HuggingFace模型和tokenizer
print("正在加载HuggingFace模型...")
hf_start = time.time()
hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map='cpu'
)
hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
hf_load_time = time.time() - hf_start
print(f"✓ HuggingFace模型加载完成 ({hf_load_time:.2f}秒)")

# 加载LLAISYS模型
print("\n正在加载LLAISYS模型...")
llaisys_start = time.time()
llaisys_model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
llaisys_load_time = time.time() - llaisys_start
print(f"✓ LLAISYS模型加载完成 ({llaisys_load_time:.2f}秒)")

print(f"\n初始tokens: {tokens}")
print(f"目标: 生成{max_new_tokens}个新token")
print()

# 测试方式：一次调用生成所有token
print("="*80)
print(f"生成{max_new_tokens}个token...")
print("="*80)
print()

# HuggingFace推理
print("运行HuggingFace推理...")
inputs = torch.tensor([tokens]).to(torch.long)
hf_start = time.time()
with torch.no_grad():
    outputs = hf_model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        do_sample=False
    )
hf_tokens = outputs[0].tolist()
hf_time = time.time() - hf_start
print(f"✓ HuggingFace完成 ({hf_time:.2f}秒)")
print(f"  生成了 {len(hf_tokens) - len(tokens)} 个新token")

# LLAISYS推理
print("\n运行LLAISYS推理...")
llaisys_start = time.time()
llaisys_generated = llaisys_model.generate(tokens, max_new_tokens=max_new_tokens)
llaisys_tokens = tokens + llaisys_generated
llaisys_time = time.time() - llaisys_start
print(f"✓ LLAISYS完成 ({llaisys_time:.2f}秒)")
print(f"  生成了 {len(llaisys_generated)} 个新token")

# 对比结果
print()
print("="*80)
print("测试结果")
print("="*80)

# 统计
total_new = len(hf_tokens) - len(tokens)
matches = sum(1 for hf, ll in zip(hf_tokens, llaisys_tokens) if hf == ll)
mismatches = total_new - (matches - len(tokens))  # 减去初始tokens
match_rate = (matches / len(hf_tokens)) * 100

print(f"\n总tokens: {len(hf_tokens)}")
print(f"匹配数量: {matches}")
print(f"不匹配数量: {len(hf_tokens) - matches}")
print(f"匹配率: {match_rate:.2f}%")

# 性能
print(f"\n性能统计:")
print(f"  HuggingFace: {hf_time:.2f}秒")
print(f"  LLAISYS:     {llaisys_time:.2f}秒")
print(f"  加速比:      {hf_time/llaisys_time:.2f}x")

# 不匹配详情
if hf_tokens != llaisys_tokens:
    print(f"\n不匹配详情:")
    print(f"{'位置':>6} {'HuggingFace':>12} {'LLAISYS':>12}")
    print("-" * 40)
    for i, (hf, ll) in enumerate(zip(hf_tokens, llaisys_tokens)):
        if hf != ll:
            print(f"{i:>6} {hf:>12} {ll:>12}")
else:
    print("\n✓✓✓ 所有token完全匹配！ ✓✓✓")

# Token序列
print(f"\nToken序列:")
print(f"  HuggingFace (前10个): {hf_tokens[:10]}")
print(f"  HuggingFace (后10个): {hf_tokens[-10:]}")
print(f"  LLAISYS (前10个):      {llaisys_tokens[:10]}")
print(f"  LLAISYS (后10个):      {llaisys_tokens[-10:]}")

# 最终判断
print()
print("="*80)
if hf_tokens == llaisys_tokens:
    print("✓✓✓ 测试通过！LLAISYS和HuggingFace完全一致 ✓✓✓")
    print(f"✓✓✓ 成功生成{total_new}个token，全部匹配 ✓✓✓")
    sys.exit(0)
else:
    print(f"✗ 测试失败：{len(hf_tokens) - matches}个token不匹配")
    sys.exit(1)
