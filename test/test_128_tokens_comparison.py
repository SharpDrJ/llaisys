#!/usr/bin/env python3
"""
完整的128 token生成测试
对比LLAISYS和HuggingFace实现的一致性
"""

import sys
sys.path.insert(0, 'python')

import llaisys
import torch
from transformers import AutoModelForCausalLM
import time

model_path = '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'

print("="*80)
print("128 Token完整对比测试")
print("="*80)
print(f"模型路径: {model_path}")
print()

# 加载HuggingFace模型
print("正在加载HuggingFace模型...")
hf_start = time.time()
hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map='cpu'
)
hf_load_time = time.time() - hf_start
print(f"✓ HuggingFace模型加载完成 ({hf_load_time:.2f}秒)")

# 加载LLAISYS模型
print("\n正在加载LLAISYS模型...")
llaisys_start = time.time()
llaisys_model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
llaisys_load_time = time.time() - llaisys_start
print(f"✓ LLAISYS模型加载完成 ({llaisys_load_time:.2f}秒)")

# 初始tokens
tokens = [151646, 151646, 151644]
print(f"\n初始tokens: {tokens}")
print(f"目标: 生成128个新token")
print()

# 测试
hf_tokens = tokens.copy()
llaisys_tokens = tokens.copy()

print("="*80)
print("开始生成128个token...")
print("="*80)
print()

mismatches = []
hf_times = []
llaisys_times = []

for i in range(128):
    # HuggingFace推理
    hf_step_start = time.time()
    inputs = torch.tensor([hf_tokens]).to(torch.long)
    with torch.no_grad():
        outputs = hf_model(inputs)
        logits = outputs.logits[0, -1, :]
        next_hf = logits.argmax().item()
    hf_tokens.append(next_hf)
    hf_step_time = time.time() - hf_step_start
    hf_times.append(hf_step_time)

    # LLAISYS推理
    llaisys_step_start = time.time()
    next_llaisys = llaisys_model.generate([llaisys_tokens[-1]], max_new_tokens=1)[0]
    llaisys_tokens.append(next_llaisys)
    llaisys_step_time = time.time() - llaisys_step_start
    llaisys_times.append(llaisys_step_time)

    # 对比
    match = (next_hf == next_llaisys)
    status = "✓" if match else "✗"

    # 每10个token或发现不匹配时打印
    if (i + 1) % 10 == 0 or not match:
        print(f"Token {i+1:3d}: HF={next_hf:6d}, LLAISYS={next_llaisys:6d} {status} "
              f"(HF: {hf_step_time*1000:5.1f}ms, LLAISYS: {llaisys_step_time*1000:5.1f}ms)")

    if not match:
        mismatches.append({
            'position': i + 1,
            'hf': next_hf,
            'llaisys': next_llaisys,
            'hf_time': hf_step_time,
            'llaisys_time': llaisys_step_time
        })

# 结果统计
print()
print("="*80)
print("测试结果")
print("="*80)

total_tokens = 128
match_count = total_tokens - len(mismatches)
match_rate = (match_count / total_tokens) * 100

print(f"\n总生成tokens: {total_tokens}")
print(f"匹配数量: {match_count}")
print(f"不匹配数量: {len(mismatches)}")
print(f"匹配率: {match_rate:.2f}%")

# 性能统计
avg_hf_time = sum(hf_times) / len(hf_times) * 1000
avg_llaisys_time = sum(llaisys_times) / len(llaisys_times) * 1000
total_hf_time = sum(hf_times)
total_llaisys_time = sum(llaisys_times)

print(f"\n性能统计:")
print(f"  HuggingFace:")
print(f"    平均每token: {avg_hf_time:.2f}ms")
print(f"    总耗时: {total_hf_time:.2f}秒")
print(f"  LLAISYS:")
print(f"    平均每token: {avg_llaisys_time:.2f}ms")
print(f"    总耗时: {total_llaisys_time:.2f}秒")
print(f"  加速比: {total_hf_time/total_llaisys_time:.2f}x")

# 不匹配详情
if mismatches:
    print(f"\n不匹配详情 (共{len(mismatches)}个):")
    print(f"{'位置':>6} {'HuggingFace':>12} {'LLAISYS':>12} {'差异':>10}")
    print("-" * 50)
    for m in mismatches[:20]:  # 只显示前20个
        diff = m['llaisys'] - m['hf']
        print(f"{m['position']:>6} {m['hf']:>12} {m['llaisys']:>12} {diff:>+10}")
    if len(mismatches) > 20:
        print(f"... 还有 {len(mismatches) - 20} 个不匹配")
else:
    print("\n✓✓✓ 所有128个token完全匹配！ ✓✓✓")

# Token序列对比
print(f"\nToken序列对比:")
print(f"  HuggingFace (最后20个): {hf_tokens[-20:]}")
print(f"  LLAISYS (最后20个):      {llaisys_tokens[-20:]}")

# 完整序列
print(f"\n完整序列:")
print(f"  长度: {len(hf_tokens)} tokens")
print(f"  前10个: {hf_tokens[:10]}")
print(f"  后10个: {hf_tokens[-10:]}")

# 最终判断
print()
print("="*80)
if len(mismatches) == 0:
    print("✓✓✓ 测试通过！LLAISYS和HuggingFace完全一致 ✓✓✓")
    sys.exit(0)
else:
    print(f"✗ 测试失败：{len(mismatches)}个token不匹配")
    sys.exit(1)
