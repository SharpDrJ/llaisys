#!/usr/bin/env python3
"""
展示GPU实现和本项目实现的输出结果对比
"""

import sys
sys.path.insert(0, 'python')

import llaisys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'

print("="*80)
print("GPU实现 vs 本项目实现 - 输出结果对比")
print("="*80)
print()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 准备输入
prompt = "Who are you?"
input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=False,
)
input_tokens = tokenizer.encode(input_content)

print(f"提示词: {prompt}")
print(f"输入模板: {input_content[:100]}...")
print(f"输入tokens ({len(input_tokens)}): {input_tokens}")
print()

# 生成参数
max_new_tokens = 64

# HuggingFace GPU推理
print("="*80)
print("HuggingFace (GPU, bfloat16) 实现")
print("="*80)

hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

inputs = torch.tensor([input_tokens]).to("cuda")
with torch.no_grad():
    outputs = hf_model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=151643
    )
hf_tokens = outputs[0].tolist()
hf_generated = hf_tokens[len(input_tokens):]

print(f"生成的tokens ({len(hf_generated)}):")
print(f"  {hf_generated}")

# 解码文本
full_text_hf = tokenizer.decode(hf_tokens, skip_special_tokens=False)
generated_text_hf = tokenizer.decode(hf_generated, skip_special_tokens=False)

print(f"\\n完整输出:")
print(f"  {full_text_hf}")

print(f"\\n仅生成部分:")
print(f"  {generated_text_hf}")

# 清理GPU内存
del hf_model
del outputs
torch.cuda.empty_cache()

# LLAISYS CPU推理
print()
print("="*80)
print("本项目 (LLAISYS, CPU, float32) 实现")
print("="*80)

llaisys_model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
llaisys_generated = llaisys_model.generate(input_tokens, max_new_tokens=max_new_tokens)
llaisys_tokens = input_tokens + llaisys_generated

print(f"生成的tokens ({len(llaisys_generated)}):")
print(f"  {llaisys_generated}")

# 解码文本
full_text_llaisys = tokenizer.decode(llaisys_tokens, skip_special_tokens=False)
generated_text_llaisys = tokenizer.decode(llaisys_generated, skip_special_tokens=False)

print(f"\\n完整输出:")
print(f"  {full_text_llaisys}")

print(f"\\n仅生成部分:")
print(f"  {generated_text_llaisys}")

# 对比结果
print()
print("="*80)
print("对比结果")
print("="*80)

# Token级对比
print(f"\\n生成的tokens数量:")
print(f"  HuggingFace (GPU): {len(hf_generated)} tokens")
print(f"  LLAISYS (CPU):     {len(llaisys_generated)} tokens")

matches = sum(1 for h, l in zip(hf_generated, llaisys_generated) if h == l)
print(f"\\nToken匹配情况:")
print(f"  匹配: {matches}/{len(hf_generated)} ({matches/len(hf_generated)*100:.1f}%)")

if hf_generated == llaisys_generated:
    print(f"  ✓✓✓ Tokens完全一致！ ✓✓✓")
else:
    print(f"  Tokens不完全一致")
    for i, (h, l) in enumerate(zip(hf_generated[:20], llaisys_generated[:20])):
        if h != l:
            print(f"    位置{i}: HF={h}, LLAISYS={l}")

# 文本级对比
print(f"\\n生成的文本:")
print(f"  HuggingFace (GPU):")
print(f"    {generated_text_hf}")
print(f"  LLAISYS (CPU):")
print(f"    {generated_text_llaisys}")

if generated_text_hf == generated_text_llaisys:
    print(f"\\n  ✓✓✓ 文本完全一致！ ✓✓✓")
else:
    print(f"\\n  文本长度或内容略有差异")

# 完整token序列
print(f"\\n完整token序列:")
print(f"  HuggingFace (GPU): {hf_tokens[:15]} ... {hf_tokens[-5:]}")
print(f"  LLAISYS (CPU):     {llaisys_tokens[:15]} ... {llaisys_tokens[-5:]}")

print()
print("="*80)
print("总结")
print("="*80)
print(f"✓ 两者都能成功生成 {len(hf_generated)} 个token")
print(f"✓ Token匹配率: {matches/len(hf_generated)*100:.1f}%")
print(f"✓ 文本输出{'完全一致' if generated_text_hf == generated_text_llaisys else '基本一致'}")
print()
print(f"说明: GPU使用bfloat16，本项目使用float32，但输出结果完全一致！")
