# Qwen2模型推理修复总结（简版）

## 快速开始

### 构建和运行

```bash
# 1. 构建项目
xmake f --root
xmake && xmake install
pip install ./python/

# 2. 运行推理
python -c "
import llaisys
model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
tokens = [151646, 151646, 151644]
generated = model.generate(tokens, max_new_tokens=64)
print(f'Generated: {generated}')
"
```

### 代码结构速览

```
src/models/qwen2/
├── qwen2_model.{hpp,cpp}    # 主模型：28层Transformer管理
├── qwen2_block.{hpp,cpp}    # Transformer块：Attn + MLP
├── qwen2_attention.{hpp,cpp} # 注意力：Q/K/V投影 + RoPE + Self-Attn
├── qwen2_mlp.{hpp,cpp}      # MLP：SwiGLU激活
└── kv_cache.{hpp,cpp}       # KV缓存：存储历史K/V
```

### 推理流程

```
输入tokens
  ↓
Embedding查找 → [seq_len, hidden_size]
  ↓
28层Transformer（每层）:
  ├─ RMS归一化
  ├─ 注意力: Q/K/V投影 → RoPE → KV缓存 → Self-Attention
  ├─ 残差连接
  ├─ RMS归一化
  ├─ MLP: Gate/Up投影 → SwiGLU → Down投影
  └─ 残差连接
  ↓
取最后token → LM Head → Argmax → 输出token
```

---

## 问题与解决方案

### 核心问题
**Tensor内存布局不匹配**导致`self_attention`计算错误，产生NaN值。

| 组件 | 期望格式 | 实际传入格式 | 状态 |
|------|---------|-------------|------|
| self_attention | `[num_heads * seq_len, head_dim]` (头优先) | `[seq_len, num_heads, head_dim]` (序列优先) | ❌ 不匹配 |

### 修复方案

#### 1. 实现`reshapeAndRearrange()`函数
```cpp
// 转换流程
[seq_len, num_heads * head_dim]
  → view → [seq_len, num_heads, head_dim]
  → permute(1,0,2) → [num_heads, seq_len, head_dim]
  → contiguous → 连续内存
  → view → [num_heads * seq_len, head_dim] ✓
```

#### 2. 修复KV缓存拼接
```cpp
// 缓存是头优先格式，拼接时需要：
缓存: [num_heads * cached_len, head_dim]
  → view → [cached_len, num_heads, head_dim]
  → concat (dim=0) → [total_len, num_heads, head_dim]
  → reshapeAndRearrange → [num_heads * total_len, head_dim] ✓
```

## 测试结果

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 基础推理（10 tokens） | ✅ 通过 | 与HuggingFace完全匹配 |
| 64 tokens生成 | ✅ 通过 | 无错误，稳定运行 |
| 128 tokens生成 | ✅ 通过 | 76 tokens后正常结束 |
| 性能 | 12x加速 | 34秒 vs HuggingFace 416秒 |

## 关键修改

**文件：** `src/models/qwen2/qwen2_attention.cpp`

**新增函数：**
```cpp
tensor_t reshapeAndRearrange(tensor_t x, size_t num_heads, size_t head_dim);
```

**修改逻辑：**
- Q、K、V在传入`self_attention`前都经过`reshapeAndRearrange()`
- KV缓存拼接时先转为序列优先，拼接后再转回头优先

## 技术要点

### 内存布局对比
```
序列优先: [s0_h0, s0_h1, ..., s1_h0, s1_h1, ...]
头优先:   [h0_s0, h0_s1, ..., h1_s0, h1_s1, ...]
```

### 为什么需要contiguous？
- `permute()`只改变strides，不改变内存顺序
- 后续`view()`要求连续内存
- `contiguous()`创建新的连续内存副本

## 总结

✅ **已解决：**
1. Tensor布局不匹配导致NaN
2. KV缓存拼接破坏内存布局
3. 长序列生成稳定性

✅ **验证通过：**
- 功能正确性（与HuggingFace对比）
- 长序列生成（64/128 tokens）
- 性能提升（12x加速）

---

**完整文档：** `WORK_SUMMARY.md`
