# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build C++ backend and install to Python package
xmake                    # Compile all C++ components
xmake install            # Install shared library to python/llaisys/libllaisys/
pip install ./python/    # Install Python package

# Configure with NVIDIA GPU support (requires CUDA)
xmake f --nv-gpu=y -cv
xmake
xmake install

# Rebuild after C++ changes
xmake                    # Rebuilds only changed files
```

## Testing Commands

```bash
# Runtime tests (device initialization and memory)
python test/test_runtime.py --device cpu
python test/test_runtime.py --device nvidia    # Requires CUDA support

# Tensor implementation tests
python test/test_tensor.py

# Operator tests (all or individual)
python test/test_ops.py
python test/ops/add.py --profile    # Profile specific operator

# Model inference tests
python test/test_infer.py --model [path/to/model] --test
python test/test_infer.py --model [path/to/model] --test --device nvidia
```

## Architecture Overview

LLAISYS is a layered C++/Python framework for AI system development:

### Layer Structure
1. **C++ Backend** (`src/`) - Core implementation compiled to shared library
2. **C API** (`include/` & `src/llaisys/`) - Public interface using `__export` macro
3. **Python Bindings** (`python/llaisys/libllaisys/`) - ctypes wrappers of C API
4. **Python Package** (`python/llaisys/`) - Pythonic API wrapper

### Core Abstractions

**Tensor System** (`src/tensor/`):
- `storage`: Shared pointer to memory block (can be shared between tensors)
- `offset`: Byte offset into storage
- `meta`: Shape, dtype, strides
- Supports non-contiguous views via strides; operations like `permute()` and `slice()` don't copy data

**Device Abstraction** (`src/device/`, `src/core/`):
- Thread-local `Context` manages device `Runtime` objects
- `Runtime`: Device-specific resource manager with generic API interface
- Each operator dispatches to device-specific implementation (`cpu/`, `nvidia/` subdirectories)
- Switch devices via `context().setDevice(device_type, device_id)`

**Operator System** (`src/ops/`):
- Each operator has directory with `op.hpp` and `op.cpp`
- Device implementations in subdirectories: `cpu/`, `nvidia/` (conditional)
- Operator entry point in `op.cpp` validates inputs and dispatches to device
- All operators must support at least Float32, Float16, BFloat16

### Build Target Dependency Chain

```
llaisys-utils (utility functions)
    ↓
llaisys-device-cpu → llaisys-device
    ↓                    ↓
    └───────→ llaisys-core
              ↓
          llaisys-tensor
              ↓
          llaisys-ops-cpu → llaisys-ops
              ↓
          llaisys (shared library with C API)
```

### Code Organization Conventions

- **Namespace**: All C++ code uses `llaisys::` namespace
- **Exported Functions**: C API functions marked with `__export` macro in headers
- **Error Checking**: Macros `CHECK_SAME_DEVICE`, `CHECK_SAME_SHAPE`, `CHECK_SAME_DTYPE`
- **Device Dispatch**: Pattern in operators:
  ```cpp
  if (device_type == LLAISYS_DEVICE_CPU) {
      return cpu::operation(...);
  }
  switch (device_type) {
      case LLAISYS_DEVICE_CPU: ...
  #ifdef ENABLE_NVIDIA_API
      case LLAISYS_DEVICE_NVIDIA: ...
  #endif
  }
  ```

### Assignment Progression

- **Assignment #0**: Setup, build, runtime tests
- **Assignment #1**: Tensor implementation (`load`, `isContiguous`, `view`, `permute`, `slice`)
- **Assignment #2**: Operators (`argmax`, `embedding`, `linear`, `rms_norm`, `rope`, `self_attention`, `swiglu`)
- **Assignment #3**: Full model inference (DeepSeek-R1-Distill-Qwen-1.5B) with KV cache

### Adding New Operators

1. Create directory `src/ops/<op_name>/` with:
   - `op.hpp`: Forward declaration in `llaisys::ops` namespace
   - `op.cpp`: Device dispatch and validation
   - `cpu/<op_name>_cpu.hpp` and `cpu/<op_name>_cpu.cpp`: CPU implementation
2. Add files to `xmake/cpu.lua` in `llaisys-ops-cpu` target
3. Create C API wrapper in `include/llaisys/ops/` and `src/llaisys/ops/`
4. Add Python ctypes wrapper in `python/llaisys/libllaisys/`
5. Add Pythonic wrapper in `python/llaisys/ops.py`

### Adding CUDA Support

1. Create `xmake/nvidia.lua` (see `xmake/cpu.lua` as reference) with:
   - `llaisys-device-nvidia` target
   - `llaisys-ops-nvidia` target
2. Implement `nvidia::getRuntimeAPI()` in `src/device/nvidia/`
3. Implement CUDA kernels in operator `nvidia/` subdirectories
4. Enable with `xmake f --nv-gpu=y` (defines `ENABLE_NVIDIA_API` macro)
5. All CUDA code is conditionally compiled behind `#ifdef ENABLE_NVIDIA_API`

### Testing Pattern

Tests compare LLAISYS output with PyTorch reference:
- Create matching tensors in both frameworks
- Run operation on both
- Compare results with tolerance for floating-point
- Use `tensor.debug()` to inspect tensor data during debugging

### Key Implementation Details

- **Type Casting**: Naive casting helper in `src/utils/` for Float16/BFloat16 support
- **Memory Management**: RAII with smart pointers throughout
- **Platform Support**: Windows (MSVC), Linux (GCC/Clang), macOS (Clang)
- **Warnings Treated as Errors**: `-Werror` equivalent in xmake config
