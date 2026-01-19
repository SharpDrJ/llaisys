#!/bin/bash

# 设置环境变量
export XMAKE_ROOT=y
export PYTHONPATH=$PYTHONPATH:$PWD/python

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "==== 开始编译与安装 LLAISYS ===="
xmake && xmake install
if [ $? -ne 0 ]; then
    echo -e "${RED}编译失败，请检查 C++ 代码。${NC}"
    exit 1
fi
echo -e "${GREEN}编译安成功！${NC}"

echo -e "\n==== 开始验证所有算子 (CPU) ===="

ops=("add" "argmax" "embedding" "linear" "rms_norm" "rope" "self_attention" "swiglu")
failed_ops=()

for op in "${ops[@]}"; do
    echo -e "\n正在测试算子: ${op}..."
    python3 test/ops/${op}.py --device cpu
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[PASS] ${op}${NC}"
    else
        echo -e "${RED}[FAIL] ${op}${NC}"
        failed_ops+=("${op}")
    fi
done

echo -e "\n================================"
if [ ${#failed_ops[@]} -eq 0 ]; then
    echo -e "${GREEN}恭喜！所有算子测试全部通过！${NC}"
else
    echo -e "${RED}测试完成，但以下算子未通过: ${failed_ops[*]}${NC}"
    exit 1
fi
