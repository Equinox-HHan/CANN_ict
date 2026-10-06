#!/bin/bash
# ==============================================================================
# CANN 开发环境一键初始化与自检脚本
# 使用方式: source scripts/env_setup.sh
# ==============================================================================

echo "=================================================="
echo ">>> 开始初始化昇腾 CANN 开发环境..."
echo "=================================================="

# 1. 寻找 CANN Toolkit 安装根目录
CANN_CANDIDATE_PATHS=(
    "/usr/local/Ascend/ascend-toolkit/latest"
    "/home/developer/Ascend/ascend-toolkit/latest"
    "/home/developer/Ascend/cann-9.0.0"
    "/usr/local/Ascend/latest"
)

FOUND_CANN_PATH=""
if [ -n "$ASCEND_TOOLKIT_HOME" ] && [ -d "$ASCEND_TOOLKIT_HOME" ]; then
    FOUND_CANN_PATH="$ASCEND_TOOLKIT_HOME"
else
    for p in "${CANN_CANDIDATE_PATHS[@]}"; do
        if [ -d "$p" ]; then
            FOUND_CANN_PATH="$p"
            break
        fi
    done
fi

if [ -z "$FOUND_CANN_PATH" ]; then
    echo "[-] 未能自动找到 CANN Toolkit 安装路径！"
    echo "    请手动检查 /usr/local/Ascend 或 /home/developer/Ascend 目录。"
    return 1 2>/dev/null || exit 1
fi

echo "[+] 找到 CANN Toolkit: $FOUND_CANN_PATH"

# 2. 载入官方环境变量
if [ -f "$FOUND_CANN_PATH/set_env.sh" ]; then
    source "$FOUND_CANN_PATH/set_env.sh"
    echo "[+] 已执行: source $FOUND_CANN_PATH/set_env.sh"
elif [ -f "$FOUND_CANN_PATH/bin/set_env.sh" ]; then
    source "$FOUND_CANN_PATH/bin/set_env.sh"
    echo "[+] 已执行: source $FOUND_CANN_PATH/bin/set_env.sh"
fi

# 补充基础导出变量
export ASCEND_TOOLKIT_HOME="$FOUND_CANN_PATH"
export ASCEND_HOME_PATH="$FOUND_CANN_PATH"
export PATH="$FOUND_CANN_PATH/bin:$FOUND_CANN_PATH/compiler/bin:$FOUND_CANN_PATH/tools/tikcpp/bin:$PATH"

# 3. 配置动态库搜索路径 (特别是 Pybind / ACLNN 所需的核心库)
OP_API_LIB="$FOUND_CANN_PATH/opp/built-in/op_impl/ai_core/tbe/op_api/lib"
OPP_RUN_LIB="$FOUND_CANN_PATH/opp/vendors/customize/op_api/lib"

export LD_LIBRARY_PATH="$FOUND_CANN_PATH/lib64:$OP_API_LIB:$OPP_RUN_LIB:$LD_LIBRARY_PATH"
export PYTHONPATH="$FOUND_CANN_PATH/python/site-packages:$PYTHONPATH"

# 4. 自动检测 NPU 芯片型号 (用于 CMakePresets.json 中的 ASCEND_COMPUTE_UNIT)
DETECTED_SOC=""
if command -v npu-smi &> /dev/null; then
    NPU_INFO=$(npu-smi info 2>/dev/null)
    if echo "$NPU_INFO" | grep -qi "910B"; then
        DETECTED_SOC="ascend910b"
    elif echo "$NPU_INFO" | grep -qi "310P"; then
        DETECTED_SOC="ascend310p"
    elif echo "$NPU_INFO" | grep -qi "910"; then
        DETECTED_SOC="ascend910"
    fi
fi

if [ -z "$DETECTED_SOC" ]; then
    # 默认值回退
    DETECTED_SOC="ascend910b"
    echo "[!] 未能通过 npu-smi 检测到具体芯片，默认指定 SOC: $DETECTED_SOC"
else
    echo "[+] 检测到当前 NPU 芯片架构 SOC: $DETECTED_SOC"
fi
export CURRENT_ASCEND_SOC="$DETECTED_SOC"

# 5. 检查关键工具与 Python 依赖
echo "--------------------------------------------------"
echo "环境依赖自检:"
echo " - Python 解释器: $(which python3) ($(python3 --version 2>&1))"
echo " - CMake 版本:    $(cmake --version 2>/dev/null | head -n 1 || echo '未安装')"

python3 -c "import torch; print(f' - PyTorch:         {torch.__version__}')" 2>/dev/null || echo " - PyTorch:         未检测到"
python3 -c "import torch_npu; print(f' - torch_npu:       {torch_npu.__version__}')" 2>/dev/null || echo " - torch_npu:       未检测到"
python3 -c "import pybind11; print(f' - pybind11:        {pybind11.__version__}')" 2>/dev/null || echo " - pybind11:        未安装 (可通过 pip install pybind11 安装)"

echo "=================================================="
echo ">>> CANN 环境变量配置完毕！"
echo "=================================================="
