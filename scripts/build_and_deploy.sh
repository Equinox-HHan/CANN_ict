#!/bin/bash
# ==============================================================================
# 算子工程一键编译、打包与系统部署脚本
# 使用方式: bash scripts/build_and_deploy.sh [可选算子目录路径]
# ==============================================================================

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 默认算子目录
DEFAULT_OP_DIR="$PROJECT_ROOT/ascendc_operator_development/vector_op_development/custom_op"
TARGET_OP_DIR="${1:-$DEFAULT_OP_DIR}"

if [ ! -d "$TARGET_OP_DIR" ]; then
    echo "[-] 错误: 算子目录不存在: $TARGET_OP_DIR"
    exit 1
fi

echo "=================================================="
echo ">>> 开始编译并部署算子工程: $TARGET_OP_DIR"
echo "=================================================="

# 1. 检查环境变量
if [ -z "$ASCEND_TOOLKIT_HOME" ] || [ -z "$CURRENT_ASCEND_SOC" ]; then
    echo "[*] 检测到尚未载入环境，正在自动执行 env_setup.sh..."
    source "$PROJECT_ROOT/scripts/env_setup.sh"
fi

cd "$TARGET_OP_DIR"

# 2. 自动更新 CMakePresets.json 中的路径与芯片型号（避免跨环境硬编码失效）
PRESETS_FILE="$TARGET_OP_DIR/CMakePresets.json"
if [ -f "$PRESETS_FILE" ]; then
    echo "[+] 动态适配 CMakePresets.json:"
    echo "    - CANN 路径: $ASCEND_TOOLKIT_HOME"
    echo "    - 计算架构:  $CURRENT_ASCEND_SOC"
    
    python3 - <<EOF
import json

file_path = "$PRESETS_FILE"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for preset in data.get("configurePresets", []):
    cache = preset.get("cacheVariables", {})
    if "ASCEND_CANN_PACKAGE_PATH" in cache:
        cache["ASCEND_CANN_PACKAGE_PATH"]["value"] = "$ASCEND_TOOLKIT_HOME"
    if "ASCEND_COMPUTE_UNIT" in cache:
        cache["ASCEND_COMPUTE_UNIT"]["value"] = "$CURRENT_ASCEND_SOC"

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
EOF
fi

# 3. 执行编译打包
echo "[+] 正在构建算子..."
if [ -f "build.sh" ]; then
    bash build.sh
else
    echo "[-] 未找到 build.sh，退出构建"
    exit 1
fi

# 4. 安装生成的自定义算子 run 包
RUN_PACKAGE=$(ls build_out/custom_opp_*.run 2>/dev/null | head -n 1)
if [ -z "$RUN_PACKAGE" ]; then
    echo "[-] 构建未生成 custom_opp_*.run 安装包，请检查编译日志！"
    exit 1
fi

echo "[+] 发现算子安装包: $RUN_PACKAGE"
echo "[+] 正在部署自定义算子到系统 OPP 目录..."
./"$RUN_PACKAGE"

echo "=================================================="
echo ">>> 恭喜！算子已成功编译并部署到系统！"
echo ">>> 现在可以在 Python / ACL 中调用最新算子了。"
echo "=================================================="
