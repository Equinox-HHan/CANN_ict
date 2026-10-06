├── opp                  // 算子库目录
│   ├── vendors          // 自定义算子所在目录
│   │   ├── config.ini
│   │   └── customize // 自定义算子安装包时配置的vendor_name，默认值为customize
│   │       ├── op_api
│   │       │   ├── include
│   │       │   │   └── aclnn_xx.h
│   │       │   └── lib
│   │       │       └── libcust_opapi.so




======================= 1. 你的源码仓库 (Git 管理) =======================
/mnt/workspace/CANN_ict/
├── ascendc_operator_development/
│   ├── vector_op_development/
│   │   ├── add_custom.json
│   │   └── custom_op/                    <--- [A. 算子开发工程 (生产者)]
│   │       ├── build.sh
│   │       ├── CMakeLists.txt
│   │       ├── op_host/                  (Tiling 及算子原型代码)
│   │       ├── op_kernel/                (AI Core 核心核函数代码)
│   │       └── build_out/                <--- 编译中间产物
│   │           └── custom_opp_*.run      (打包出来的安装包)
│   │
│   └── 03_intermediate_.../
│       └── 03.03_call_project/           <--- [B. 调用工程/测试包 (消费者)]
│           ├── main.cpp (或 pybind.cpp)   (单算子 API 调用代码)
│           ├── CMakeLists.txt            (配置如何找到下面的 OPP 库)
│           └── test_call.py              (Python 测试脚本)
│
======================= 2. 算子安装部署目录 (系统/用户空间) =================
${INSTALL_DIR}/ (默认是 CANN 根目录，或用 --install-path 指定的目录)
└── opp/                                  <--- [C. OPP 算子库 (桥梁与纽带)]
    └── vendors/
        ├── config.ini
        └── customize/                    (对应 CMakePresets 中的 vendor_name)
            ├── op_api/                   ★★ 核心交汇点 ★★
            │   ├── include/
            │   │   └── aclnn_add_custom_template.h  (生成的单算子 C++ 头文件)
            │   └── lib/
            │       └── libcust_opapi.so             (生成的单算子动态链接库)
            ├── op_impl/ai_core/          (编译好的 NPU 机器码二进制)
            └── op_proto/lib/             (算子原型库)




CANN_ict/
├── .gitignore
├── scripts/
│   ├── env_setup.sh               # 环境变量一键初始化
│   └── build_and_deploy.sh        # 算子一键构建部署到 OPP
│
└── ascendc_operator_development/
    └── vector_op_development/
        │
        ├── custom_op/             # 【A. 算子工程 (生产者)】
        │   ├── add_custom.json    # (建议移到这里) 算子规格输入
        │   ├── CMakeLists.txt
        │   ├── CMakePresets.json
        │   ├── build.sh
        │   ├── op_host/           # Host 侧 (Tiling)
        │   └── op_kernel/         # Kernel 侧 (算子计算类)
        │
        └── acl_pybind_call/       # 【B. 调用与测试工程 (消费者)】
            ├── CMakeLists.txt     # 配置如何引入系统 OPP 里的头文件和库
            ├── pybind_add.cpp     # Pybind11 / C++ 包装单算子 API
            └── test_add.py        # Python 测试脚本 (验证算子精度与耗时)