#include"register/op_def_registry.h"//固定引用
#include"../op_kernel/add_custom_template_tiling.h"//自定义的Tiling结构体文件
#include"graph/utils/type_utils.h"//数据类型相关的工具类
#include"tiling/platform/platform_ascendc.h"//平台类型相关的工具类

//获取输入大小和核数
namespace optiling{
    static ge::graphStatus TilingFunc(gert::TilingContext *context){
        auto ascendcPlatform=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto coreNum=ascendcPlatform.GetCoreNum();
        uint32_t inputNum=context->GetInputShape(0)//获取第0个输入的Tensor描述
        ->GetStorageShape()//获取实际物理存储形式（考虑了内存排布对齐等）
        .GetShapeSize();//自动计算各维度乘积：Tensor包含的总元素个数
        uint32_t typeLength=0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType()/*提取实际传入的数据类型*/,typeLength);
        //将类型转换为实际所占的字节数
        uint32_t inputLength=inputNum*typeLength;
        //设置用户需要的workspace大小为256字节
        //size_t usrSize=256;
        //size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        //通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数,当前限制使用一块
        //currentWorkspace[0] = usrSize + sysWorkspaceSize;
        //设置总的workspace的数值大小，总的workspace空间由框架来申请并管理


        //计算每个核处理的数据量(核外数据切分)
        /*
        依据内存对齐原则：
        先将输入数据的总大小向上对齐至32B整数倍，得到对齐后的总32B数据块数；
        以每个核至少分配1个32B数据块为基准，判断是否启用全部核数：若对齐后的数据块总量过小，核数不足时则至少启用1个核完成计算；
        确定实际使用的核数后，计算核均基础处理的32B数据块数，对无法均分的剩余数据块，采用前N核补块的方式分配：
        让前tailBlockNum个核各多处理1个32B数据块，其余核按基础数处理；
        按补块规则划分核类型：多处理1个32B 数据块的核称为大核，按基础数处理的核称为小核
        */

        const uint32_t BLOCK_SIZE=32;
        uint32_t inputLengthAlign32=(((inputLength+BLOCK_SIZE-1)/BLOCK_SIZE)*BLOCK_SIZE);
        coreNum=std::min(coreNum,inputLengthAlign32/BLOCK_SIZE);
        coreNum=std::max(coreNum,static_cast<uint32_t>(1));
        uint32_t everyCoreInputBlockNum=inputLengthAlign32/BLOCK_SIZE/coreNum;//每个核处理的32B数据块的个数
        uint32_t tailBlockNum=(inputLengthAlign32/BLOCK_SIZE)%coreNum;
        context->SetBlockDim(coreNum);

        //核内数据切分
        /*
        1.获取单核可用的UB大小
        2.假设Add算子两个输入一个输出：总共要使用三个Buffer块
        3.计算出单个Buffer最多可容纳32B数据块和对应的元素个数
        */

        uint64_t ubSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubSize);
        //获取硬件平台存储空间的大小
        uint32_t ubDataNumber=3;
        uint32_t tileBlockNum=(ubSize/BLOCK_SIZE)/ubDataNumber;//单核每次最多可处理的数据块个数
        uint32_t tileDataNum=(tileBlockNum*BLOCK_SIZE)/typeLength;//每个核单次可以处理的数据量

        //针对小核的计算
        uint32_t smallCoreDataNum=everyCoreInputBlockNum*BLOCK_SIZE/typeLength;
        uint32_t smallTileNum=everyCoreInputBlockNum/tileBlockNum;
        //每个核分到的数据不能一次运算完成时需要循环计算的次数
        uint32_t finalSmallTileNum=(everyCoreInputBlockNum%tileBlockNum)==0?smallTileNum:smallTileNum+1;
        uint32_t smallTailDataNum=smallCoreDataNum-(tileDataNum*smallTileNum);//最后一次循环需要处理的数据量
        smallTailDataNum=smallTailDataNum==0?tileDataNum:smallTailDataNum;

        //针对大核的计算
        everyCoreInputBlockNum+=1;
        uint32_t bigCoreDataNum=everyCoreInputBlockNum*BLOCK_SIZE/typeLength;
        uint32_t bigTileNum=everyCoreInputBlockNum/tileBlockNum;
        uint32_t finalBigTileNum=(everyCoreInputBlockNum%tileBlockNum)==0 ? bigTileNum:bigTileNum + 1;
        uint32_t bigTailDataNum=bigCoreDataNum-tileDataNum*bigTileNum;
        bigTailDataNum=bigTailDataNum==0?tileDataNum:bigTailDataNum;
    }
}

//算子原型注册和shape推导
namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *intputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *intputShape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context)
        {
            context->SetOutputDataType(0, context->GetInputDataType(0));
            return ge::GRAPH_SUCCESS;
        }
}  // namespace ge

namespace ops {
    class AddCustomTemplate : public OpDef {
    public:
        explicit AddCustomTemplate(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b")
                .AddConfig("ascend910_93");
        }
    };
    OP_ADD(AddCustomTemplate);
}  // namespace ops