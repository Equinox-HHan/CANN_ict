//引用固定头文件
//强调：模板没有使用Double Buffer
#include"kernel_operator.h"
#include"add_custom_template_tiling.h"
#include"kernel_operator_dump_tensor_intf_impl.h"
constexpr int32_t BUFFER_NUM=1;
constexpr int32_t QUEUE_DEPTH=1;

//定义算子类
template<typename TYPE_X,typename TYPE_Y,typename TYPE_Z>
class KernelAdd{
    public:
        __aicore__ inline KernelAdd(){}
        //声明init函数并且完成算子参数的初始化
        /*
        1. 基于大核处理元素数，计算当前核的GM地址偏移
        2. 通过当前核索引与tailBlockNum的对比结果判断核类型
            1. 若为大核，直接配置算子类的大核规格参数
            2. 若为小核，则配置小核规格参数，同时重新计算GlobalMemory地址偏移
        1. 最后根据已配置的核规格参数，完成GlobalTensor的初始化与LocalMemory的内存分配
        */ 
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t smallCoreDataNum,uint32_t bigCoreDataNum, 
            uint32_t finalBigTileNum, uint32_t finalSmallTileNum, uint32_t tileDataNum, 
            uint32_t smallTailDataNum, uint32_t bigTailDataNum, uint32_t tailBlockNum) {
                
                uint32_t blockIdx = AscendC::GetBlockIdx();
                uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
                this->tileDataNum = tileDataNum;
                if (blockIdx < tailBlockNum) { 
                    this->coreDataNum = bigCoreDataNum;
                    this->tileNum = finalBigTileNum;
                    this->tailDataNum = bigTailDataNum;
                }
                else { 
                    this->coreDataNum = smallCoreDataNum;
                    this->tileNum = finalSmallTileNum;
                    this->tailDataNum = smallTailDataNum;
                    globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
                }
                xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
                yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
                zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + globalBufferIndex, this->coreDataNum);
                pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
                pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
                pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Z));
                // 调用InitBuffer接口为TBuf变量分配内存
                //pipe.InitBuffer(tmpBuf0, TOTAL_LENGTH * sizeof(float));
                //pipe.InitBuffer(tmpBuf1, TOTAL_LENGTH * sizeof(float));
                //pipe.InitBuffer(tmpBuf2, TOTAL_LENGTH * sizeof(float));
            }
        __aicore__ inline void Process(){
            int32_t loopCount = this->tileNum;
            this->processDataNum = this->tileDataNum;
            for (int32_t i = 0; i < loopCount; i++) {
                if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
                }
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }

    private:
        __aicore__ inline void CopyIn(int32_t progress){
            AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
            AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
            AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
            AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
            inQueueX.EnQue(xLocal);
            inQueueY.EnQue(yLocal);
        }

        //compute函数获取暂存数据的Tensor，使用Tensor参与运算
        //不要对TBuf.Get()获取的Tensor进行入队出队或者释放操作
        __aicore__ inline void Compute(int32_t progress){
            AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
            AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
            AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();

            // 获取暂存数据的Tensor
            //AscendC::LocalTensor<float> tmpTensor0 = tmpBuf0.Get<float>();
            //AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();
            //AscendC::LocalTensor<float> tmpTensor2 = tmpBuf2.Get<float>();
            //AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);
            //AscendC::Cast(tmpTensor1, yLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);
            //AscendC::Add(tmpTensor2, tmpTensor0, tmpTensor1, TOTAL_LENGTH);
            // 运算完成后，把数据放入结果Tensor中，不要对TBuf的Tensor进行搬运或入队、出队、释放操作
            //AscendC::Cast(zLocal, tmpTensor2, AscendC::RoundMode::CAST_RINT, TOTAL_LENGTH);
            AscendC::Add(zLocal, xLocal, yLocal, this->processDataNum);
            outQueueZ.EnQue<TYPE_Z>(zLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueY.FreeTensor(yLocal);
        }
        __aicore__ inline void CopyOut(int32_t progress){
            AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();  
            AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
            outQueueZ.FreeTensor(zLocal);
        }

        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;
        AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueZ;
        //新增三个TBuf类型的成员变量
        //AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;     
        AscendC::GlobalTensor<TYPE_X> xGm;
        AscendC::GlobalTensor<TYPE_Y> yGm;
        AscendC::GlobalTensor<TYPE_Z> zGm;
        uint32_t coreDataNum;
        uint32_t tileNum;
        uint32_t tileDataNum;
        uint32_t tailDataNum;
        uint32_t processDataNum;
};


__global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TilingDataTemplate);
    GET_TILING_DATA_WITH_STRUCT(TilingDataTemplate, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.smallCoreDataNum, 
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
            tiling_data.tailBlockNum);
    op.Process();
}

/*
    用户自定义的kernel函数
    template <typename D_T_X, typename D_T_Y>
    __global__ __aicore__ void clamp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        AscendC::GlobalTensor<float> tmpGm; 
        tmpGm.SetGlobalBuffer((__gm__ float *)workspace); 
        // 初始化GlobalTensor，将workspace内存作为GlobalTensor的内存
    }
*/