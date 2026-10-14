#include "kernel_operator.h"
#include "clamp_tiling.h"
#include "tiling_key_clamp.h"
#include "kernel_operator_dump_tensor_intf_impl.h"
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t QUEUE_DEPTH = 1;

template <class dtypeX, class dtypeY>
class KernelClamp {
public:
    __aicore__ inline KernelClamp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, uint32_t totalLength, uint32_t tileNum, float min)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->min = min;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        tmpGm.SetGlobalBuffer((__gm__ float *)workspace);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueTmp, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        AscendC::printf("Dtype float, Core %ld executed %d times in total\n",  AscendC::GetBlockIdx(), loopCount);
    }

    __aicore__ inline void Process2()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute2(i);
            CopyOut(i);
        }
        AscendC::printf("Dtype int32, Core %ld executed %d times in total\n",  AscendC::GetBlockIdx(), loopCount);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
     
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = outQueueY.AllocTensor<dtypeY>();
        AscendC::Maxs(yLocal, xLocal, this->min, this->tileLength);
        outQueueY.EnQue<dtypeY>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void Compute2(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = outQueueY.AllocTensor<dtypeY>();
        // 这里假设我们把输入先转成float，然后暂存于workspace; ps: 实际上并不需要，这里只是为了演示如何使用workspace
        AscendC::LocalTensor<float> tmpLocal =  outQueueTmp.AllocTensor<float>();
        AscendC::Cast(tmpLocal, xLocal, AscendC::RoundMode::CAST_RINT, this->tileLength);
        AscendC::DataCopy(tmpGm, tmpLocal, this->tileLength);
        // 然后从workspace中取出缓存的float类型数据，再和float类型min计算
        int32_t eventIDMTE3_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        AscendC::DataCopy(tmpLocal, tmpGm, this->tileLength);
        int32_t eventIDMTE2_V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2_V);
        AscendC::Maxs(tmpLocal, tmpLocal, this->min, this->tileLength);
        AscendC::Cast(yLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, this->tileLength);
        outQueueTmp.FreeTensor(tmpLocal);
        outQueueY.EnQue<dtypeY>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeY> yLocal = outQueueY.DeQue<dtypeY>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueTmp;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<float> tmpGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float min;
};

template <typename D_T_X, typename D_T_Y>
 __global__ __aicore__ void clamp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClampTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClampTilingData, tilingData, tiling);
    KernelClamp<D_T_X, D_T_Y> op;
    op.Init(x, y ,workspace,  tilingData.totalLength, tilingData.tileNum,tilingData.min);

    if constexpr (std::is_same_v<D_T_X, float> && std::is_same_v<D_T_Y, float>) {
        op.Process();
    } else {
        op.Process2();
    }
}