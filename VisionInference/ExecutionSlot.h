#pragma once

#include "Buffers.h" // for TensorBuffer, CudaStream

#include <NvInferRuntime.h> // for nvinfer1::IExecutionContext

#include "VisionInference.h" // for VI_ImageView
#include "InferenceTaskTypes.h" // for ImageTransform

#include <memory> // for std::unique_ptr

// 인퍼런스 한 번에 사용하는 실행 자원을 묶는다.
// 현재는 입력 하나, 출력 하나, 동기 실행만 지원
struct ExecutionSlot final
{
    ExecutionSlot() noexcept = default;
    ~ExecutionSlot() noexcept = default;

    ExecutionSlot(const ExecutionSlot&) = delete;
    ExecutionSlot& operator=(const ExecutionSlot&) = delete;

    ExecutionSlot(ExecutionSlot&&) = delete;
    ExecutionSlot& operator=(ExecutionSlot&&) = delete;

    // 현재 동기 호출 동안만 빌려 사용하는 이미지
    // 비동기 요청을 구현할 때는 입력 풀의 메모리를 연결
    VI_ImageView image{};

    // 현재 이미지의 전처리 결과 정보
    ImageTransform transform{};

    TensorBuffer input;
    TensorBuffer output;
    CudaStream stream;

    // 슬롯 내부에서는 context가 stream과 버퍼보다 먼저 파괴된다.
    // engine과의 수명 순서는 InferenceInstance의 멤버 배치와 Close가 보장한다.
    std::unique_ptr<nvinfer1::IExecutionContext> context;
};