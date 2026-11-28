#pragma once

#include "InferenceTypes.h" // for InferenceStatus, TensorCatalog
#include "InferenceTaskTypes.h" // for InferenceTaskKind, ModelOptions

struct ExecutionSlot;
class ResultBuilder;

class InferenceInterface
{
public:
	virtual ~InferenceInterface() = default;

	// 분류·검출·분할 중 어떤 Task인지 반환
	virtual InferenceTaskKind Kind() const noexcept = 0;

    // 모델 정보와 고정 옵션을 검증한다.
    virtual InferenceStatus Configure(const TensorCatalog& tensors, const ModelOptions& options) = 0;

    // 슬롯의 이미지를 읽어 입력 텐서를 만든다.
    virtual InferenceStatus Preprocess(ExecutionSlot& slot) const = 0;

    // 필요한 GPU 후처리와 D2H 복사를 제출한다.
    // 이 함수 안에서 완료를 기다리지는 않는다.
    virtual InferenceStatus EnqueuePostprocess(ExecutionSlot& slot) const = 0;

    // 실행기가 완료를 확인한 뒤 최종 결과를 만든다.
    virtual InferenceStatus BuildResult(const ExecutionSlot& slot, ResultBuilder& result) const = 0;
};