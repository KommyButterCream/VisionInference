#pragma once

#include "VisionInference.h" // for VI_ImageView, VI_ClassificationResult
#include "InferenceInterface.h" // for InferenceInterface and task contracts

#include <cstddef> // for std::size_t
#include <span> // for std::span

class ClassificationTask final : public InferenceInterface
{
public:
    InferenceTaskKind Kind() const noexcept override;

    InferenceStatus Configure(const TensorCatalog& tensors, const ModelOptions& options) override;

    InferenceStatus Preprocess(ExecutionSlot& slot) const override;

    InferenceStatus EnqueuePostprocess(ExecutionSlot& slot) const override;

    InferenceStatus BuildResult(const ExecutionSlot& slot, ResultBuilder& result) const override;

    void Reset() noexcept;

private:
    // 기존에 구현한 모델 검증과 분류 알고리즘이다.
    bool ConfigureTensors(const TensorInfo& inputInfo, const TensorInfo& outputInfo) noexcept;

    bool PreprocessImage(const VI_ImageView& image, std::span<float> inputTensor, ImageTransform* transform) const noexcept;

    bool SelectTop1(std::span<const float> outputTensor, VI_ClassificationResult& result) const noexcept;

private:
	int m_targetSize = 0;
	std::size_t m_inputElementCount = 0;
	std::size_t m_classCount = 0;

    TensorInfo m_configuredInput;
    TensorInfo m_configuredOutput;
    ModelOptions m_options;
};
