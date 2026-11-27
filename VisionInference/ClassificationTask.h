#pragma once

#include "VisionInference.h" // for VI_ImageView, VI_ClassificationResult
#include "InferenceTypes.h" // for TensorInfo

#include <cstddef> // for std::size_t
#include <span> // for std::span

class ClassificationTask final
{
public:
	bool Configure(
		const TensorInfo& inputInfo,
		const TensorInfo& outputInfo) noexcept;

	void Reset() noexcept;

	bool Preprocess(
		const VI_ImageView& image, 
		std::span<float> inputTensor) const noexcept;

	bool Postprocess(
		std::span<const float> outputTensor,
		VI_ClassificationResult& result) const noexcept;

private:
	int m_targetSize = 0;
	std::size_t m_inputElementCount = 0;
	std::size_t m_classCount = 0;
};