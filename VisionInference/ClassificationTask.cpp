#include "ClassificationTask.h" // for ClassificationTask

#include "ImageValidation.h" // for IsValidImageView
#include "InferenceResult.h" // for ResultBuilder, ClassificationScore

#include <opencv2/core.hpp> // for cv::Mat, cv::Rect, cv::Size, cv::Vec3b
#include <opencv2/imgproc.hpp> // for cv::resize, cv::INTER_LINEAR

#include "ExecutionSlot.h" // for ExecutionSlot, TensorBuffer
#include <cuda_runtime_api.h> // for cudaMemcpyAsync, cudaGetErrorString

#include <algorithm> // for std::min
#include <cmath> // for std::isfinite
#include <cstddef> // for std::size_t
#include <cstdint> // for std::int64_t
#include <cstdio> // for std::fprintf, stderr
#include <exception> // for std::exception
#include <limits> // for std::numeric_limits
#include <new> // for std::bad_alloc

namespace
{
	bool VerifyTensor(const TensorBuffer& buffer, const TensorInfo& expected) noexcept
	{
		const TensorInfo* actual = buffer.Info();

		// TensorBuffer가 두 메모리를 준비한 상태인지 확인한다.
		if (actual == nullptr ||
			buffer.DeviceData() == nullptr ||
			buffer.Host().size() != expected.elementCount ||
			buffer.SizeBytes() != expected.byteSize)
		{
			return false;
		}

		// 다른 모델의 텐서가 실수로 연결되지 않았는지 확인한다.
		if (actual->name != expected.name ||
			actual->dataType != expected.dataType ||
			actual->elementCount != expected.elementCount ||
			actual->byteSize != expected.byteSize ||
			actual->shape.nbDims != expected.shape.nbDims ||
			actual->shape.nbDims <= 0 ||
			actual->shape.nbDims > nvinfer1::Dims::MAX_DIMS)
		{
			return false;
		}

		for (int index = 0; index < actual->shape.nbDims; ++index)
		{
			if (actual->shape.d[index] != expected.shape.d[index])
			{
				return false;
			}
		}

		return true;
	}
}

bool ClassificationTask::ConfigureTensors(const TensorInfo& inputInfo, const TensorInfo& outputInfo) noexcept
{
	Reset();

	if (inputInfo.dataType != nvinfer1::DataType::kFLOAT ||
		outputInfo.dataType != nvinfer1::DataType::kFLOAT)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Classification requires FP32 I/O.\n");

		return false;
	}

	const auto& inputShape = inputInfo.shape;
	const auto& outputShape = outputInfo.shape;

	if (inputShape.nbDims != 4 || // 차원 수량
		inputShape.d[0] != 1 ||	  // 한 번에 처리할 이미지가 1장인가?
		inputShape.d[1] != 3 ||	  // 채널 수
		inputShape.d[2] <= 0 ||	  // 높이
		inputShape.d[2] != inputShape.d[3] || // 높이와 너비 비교
		inputShape.d[2] > (std::numeric_limits<int>::max)())
	{
		std::fprintf(
			stderr,
			"[VisionInference] Expected input shape [1, 3, S, S].\n");

		return false;
	}

	if (outputShape.nbDims != 2 ||
		outputShape.d[0] != 1 ||	// 한 번에 검사할 이미지 수량
		outputShape.d[1] <= 0 ||	// 이미지 한 장에 대한 클래스 점수 개수
		outputShape.d[1] > (std::numeric_limits<int32_t>::max)())
	{
		std::fprintf(
			stderr,
			"[VisionInference] Expected output shape [1, C].\n");

		return false;
	}

	const std::size_t side = static_cast<std::size_t>(inputShape.d[2]);
	const std::size_t classCount = static_cast<std::size_t>(outputShape.d[1]);
	constexpr std::size_t maxSize = (std::numeric_limits<std::size_t>::max)();

	if (side > maxSize / side)
	{
		return false;
	}
	// 채널 하나 : 224 × 224 = 50,176개
	// 전체 입력 : 50,176 × 3 = 150,528개
	// [R 50,176개][G 50,176개][B 50,176개]
	// 채널 하나의 원소 수(R,G,B)
	const std::size_t planeSize = side * side;

	if (planeSize > maxSize / 3)
	{
		return false;
	}

	const std::size_t inputCount = planeSize * 3;

	if (inputCount > maxSize / sizeof(float) || classCount > maxSize / sizeof(float))
	{
		return false;
	}

	// shape와 버퍼 크기 정보가 서로 일치하는지도 확인한다.
	if (inputInfo.elementCount != inputCount ||
		inputInfo.byteSize != inputCount * sizeof(float) ||
		outputInfo.elementCount != classCount ||
		outputInfo.byteSize != classCount * sizeof(float))
	{
		std::fprintf(
			stderr,
			"[VisionInference] Inconsistent classification tensor sizes.\n");

		return false;
	}

	// 모든 검사가 성공한 뒤 설정을 확정한다.
	m_targetSize = static_cast<int>(side);
	m_inputElementCount = inputCount;
	m_classCount = classCount;

	return true;
}

void ClassificationTask::Reset() noexcept
{
	m_targetSize = 0;
	m_inputElementCount = 0;
	m_classCount = 0;
	m_configuredInput = {};
	m_configuredOutput = {};
	m_options = {};
}

bool ClassificationTask::PreprocessImage(const VI_ImageView& image, std::span<float> inputTensor, ImageTransform* transform) const noexcept
{
    // 실패 시 이전 이미지의 변환 정보가 남지 않도록 초기화한다.
    if (transform != nullptr)
    {
        *transform = {};
    }

	try
	{
		if (!IsValidImageView(image))
		{
			return false;
		}

		if (m_targetSize <= 0 || inputTensor.size() != m_inputElementCount)
		{
			return false;
		}

		const int targetSize = m_targetSize;

		const cv::Mat cvSource(
			image.height,
			image.width,
			CV_8UC3,
			const_cast<uint8_t*>(image.data),
			image.strideBytes);

		// 짧은 변을 targetSize로 맞추고 종횡비를 유지한다.
		const int shortSide = (std::min)(image.width, image.height);

		const std::int64_t resizedWidth64 = static_cast<std::int64_t>(image.width) * targetSize / shortSide;
		const std::int64_t resizedHeight64 = static_cast<std::int64_t>(image.height) * targetSize / shortSide;

		if (resizedWidth64 > (std::numeric_limits<int>::max)() ||
			resizedHeight64 > (std::numeric_limits<int>::max)())
		{
			return false;
		}

		const int resizedWidth = static_cast<int>(resizedWidth64);
		const int resizedHeight = static_cast<int>(resizedHeight64);

		cv::Mat cvResized;

		cv::resize(cvSource, cvResized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0, cv::INTER_LINEAR);

		// Python CenterCrop의 round(diff / 2) 와 같은 짝수 반올림.
		const auto centerOffset = [](int difference) noexcept
			{
				const int half = difference / 2;
				return half + ((difference % 2 != 0 && half % 2 != 0) ? 1 : 0);
			};

		const int left = centerOffset(resizedWidth - targetSize);
		const int top = centerOffset(resizedHeight - targetSize);

		const cv::Mat cvCropped = cvResized(cv::Rect(left, top, targetSize, targetSize));

		const std::size_t planeSize = m_inputElementCount / 3;

		float* const red = inputTensor.data();
		float* const green = red + planeSize;
		float* const blue = green + planeSize;

		constexpr float scale = 1.0f / 255.0f;

		for (int y = 0; y < targetSize; ++y)
		{
			const cv::Vec3b* row = cvCropped.ptr<cv::Vec3b>(y);

			for (int x = 0; x < targetSize; ++x)
			{
				const std::size_t index = static_cast<std::size_t>(y) * targetSize + x;

				// OpenCV의 BGR을 모델의 RGB 순서로 바꾸면서
				// uint8 -> float, HWC -> CHW, 0~255 -> 0~1 변환.
				red[index] = row[x][2] * scale;
				green[index] = row[x][1] * scale;
				blue[index] = row[x][0] * scale;
			}
		}

        // 실제 전처리에 사용한 크기와 crop 위치를 후처리에 전달한다.
        if (transform != nullptr)
        {
            transform->originalWidth = image.width;
            transform->originalHeight = image.height;
            transform->resizedWidth = resizedWidth;
            transform->resizedHeight = resizedHeight;
            transform->cropLeft = left;
            transform->cropTop = top;
            transform->targetWidth = targetSize;
            transform->targetHeight = targetSize;
        }

		return true;

	}
	catch (const std::exception& exception)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Preprocess failed: %s\n",
			exception.what());

		return false;
	}
	catch (...)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Preprocess failed: unknown exception.\n");

		return false;
	}
}

bool ClassificationTask::SelectTop1(std::span<const float> outputTensor, VI_ClassificationResult& result) const noexcept
{
	result = { -1, 0.0f };

	if (m_classCount == 0 || outputTensor.size() != m_classCount)
	{
		return false;
	}

	const std::size_t classCount = m_classCount;

	std::size_t bestIndex = 0;
	float bestScore = (std::numeric_limits<float>::lowest)();

	for (std::size_t index = 0; index < classCount; ++index)
	{
		const float score = outputTensor[index];

		if (!std::isfinite(score))
		{
			std::fprintf(
				stderr,
				"[VisionInference] Non-finite output at index %zu.\n",
				index);

			return false;
		}

		if (score > bestScore)
		{
			bestScore = score;
			bestIndex = index;
		}
	}

	result.classId = static_cast<int32_t>(bestIndex);
	result.score = bestScore;

	return true;
}

InferenceTaskKind ClassificationTask::Kind() const noexcept
{
	return InferenceTaskKind::Classification;
}

InferenceStatus ClassificationTask::Configure(const TensorCatalog& tensors, const ModelOptions& options)
{
	Reset();

	if (options.taskKind != Kind() ||
		tensors.inputs.size() != 1 ||
		tensors.outputs.size() != 1)
	{
		return InferenceStatus::UnsupportedModel;
	}

	// 아직 구현하지 않은 backend를 조용히 CPU로 바꾸지 않는다.
	if (options.preprocessBackend != ProcessingBackend::Cpu ||
		options.postprocessBackend != ProcessingBackend::Cpu)
	{
		return InferenceStatus::InvalidArgument;
	}

	try
	{
		const auto& input = tensors.inputs[0];
		const auto& output = tensors.outputs[0];

		if (input.name.empty() || output.name.empty() || input.name == output.name)
		{
			return InferenceStatus::UnsupportedModel;
		}

        // 기존 검증으로 FP32, shape, 원소 수를 확인하고 내부 크기도 설정한다.
        if (!ConfigureTensors(input, output))
        {
            return InferenceStatus::UnsupportedModel;
        }

		m_configuredInput = input;
		m_configuredOutput = output;
		m_options = options;

		return InferenceStatus::Ok;
	}
	catch (const std::bad_alloc&)
	{
		Reset();
		return InferenceStatus::OutOfMemory;
	}
	catch (...)
	{
		Reset();
		return InferenceStatus::InternalError;
	}
}

InferenceStatus ClassificationTask::Preprocess(ExecutionSlot& slot) const
{
    // 이전 요청의 출력 메타데이터를 지우고 새 요청을 검증한다.
	slot.transform = {};

	if (m_configuredInput.name.empty())
	{
		return InferenceStatus::ModelNotOpen;
	}

    // 현재 구현은 CPU 전처리, 이미지 1장, 입력 텐서 1개만 지원한다.
	if (m_options.preprocessBackend != ProcessingBackend::Cpu ||
		!VerifyTensor(slot.input, m_configuredInput))
	{
		return InferenceStatus::InvalidArgument;
	}

	// 이미지와 작업 버퍼를 슬롯에서 바로 가져온다.
	if (!PreprocessImage(slot.image, slot.input.Host(), &slot.transform))
	{
		return InferenceStatus::PreprocessFailed;
	}

	return InferenceStatus::Ok;
}

InferenceStatus ClassificationTask::EnqueuePostprocess(ExecutionSlot& slot) const
{
	if (m_configuredOutput.name.empty())
	{
		return InferenceStatus::ModelNotOpen;
	}

	if (m_options.postprocessBackend != ProcessingBackend::Cpu ||
		slot.stream.Get() == nullptr ||
		!VerifyTensor(slot.output, m_configuredOutput))
	{
		return InferenceStatus::InvalidArgument;
	}

	// CPU 후처리이므로 원시 출력 텐서를 CPU로 가져온다.
	// 같은 stream 에 먼저 제출한 TensorRT 작업 다음에 실행된다.
	const cudaError_t status = cudaMemcpyAsync(
		slot.output.Host().data(),
		slot.output.DeviceData(),
		slot.output.SizeBytes(),
		cudaMemcpyDeviceToHost,
		slot.stream.Get());

	if (status != cudaSuccess)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Output copy failed: %s\n",
			cudaGetErrorString(status));

		return InferenceStatus::ExecutionFailed;
	}

	// 복사 제출 성공일 뿐, 완료됐다는 의미는 아니다.
	return InferenceStatus::Ok;
}

InferenceStatus ClassificationTask::BuildResult(const ExecutionSlot& slot, ResultBuilder& result) const
{
	result.Reset();

	if (m_configuredOutput.name.empty())
	{
		return InferenceStatus::ModelNotOpen;
	}

	if (!VerifyTensor(slot.output, m_configuredOutput))
	{
		return InferenceStatus::InvalidArgument;
	}

	// D2H 완료를 확인한 뒤 호출해야 한다.
	VI_ClassificationResult top1{};

	if (!SelectTop1(slot.output.Host(), top1))
	{
		return InferenceStatus::PostprocessFailed;
	}

	try
	{
		const ClassificationScore score { top1.classId, top1.score };

		// 작업용 텐서와 독립된 결과 배열에 값을 복사한다.
		result.AddClassificationResult(std::span<const ClassificationScore>(&score, 1));

		return InferenceStatus::Ok;
	}
	catch (const std::bad_alloc&)
	{
		result.Reset();
		return InferenceStatus::OutOfMemory;
	}
	catch (...)
	{
		result.Reset();
		return InferenceStatus::PostprocessFailed;
	}
}
