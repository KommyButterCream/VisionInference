#include "ClassificationTask.h" // for ClassificationTask

#include <opencv2/core.hpp> // for cv::Mat, cv::Rect, cv::Size, cv::Vec3b
#include <opencv2/imgproc.hpp> // for cv::resize, cv::INTER_LINEAR

#include <algorithm> // for std::min
#include <cmath> // for std::isfinite
#include <cstddef> // for std::size_t
#include <cstdint> // for std::int64_t
#include <cstdio> // for std::fprintf, stderr
#include <exception> // for std::exception
#include <limits> // for std::numeric_limits



bool ClassificationTask::Configure(
	const TensorInfo& inputInfo,
	const TensorInfo& outputInfo) noexcept
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
		inputShape.d[0] != 1 ||	  // 한번에 처리할 이미지가 1장인가?
		inputShape.d[1] != 3 ||	  // 채널 수
		inputShape.d[2] <= 0 ||	  // 높이
		inputShape.d[2] != inputShape.d[3] || // 너비
		inputShape.d[2] > (std::numeric_limits<int>::max)())
	{
		std::fprintf(
			stderr,
			"[VisionInference] Expected input shape [1, 3, S, S].\n");

		return false;
	}

	if (outputShape.nbDims != 2 ||
		outputShape.d[0] != 1 ||	// 한 번에 검사할이미지 수량
		outputShape.d[1] <= 0 ||	// 이미지 한장에 대한 클래스 점수 개수
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
	// 채널 하나 : 224 × 224 = 50, 176개
	// 전체 입력 : 50, 176 × 3 = 150, 528개
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
}

bool ClassificationTask::Preprocess(
    const VI_ImageView& image,
    std::span<float> inputTensor) const noexcept
{
	try
	{
		if (image.data == nullptr ||
			image.width <= 0 ||
			image.height <= 0 ||
			image.pixelFormat != VI_PIXEL_FORMAT_BGR8 ||
			image.strideBytes / 3 < static_cast<std::size_t>(image.width))
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

		// Python CenterCrop 의 Round(diff / 2) 와 같은 짝수 반올림.
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

bool ClassificationTask::Postprocess(
	std::span<const float> outputTensor,
	VI_ClassificationResult& result) const noexcept
{
	result = { -1, 0.0f };

	if (m_classCount == 0 || outputTensor.size() != m_classCount)
	{
		return false;
	}

	const std::size_t classCount = m_classCount;

	if (outputTensor.size() != classCount)
	{
		return false;
	}

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