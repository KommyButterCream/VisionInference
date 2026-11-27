#include "InferenceInstance.h" // for InferenceInstance, TensorRTLogger

#include <opencv2/core.hpp> // for cv::Mat, cv::Rect, cv::Size, cv::Vec3b
#include <opencv2/imgproc.hpp> // for cv::resize, cv::INTER_LINEAR

#include <cuda_runtime_api.h> // for cudaGetDeviceCount, cudaSetDevice, cudaGetErrorString

#include <cstdio>             // for std::fprintf, stderr
#include <fstream>            // for std::ifstream, std::ios, std::streamoff
#include <vector>             // for std::vector
#include <cstddef>            // for std::size_t
#include <exception>          // for std::exception
#include <limits>			  // for std::numeric_limits
#include <algorithm>		  // for std::fill, std::min
#include <cmath> // for std::isfinite

void TensorRTLogger::log(Severity severity, const char* message) noexcept
{
	// 경고보다 낮은 중요도의 메시지를 출력하지 않는다.
	if (severity > Severity::kWARNING)
	{
		return;
	}

	if (message == nullptr)
	{
		return;
	}

	std::fprintf(stderr, "[TensorRT] %s\n", message);
}


InferenceInstance::~InferenceInstance() noexcept
{
	Close();
}

bool InferenceInstance::Open(const std::filesystem::path& enginePath, std::int32_t deviceId)
{
	Close();

	try
	{
		cudaError_t cudaStatus = cudaSuccess;

		int deviceCount = 0;
		cudaStatus = cudaGetDeviceCount(&deviceCount);

		if (cudaStatus != cudaSuccess)
		{
			std::fprintf(stderr,
				"[VisionInference] cudaGetDeviceCount failed : %s\n",
				cudaGetErrorString(cudaStatus));

			return false;
		}

		if (deviceId < 0 || deviceId >= deviceCount)
		{
			return false;
		}

		// 1. 사용할 GPU 선택
		cudaStatus = cudaSetDevice(deviceId);

		if (cudaStatus != cudaSuccess)
		{
			std::fprintf(stderr,
				"[VisionInference] cudaSetDevice failed : %s\n",
				cudaGetErrorString(cudaStatus));

			return false;
		}

		// 2. 엔진 파일 열기
		std::ifstream file(enginePath, std::ios::binary | std::ios::ate);

		if (!file.is_open())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to open engine file.\n");

			return false;
		}

		// 3. 파일 크기 확인
		const std::streamoff fileSize = file.tellg();

		if (fileSize <= 0)
		{
			std::fprintf(
				stderr,
				"[VisionInference] Engine file is empty or size query failed.\n");

			return false;
		}

		std::vector<char> engineData(static_cast<std::size_t>(fileSize));

		// 4. 파일 처음으로 이동한 뒤 전체 읽기
		file.seekg(0, std::ios::beg);

		if (!file.read(engineData.data(), fileSize))
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to read engine file.\n");

			return false;
		}

		m_deviceId = deviceId;

		// 5. TensorRT runtime 생성
		m_runtime.reset(nvinfer1::createInferRuntime(m_logger));

		if (!m_runtime)
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to create TensorRT runtime.\n");

			Close();
			return false;
		}

		// 6. 엔진 역직렬화
		m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), engineData.size()));

		if (!m_engine)
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to deserialize TensorRT engine.\n");

			Close();
			return false;
		}
		
		PrintModelInfo();

		// 7. 추론 실행 Context 생성
		m_executionContext.reset(m_engine->createExecutionContext());

		if (!m_executionContext)
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to create execution context.\n");

			Close();
			return false;
		}

		if (!ReadModelInfo())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to read supported model information.\n");

			Close();
			return false;
		}

		if (!AllocateBuffers())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to allocate buffers.\n");

			Close();
			return false;
		}

		if (!PrepareExecution())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to prepare execution.\n");

			Close();
			return false;
		}

		return true;
	}
	catch (const std::exception& exception)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Open failed: %s\n",
			exception.what());

		Close();
		return false;
	}
	catch (...)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Open failed with an unknown exception.\n");

		Close();
		return false;
	}
}

void InferenceInstance::Close() noexcept
{
	if (m_deviceId >= 0)
	{
		if (!SelectDevice())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Device selection failed during Close; "
				"attempting resource cleanup.\n");
		}
	}

	if (m_stream != nullptr)
	{
		const cudaError_t status = cudaStreamSynchronize(m_stream);

		if (status != cudaSuccess)
		{
			std::fprintf(
				stderr,
				"[VisionInference] cudaStreamSynchronize failed: %s\n",
				cudaGetErrorString(status));
		}
	}

	m_executionContext.reset();

	ReleaseBuffers();

	if (m_stream != nullptr)
	{
		const cudaError_t status = cudaStreamDestroy(m_stream);

		if (status != cudaSuccess)
		{
			std::fprintf(
				stderr,
				"[VisionInference] cudaStreamDestroy failed: %s\n",
				cudaGetErrorString(status));
		}

		m_stream = nullptr;
	}

	m_engine.reset();
	m_runtime.reset();

	m_inputInfo = {};
	m_outputInfo = {};

	m_deviceId = -1;
}

bool InferenceInstance::IsOpen() const noexcept
{
	return m_deviceId >= 0 &&
		m_engine != nullptr &&
		m_executionContext != nullptr &&
		m_deviceInput != nullptr &&
		m_deviceOutput != nullptr &&
		m_stream != nullptr;
}

bool InferenceInstance::Infer(const VI_ImageView& image, VI_ClassificationResult& result) noexcept
{
	result = { -1, 0.0f };

	if (!IsOpen())
	{
		return false;
	}

	if (!Preprocess(image))
	{
		return false;
	}

	if (!RunTensor())
	{
		return false;
	}

	return Postprocess(result);
}

void InferenceInstance::PrintModelInfo() const noexcept
{
	if (!m_engine)
	{
		return;
	}

	const auto tensorCount = m_engine->getNbIOTensors();

	std::fprintf(
		stderr,
		"[VisionInference] I/O tensor count: %d\n",
		static_cast<int>(tensorCount));
	
	for (int index = 0; index < tensorCount; ++index)
	{
		const char* name = m_engine->getIOTensorName(index);

		if (name == nullptr)
		{
			continue;
		}

		const auto mode = m_engine->getTensorIOMode(name);
		const auto shape = m_engine->getTensorShape(name);
		const auto dataType = m_engine->getTensorDataType(name);

		const char* modeName = "UNKNOWN";

		switch (mode)
		{
		case nvinfer1::TensorIOMode::kINPUT:
			modeName = "INPUT";
			break;

		case nvinfer1::TensorIOMode::kOUTPUT:
			modeName = "OUTPUT";
			break;

		default:
			break;
		}

		const char* typeName = "OTHER";

		switch (dataType)
		{
		case nvinfer1::DataType::kFLOAT:
			typeName = "FP32";
			break;

		case nvinfer1::DataType::kHALF:
			typeName = "FP16";
			break;

		case nvinfer1::DataType::kINT32:
			typeName = "INT32";
			break;

		case nvinfer1::DataType::kINT8:
			typeName = "INT8";
			break;

		default:
			break;
		}
		
		// Tensor shape 출력
		std::fprintf(
			stderr,
			"[VisionInference] %s name=%s, dtype=%s(%d), shape=[",
			modeName,
			name,
			typeName,
			static_cast<int>(dataType));

		for (int dimension = 0; dimension < shape.nbDims; ++dimension)
		{
			if (dimension > 0)
			{
				std::fprintf(stderr, ", ");
			}

			std::fprintf(
				stderr,
				"%lld",
				static_cast<long long>(shape.d[dimension]));
		}

		std::fprintf(stderr, "]\n");
	}
}

bool InferenceInstance::SelectDevice() const noexcept
{
	if (m_deviceId < 0)
	{
		std::fprintf(
			stderr,
			"[VisionInference] No device has been assigned.\n");

		return false;
	}

	const cudaError_t status = cudaSetDevice(m_deviceId);

	if (status != cudaSuccess)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Failed to select device %d: %s\n",
			static_cast<int>(m_deviceId),
			cudaGetErrorString(status));

		return false;
	}

	return true;
}

bool InferenceInstance::ReadModelInfo()
{
	m_inputInfo = {};
	m_outputInfo = {};

	// 일단 입력 하나와 출력 하나만 지원
	if (!m_engine || m_engine->getNbIOTensors() != 2)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Expected one input and one output.\n");

		return false;
	}

	for (int index = 0; index < 2; ++index)
	{
		const char* name = m_engine->getIOTensorName(index);

		if (name == nullptr)
		{
			return false;
		}

		const auto mode = m_engine->getTensorIOMode(name);

		TensorInfo* info = nullptr;

		if (mode == nvinfer1::TensorIOMode::kINPUT)
		{
			info = &m_inputInfo;
		}
		else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
		{
			info = &m_outputInfo;
		}
		else
		{
			return false;
		}

		if (!info->name.empty())
		{
			return false;
		}

		info->name = name;
		info->shape = m_engine->getTensorShape(name);
		info->dataType = m_engine->getTensorDataType(name);

		// 일단 FP32 GPU 버퍼로 처리할 수 있는 형식만 지원
		if (info->dataType != nvinfer1::DataType::kFLOAT ||
			m_engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR ||
			m_engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
			m_engine->isShapeInferenceIO(name))
		{
			std::fprintf(
				stderr,
				"[VisionInference] Unsupported tensor format: %s\n",
				name);

			return false;
		}

		if (info->shape.nbDims <= 0)
		{
			return false;
		}

		std::size_t elementCount = 1;

		constexpr std::size_t maxSize = (std::numeric_limits<std::size_t>::max)();

		for (int dimension = 0; dimension < info->shape.nbDims; ++dimension)
		{
			const auto size = info->shape.d[dimension];

			// -1은 동적 차원이다. 첫 버전에서는 지원하지 않는다.
			if (size <= 0)
			{
				std::fprintf(
					stderr,
					"[VisionInference] Expected a fixed positive shape: %s\n",
					name);

				return false;
			}

			// Overflow 확인
			if (static_cast<std::uint64_t>(size) > maxSize / elementCount)
			{
				return false;
			}

			elementCount *= static_cast<std::size_t>(size);
		}

		// Overflow 확인
		if (elementCount > maxSize / sizeof(float))
		{
			return false;
		}

		info->elementCount = elementCount;
		info->byteSize = elementCount * sizeof(float);

		std::fprintf(
			stderr,
			"[VisionInference] %s: elements=%zu, bytes=%zu\n",
			name,
			info->elementCount,
			info->byteSize);
	}

	return !m_inputInfo.name.empty() && !m_outputInfo.name.empty();
}

bool InferenceInstance::AllocateBuffers()
{
	if (m_deviceInput != nullptr || m_deviceOutput != nullptr)
	{
		return false;
	}

	if (m_inputInfo.byteSize == 0 || m_outputInfo.byteSize == 0)
	{
		return false;
	}

	if (!SelectDevice())
	{
		return false;
	}

	m_hostInput.assign(m_inputInfo.elementCount, 0.0f);
	m_hostOutput.assign(m_outputInfo.elementCount, 0.0f);

	cudaError_t status = cudaMalloc(&m_deviceInput, m_inputInfo.byteSize);

	if (status != cudaSuccess)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Input cudaMalloc failed: %s\n",
			cudaGetErrorString(status));

		return false;
	}

	status = cudaMalloc(&m_deviceOutput, m_outputInfo.byteSize);

	if (status != cudaSuccess)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Output cudaMalloc failed: %s\n",
			cudaGetErrorString(status));

		return false;
	}

	return true;
}

void InferenceInstance::ReleaseBuffers() noexcept
{
	if (m_deviceOutput != nullptr)
	{
		const cudaError_t status = cudaFree(m_deviceOutput);

		if (status != cudaSuccess)
		{
			std::fprintf(
				stderr,
				"[VisionInference] Output cudaFree failed: %s\n",
				cudaGetErrorString(status));
		}

		m_deviceOutput = nullptr;
	}

	if (m_deviceInput != nullptr)
	{
		const cudaError_t status = cudaFree(m_deviceInput);

		if (status != cudaSuccess)
		{
			std::fprintf(
				stderr,
				"[VisionInference] Input cudaFree failed: %s\n",
				cudaGetErrorString(status));
		}

		m_deviceInput = nullptr;
	}

	std::vector<float>().swap(m_hostOutput);
	std::vector<float>().swap(m_hostInput);
}

bool InferenceInstance::PrepareExecution() noexcept
{
	if (!m_executionContext ||
		m_deviceInput == nullptr ||
		m_deviceOutput == nullptr ||
		m_stream != nullptr)
	{
		return false;
	}

	if (!SelectDevice())
	{
		return false;
	}

	// 1. GPU 작업을 제출할 stream 생성
	const cudaError_t status = cudaStreamCreate(&m_stream);

	if (status != cudaSuccess)
	{
		std::fprintf(
			stderr,
			"[VisionInference] cudaStreamCreate failed: %s\n",
			cudaGetErrorString(status));

		return false;
	}
	
	// 2. 입력 tensor 에 GPU 입력 버퍼 주소 등록
	if (!m_executionContext->setTensorAddress(m_inputInfo.name.c_str(), m_deviceInput))
	{
		std::fprintf(
			stderr,
			"[VisionInference] Failed to set input tensor address.\n");

		return false;
	}

	// 3. 출력 tensor 에 GPU 출력 버퍼 주소 등록
	if (!m_executionContext->setTensorAddress(m_outputInfo.name.c_str(), m_deviceOutput))
	{
		std::fprintf(
			stderr,
			"[VisionInference] Failed to set output tensor address.\n");

		return false;
	}

	return true;
}

bool InferenceInstance::Preprocess(const VI_ImageView& image) noexcept
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

		// batch = 1, RGB 3 채널, 정사각형 NCHW 입력만 지원
		const auto& shape = m_inputInfo.shape;

		if (shape.nbDims != 4 ||
			shape.d[0] != 1 ||
			shape.d[1] != 3 ||
			shape.d[2] <= 0 ||
			shape.d[2] != shape.d[3] ||
			shape.d[2] > (std::numeric_limits<int>::max)())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Expected input shape [1, 3, S, S].\n");

			return false;
		}

		const int targetSize = static_cast<int>(shape.d[2]);

		// Open에서 계산하고 할당한 입력 버퍼를 사용한다.
		if (m_hostInput.size() != m_inputInfo.elementCount || m_hostInput.empty())
		{
			return false;
		}

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

		const std::size_t planeSize = m_inputInfo.elementCount / 3;

		float* const red = m_hostInput.data();
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

bool InferenceInstance::RunTensor() noexcept
{
	if (!IsOpen())
	{
		std::fprintf(
			stderr,
			"[VisionInference] Model is not ready.\n");

		return false;
	}

	if (!SelectDevice())
	{
		return false;
	}

	// 1. CPU 입력 -> GPU 입력
	cudaError_t status = cudaMemcpyAsync(
		m_deviceInput,
		m_hostInput.data(),
		m_inputInfo.byteSize,
		cudaMemcpyHostToDevice,
		m_stream);


	if (status != cudaSuccess)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Input copy failed: %s\n",
			cudaGetErrorString(status));

		Close();
		return false;
	}

	// 2. 추론 실행 요청
	if (!m_executionContext->enqueueV3(m_stream))
	{
		std::fprintf(
			stderr,
			"[VisionInference] TensorRT enqueueV3 failed.\n");

		Close();
		return false;
	}

	// 3. GPU 출력 -> CPU 출력
	status = cudaMemcpyAsync(
		m_hostOutput.data(),
		m_deviceOutput,
		m_outputInfo.byteSize,
		cudaMemcpyDeviceToHost,
		m_stream);

	if (status != cudaSuccess)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Output copy failed: %s\n",
			cudaGetErrorString(status));

		Close();
		return false;
	}

	// 4. 복사 완료 대기
	status = cudaStreamSynchronize(m_stream);

	if (status != cudaSuccess)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Inference synchronization failed: %s\n",
			cudaGetErrorString(status));

		Close();
		return false;
	}

	return true;
}

bool InferenceInstance::Postprocess(VI_ClassificationResult& result) const noexcept
{
	result = { -1, 0.0f };

	const auto& shape = m_outputInfo.shape;

	// 현재는 이미지 한 장의 클래스별 점수 [1, C]만 지원한다.
	if (shape.nbDims != 2 ||
		shape.d[0] != 1 ||
		shape.d[1] <= 0 ||
		shape.d[1] > (std::numeric_limits<int32_t>::max)())
	{
		std::fprintf(
			stderr,
			"[VisionInference] Expected output shape [1, C].\n");

		return false;
	}

	const std::size_t classCount = static_cast<std::size_t>(shape.d[1]);

	if (m_hostOutput.size() != classCount)
	{
		return false;
	}

	std::size_t bestIndex = 0;
	float bestScore = (std::numeric_limits<float>::lowest)();

	for (std::size_t index = 0; index < classCount; ++index)
	{
		const float score = m_hostOutput[index];

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
