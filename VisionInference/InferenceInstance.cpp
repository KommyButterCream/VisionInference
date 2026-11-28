#include "InferenceInstance.h" // for InferenceInstance, TensorRTLogger
#include "ImageValidation.h"

#include <cuda_runtime_api.h> // for cudaGetDeviceCount, cudaSetDevice, cudaGetErrorString

#include <cstdio>             // for std::fprintf, stderr
#include <fstream>            // for std::ifstream, std::ios, std::streamoff
#include <vector>             // for std::vector
#include <cstddef>            // for std::size_t
#include <exception>          // for std::exception
#include <limits>			  // for std::numeric_limits
#include <span> // for std::span
#include <new> // for std::bad_alloc

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

InferenceStatus InferenceInstance::Open(const std::filesystem::path& enginePath, std::int32_t deviceId) noexcept
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

			return InferenceStatus::ResourcePreparationFailed;
		}

		if (deviceId < 0 || deviceId >= deviceCount)
		{
			return InferenceStatus::InvalidArgument;
		}

		// 1. 사용할 GPU 선택
		cudaStatus = cudaSetDevice(deviceId);

		if (cudaStatus != cudaSuccess)
		{
			std::fprintf(stderr,
				"[VisionInference] cudaSetDevice failed : %s\n",
				cudaGetErrorString(cudaStatus));

			return InferenceStatus::ResourcePreparationFailed;
		}

		// 2. 엔진 파일 열기
		std::ifstream file(enginePath, std::ios::binary | std::ios::ate);

		if (!file.is_open())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to open engine file.\n");

			return InferenceStatus::FileReadFailed;
		}

		// 3. 파일 크기 확인
		const std::streamoff fileSize = file.tellg();

		if (fileSize <= 0)
		{
			std::fprintf(
				stderr,
				"[VisionInference] Engine file is empty or size query failed.\n");

			return InferenceStatus::FileReadFailed;
		}

		std::vector<char> engineData(static_cast<std::size_t>(fileSize));

		// 4. 파일 처음으로 이동한 뒤 전체 읽기
		file.seekg(0, std::ios::beg);

		if (!file.read(engineData.data(), fileSize))
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to read engine file.\n");

			return InferenceStatus::FileReadFailed;
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
			return InferenceStatus::ResourcePreparationFailed;
		}

		// 6. 엔진 역직렬화
		m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), engineData.size()));

		if (!m_engine)
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to deserialize TensorRT engine.\n");

			Close();
			return InferenceStatus::ModelOpenFailed;
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
			return InferenceStatus::ResourcePreparationFailed;
		}

		if (!ReadModelInfo())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to read supported model information.\n");

			Close();
			return InferenceStatus::UnsupportedModel;
		}

		if (!m_classificationTask.Configure(m_inputInfo, m_outputInfo))
		{
			std::fprintf(
				stderr,
				"[VisionInference] Unsupported classification model contract.\n");

			Close();
			return InferenceStatus::UnsupportedModel;
		}

		if (!AllocateBuffers())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to allocate buffers.\n");

			Close();
			return InferenceStatus::ResourcePreparationFailed;
		}

		if (!PrepareExecution())
		{
			std::fprintf(
				stderr,
				"[VisionInference] Failed to prepare execution.\n");

			Close();
			return InferenceStatus::ResourcePreparationFailed;
		}

		return InferenceStatus::Ok;
	}
	catch (const std::bad_alloc& exception)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Open allocation failed: %s\n",
			exception.what());

		Close();
		return InferenceStatus::OutOfMemory;
	}
	catch (const std::exception& exception)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Open failed: %s\n",
			exception.what());

		Close();
		return InferenceStatus::InternalError;
	}
	catch (...)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Open failed with an unknown exception.\n");

		Close();
		return InferenceStatus::InternalError;
	}
}

void InferenceInstance::Close() noexcept
{
	if (m_deviceId >= 0 && !SelectDevice())
	{
		std::fprintf(
			stderr,
			"[VisionInference] Device selection failed during Close; "
			"attempting resource cleanup.\n");
	}

	// Context와 버퍼를 해제하기 전에 실행 완료를 기다린다.
	if (!m_stream.Synchronize())
	{
		std::fprintf(
			stderr,
			"[VisionInference] Stream synchronization failed during Close; "
			"attempting resource cleanup.\n");
	}

	m_executionContext.reset();

	ReleaseBuffers();

	m_stream.Reset();

	m_engine.reset();
	m_runtime.reset();

	m_classificationTask.Reset();

	m_inputInfo = {};
	m_outputInfo = {};

	m_deviceId = -1;
}

bool InferenceInstance::IsOpen() const noexcept
{
	return m_deviceId >= 0 &&
		m_engine != nullptr &&
		m_executionContext != nullptr &&
		m_deviceInput.Data() != nullptr &&
		m_deviceOutput.Data() != nullptr &&
		m_stream.Get() != nullptr;
}

InferenceStatus InferenceInstance::Infer(const VI_ImageView& image, VI_ClassificationResult& result) noexcept
{
	result = { -1, 0.0f };

	if (!IsValidImageView(image))
	{
		return InferenceStatus::InvalidArgument;
	}

	if (!IsOpen())
	{
		return InferenceStatus::ModelNotOpen;
	}

	const std::span<float> inputTensor(m_hostInput.data(), m_hostInput.size());

	if (!m_classificationTask.Preprocess(image, inputTensor))
	{
		return InferenceStatus::PreprocessFailed;
	}

	if (!RunTensor())
	{
		return InferenceStatus::ExecutionFailed;
	}

	const std::span<const float> outputTensor(m_hostOutput.data(), m_hostOutput.size());

	if (!m_classificationTask.Postprocess(outputTensor, result))
	{
		return InferenceStatus::PostprocessFailed;
	}

	return InferenceStatus::Ok;
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
	if (m_deviceInput.Data() != nullptr || m_deviceOutput.Data() != nullptr)
	{
		return false;
	}

	if (m_inputInfo.byteSize == 0 || m_outputInfo.byteSize == 0)
	{
		return false;
	}

	m_hostInput.assign(m_inputInfo.elementCount, 0.0f);
	m_hostOutput.assign(m_outputInfo.elementCount, 0.0f);

	if (!m_deviceInput.Allocate(m_inputInfo.byteSize, m_deviceId))
	{
		return false;
	}

	if (!m_deviceOutput.Allocate(m_outputInfo.byteSize, m_deviceId))
	{
		return false;
	}

	return true;
}

void InferenceInstance::ReleaseBuffers() noexcept
{
	m_deviceOutput.Reset();
	m_deviceInput.Reset();

	std::vector<float>().swap(m_hostOutput);
	std::vector<float>().swap(m_hostInput);
}

bool InferenceInstance::PrepareExecution() noexcept
{
	if (!m_executionContext ||
		m_deviceInput.Data() == nullptr ||
		m_deviceOutput.Data() == nullptr ||
		m_stream.Get() != nullptr)
	{
		return false;
	}

	if (!m_stream.Create(m_deviceId))
	{
		return false;
	}

	// 2. 입력 tensor 에 GPU 입력 버퍼 주소 등록
	if (!m_executionContext->setTensorAddress(m_inputInfo.name.c_str(), m_deviceInput.Data()))
	{
		std::fprintf(
			stderr,
			"[VisionInference] Failed to set input tensor address.\n");

		return false;
	}

	// 3. 출력 tensor 에 GPU 출력 버퍼 주소 등록
	if (!m_executionContext->setTensorAddress(m_outputInfo.name.c_str(), m_deviceOutput.Data()))
	{
		std::fprintf(
			stderr,
			"[VisionInference] Failed to set output tensor address.\n");

		return false;
	}

	return true;
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
		m_deviceInput.Data(),
		m_hostInput.data(),
		m_inputInfo.byteSize,
		cudaMemcpyHostToDevice,
		m_stream.Get());

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
	if (!m_executionContext->enqueueV3(m_stream.Get()))
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
		m_deviceOutput.Data(),
		m_outputInfo.byteSize,
		cudaMemcpyDeviceToHost,
		m_stream.Get());

	if (status != cudaSuccess)
	{
		std::fprintf(
			stderr,
			"[VisionInference] Output copy failed: %s\n",
			cudaGetErrorString(status));

		Close();
		return false;
	}

	if (!m_stream.Synchronize())
	{
		Close();
		return false;
	}

	return true;
}
