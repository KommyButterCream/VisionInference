#include "InferenceInstance.h" // for InferenceInstance, TensorRTLogger

#include <cuda_runtime_api.h> // for cudaSetDevice, cudaGetErrorString
#include <cstdio>             // for std::fprintf, stderr
#include <fstream>            // for std::ifstream, std::ios, std::streamoff
#include <vector>             // for std::vector
#include <cstddef>            // for std::size_t
#include <exception>          // for std::exception

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
	m_engine.reset();

	m_runtime.reset();
}

bool InferenceInstance::IsOpen() const noexcept
{
	return m_engine != nullptr;
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
