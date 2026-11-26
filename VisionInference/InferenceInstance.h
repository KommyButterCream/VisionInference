#pragma once

#include <NvInferRuntime.h> // for nvinfer1::ILogger, IRuntime, ICudaEngine, IExecutionContext
#include <cuda_runtime_api.h> // for cudaStream_t

#include <memory>           // for std::unique_ptr
#include <cstdint> // for std::int32_t
#include <filesystem>       // for std::filesystem::path
#include <string>  // for std::string
#include <cstddef> // for std::size_t
#include <vector> // for std::vector

// TensorRT가 전달하는 오류, 경고 메시지를 처리하기 위한 로거 클래스
class TensorRTLogger final : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* message) noexcept override;
};

struct TensorInfo
{
	std::string name;

	nvinfer1::Dims shape{};
	nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;

	std::size_t elementCount = 0;
	std::size_t byteSize = 0;
};

class InferenceInstance
{
public:
	InferenceInstance() noexcept = default;
	~InferenceInstance() noexcept;

	InferenceInstance(const InferenceInstance&) = delete;
	InferenceInstance& operator=(const InferenceInstance&) = delete;

	InferenceInstance(InferenceInstance&&) = delete;
	InferenceInstance& operator=(InferenceInstance&&) = delete;
	
	bool Open(const std::filesystem::path& enginePath, std::int32_t deviceId = 0);
	void Close() noexcept;
	bool IsOpen() const noexcept;

	bool Infer() noexcept;

private:
	void PrintModelInfo() const noexcept;
	bool SelectDevice() const noexcept;
	bool ReadModelInfo();

	bool AllocateBuffers();
	void ReleaseBuffers() noexcept;

	bool PrepareExecution() noexcept;

	bool RunTensor() noexcept;

private:
	std::int32_t m_deviceId = -1;

	TensorInfo m_inputInfo;
	TensorInfo m_outputInfo;

	// CPU memory
	std::vector<float> m_hostInput;
	std::vector<float> m_hostOutput;

	// GPU memory
	void* m_deviceInput = nullptr;
	void* m_deviceOutput = nullptr;

	cudaStream_t m_stream = nullptr;

	TensorRTLogger m_logger;

	std::unique_ptr<nvinfer1::IRuntime> m_runtime;
	std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
	std::unique_ptr<nvinfer1::IExecutionContext> m_executionContext;
};

