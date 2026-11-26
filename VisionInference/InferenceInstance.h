#pragma once

#include <NvInferRuntime.h> // for nvinfer1::ILogger, IRuntime, ICudaEngine

#include <memory>           // for std::unique_ptr
#include <cstdint> // for std::int32_t
#include <filesystem>       // for std::filesystem::path

// TensorRT가 전달하는 오류, 경고 메시지를 처리하기 위한 로거 클래스
class TensorRTLogger final : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* message) noexcept override;
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

private:
	void PrintModelInfo() const noexcept;

private:
	TensorRTLogger m_logger;

	std::unique_ptr<nvinfer1::IRuntime> m_runtime;
	std::unique_ptr<nvinfer1::ICudaEngine> m_engine;

};

