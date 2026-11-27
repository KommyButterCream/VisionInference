#include "Buffers.h"

#include <cuda_runtime_api.h> // for CUDA allocation, stream, device and error APIs
#include <cstdio> // for std::fprintf, stderr

namespace
{
	bool CheckCuda(cudaError_t status, const char* operation) noexcept
	{
		if (status == cudaSuccess)
		{
			return true;
		}

		std::fprintf(
			stderr,
			"[VisionInference] %s failed: %s\n",
			operation,
			cudaGetErrorString(status));

		return false;
	}
}

DeviceBuffer::~DeviceBuffer() noexcept
{
	Reset();
}

bool DeviceBuffer::Allocate(std::size_t byteSize, int deviceId) noexcept
{
	// 이미 소유 중인 버퍼는 암묵적으로 해제하거나 교체하지 않는다.
	if (m_data != nullptr || byteSize == 0 || deviceId < 0)
	{
		return false;
	}

	if (!CheckCuda(cudaSetDevice(deviceId), "cudaSetDevice"))
	{
		return false;
	}

	void* data = nullptr;

	if (!CheckCuda(cudaMalloc(&data, byteSize), "cudaMalloc"))
	{
		return false;
	}

	m_data = data;
	m_byteSize = byteSize;
	m_deviceId = deviceId;

	return true;
}

void DeviceBuffer::Reset() noexcept
{
	if (m_data == nullptr)
	{
		return;
	}

	if (!CheckCuda(cudaSetDevice(m_deviceId), "cudaSetDevice during buffer reset"))
	{
		return;
	}

	CheckCuda(cudaFree(m_data), "cudaFree");

	m_data = nullptr;
	m_byteSize = 0;
	m_deviceId = -1;
}

CudaStream::~CudaStream() noexcept
{
	Reset();
}

bool CudaStream::Create(int deviceId) noexcept
{
	if (m_stream != nullptr || deviceId < 0)
	{
		return false;
	}

	if (!CheckCuda(cudaSetDevice(deviceId), "cudaSetDevice"))
	{
		return false;
	}

	cudaStream_t stream = nullptr;

	if (!CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate"))
	{
		return false;
	}

	m_stream = stream;
	m_deviceId = deviceId;

	return true;
}

bool CudaStream::Synchronize() const noexcept
{
	// 아직 생성하지 않았거나 이미 정리된 경우에는 기다릴 작업이 없다.
	if (m_stream == nullptr)
	{
		return true;
	}

	if (!CheckCuda(cudaSetDevice(m_deviceId), "cudaSetDevice during synchronization"))
	{
		return false;
	}

	return CheckCuda(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize");
}

void CudaStream::Reset() noexcept
{
	if (m_stream == nullptr)
	{
		return;
	}

	if (!CheckCuda(cudaSetDevice(m_deviceId), "cudaSetDevice during stream reset"))
	{
		return;
	}

	CheckCuda(cudaStreamDestroy(m_stream), "cudaStreamDestroy");

	m_stream = nullptr;
	m_deviceId = -1;
}
