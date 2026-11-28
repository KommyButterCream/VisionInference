#include "Buffers.h" // for DeviceBuffer, CudaStream, TensorBuffer

#include <cuda_runtime_api.h> // for CUDA allocation, stream, device and error APIs
#include <cstdio> // for std::fprintf, stderr
#include <limits> // for std::numeric_limits

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

bool TensorBuffer::Allocate(const TensorInfo& info, int deviceId)
{
	// 사용 중인 버퍼를 암묵적으로 교체하지 않는다.
	if (m_info != nullptr || m_device.Data() != nullptr)
	{
		return false;
	}

	// 현재는 FP32 텐서 지원
	if (deviceId < 0 ||
		info.dataType != nvinfer1::DataType::kFLOAT ||
		info.elementCount == 0)
	{
		return false;
	}

	constexpr std::size_t maxSize = (std::numeric_limits<std::size_t>::max)();

	if (info.elementCount > maxSize / sizeof(float))
	{
		return false;
	}

	const std::size_t byteSize = info.elementCount * sizeof(float);

	if (info.byteSize != byteSize)
	{
		return false;
	}

	// CPU 메모리 할당이 실패하면 예외 발생
	// 현재 Open 의 예외 처리에서 OutOfMemory 로 변환
	std::vector<float> host(info.elementCount, 0.0f);

	if (!m_device.Allocate(byteSize, deviceId))
	{
		return false;
	}

	// 두 메모리의 준비가 끝난 뒤에 소유권과 모델 참조를 확정한다.
	m_host.swap(host);
	m_info = &info;

	return true;
}

void TensorBuffer::Reset() noexcept
{
	m_device.Reset();

	std::vector<float>().swap(m_host);

	m_info = nullptr;
}