#pragma once

#include "TensorTypes.h" // for TensorInfo

#include <cuda_runtime_api.h> // for cudaStream_t
#include <cstddef> // for std::size_t
#include <span> // for std::span
#include <vector> // for std::vector

// CUDA GPU memory buffer 를 할당, 해제, 관리하는 클래스
class DeviceBuffer final
{
public:
    DeviceBuffer() noexcept = default;
    ~DeviceBuffer() noexcept;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    bool Allocate(std::size_t byteSize, int deviceId) noexcept;
    void Reset() noexcept;

    void* Data() const noexcept { return m_data; }
    std::size_t SizeBytes() const noexcept { return m_byteSize; }

private:
    void* m_data = nullptr;
    std::size_t m_byteSize = 0;
    int m_deviceId = -1;
};

// GPU에 제출하는 작업의 실행 순서를 관리하는 클래스
class CudaStream final
{
public:
    CudaStream() noexcept = default;
    ~CudaStream() noexcept;

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&&) = delete;
    CudaStream& operator=(CudaStream&&) = delete;

    bool Create(int deviceId) noexcept;
    bool Synchronize() const noexcept;
    void Reset() noexcept;

    cudaStream_t Get() const noexcept { return m_stream; }

private:
    cudaStream_t m_stream = nullptr;
    int m_deviceId = -1;
};

// 텐서 하나에 필요한 CPU 메모리와 GPU 메모리를 함께 관리
// FP32 텐서에 맞춰 CPU 메모리는 float 배열로 둔다.
class TensorBuffer final
{
public:
    TensorBuffer() noexcept = default;
    ~TensorBuffer() noexcept = default;

    // GPU 메모리의 소유권이 중복되지 않도록 복사와 이동을 막는다.
    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;

    TensorBuffer(TensorBuffer&&) = delete;
    TensorBuffer& operator=(TensorBuffer&&) = delete;

    bool Allocate(const TensorInfo& info, int deviceId);

    // 이 버퍼를 사용하는 GPU 작업이 끝난 뒤 호출해야 한다.
    void Reset() noexcept;

    std::span<float> Host() noexcept
    {
        return m_host;
    }

    std::span<const float> Host() const noexcept
    {
        return m_host;
    }

    void* DeviceData() noexcept
    {
        return m_device.Data();
    }

    const void* DeviceData() const noexcept
    {
        return m_device.Data();
    }

    std::size_t SizeBytes() const noexcept
    {
        return m_device.SizeBytes();
    }

    const TensorInfo* Info() const noexcept
    {
        return m_info;
    }

private:
    // 모델 정보를 복사하지 않고 참조
    // 참조 대상은 Reset 전까지 살아 있어야하고 변경하면 안된다.
    const TensorInfo* m_info = nullptr;

    std::vector<float> m_host;
    DeviceBuffer m_device;
};