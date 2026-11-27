#pragma once

#include <cuda_runtime_api.h> // for cudaStream_t
#include <cstddef> // for std::size_t

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

