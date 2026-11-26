#include "VisionInference.h"  // for public C API declarations, VI_Status

#include "InferenceInstance.h" // for InferenceInstance

#include <new>        // for std::nothrow, std::bad_alloc
#include <filesystem> // for std::filesystem::path

// 실제 정의는 DLL 내부에서만 공개
struct VI_Context final
{
    InferenceInstance instance;

	VI_Context() noexcept = default;
	~VI_Context() noexcept = default;

	VI_Context(const VI_Context&) = delete;
	VI_Context& operator=(const VI_Context&) = delete;
};

VI_Status VI_CALL VI_Create(VI_Handle* out_handle) noexcept
{
	if (out_handle == nullptr)
	{
		return VI_STATUS_INVALID_ARGUMENT;
	}

	*out_handle = nullptr;

	VI_Context* context = new (std::nothrow) VI_Context{};

	if (context == nullptr)
	{
		return VI_STATUS_OUT_OF_MEMORY;
	}
	
	*out_handle = context;

	return VI_STATUS_OK;
}

void VI_CALL VI_Destroy(VI_Handle handle) noexcept
{
	delete handle;
}

VI_Status VI_CALL VI_OpenModel(VI_Handle handle, const wchar_t* enginePath, int32_t deviceId) noexcept
{
	if (handle == nullptr || enginePath == nullptr || enginePath[0] == L'\0' || deviceId < 0)
	{
		return VI_STATUS_INVALID_ARGUMENT;
	}

	try
	{
		const std::filesystem::path path(enginePath);

		const bool opened = handle->instance.Open(path, deviceId);

		if (!opened)
		{
			return VI_STATUS_MODEL_OPEN_FAILED;
		}

		return VI_STATUS_OK;
	}
	catch (const std::bad_alloc&)
	{
		return VI_STATUS_OUT_OF_MEMORY;
	}
	catch (...)
	{
		return VI_STATUS_INTERNAL_ERROR;
	}
}

void VI_CALL VI_CloseModel(VI_Handle handle) noexcept
{
	if (handle == nullptr)
	{
		return;
	}

	handle->instance.Close();
}

VI_Status VI_CALL VI_InferClassification(VI_Handle handle, const VI_ImageView* image) noexcept
{
	if (handle == nullptr || image == nullptr)
	{
		return VI_STATUS_INVALID_ARGUMENT;
	}

	if (image->data == nullptr || image->width <= 0 || image->height <= 0)
	{
		return VI_STATUS_INVALID_ARGUMENT;
	}

	if (image->pixelFormat != VI_PIXEL_FORMAT_BGR8)
	{
		return VI_STATUS_INVALID_ARGUMENT;
	}

	// BGR8은 한 픽셀에 3바이트가 필요하다.
	// 나눗셈으로 비교하여 width * 3의 오버플로를 피한다.
	if (image->strideBytes / 3 < static_cast<size_t>(image->width))
	{
		return VI_STATUS_INVALID_ARGUMENT;
	}

	if (!handle->instance.IsOpen())
	{
		return VI_STATUS_MODEL_NOT_OPEN;
	}

	return handle->instance.Infer(*image) ? VI_STATUS_OK : VI_STATUS_INFERENCE_FAILED;
}
