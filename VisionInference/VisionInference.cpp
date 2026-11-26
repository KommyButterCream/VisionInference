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

void VI_CALL VI_InferClassification(VI_Handle handle) noexcept
{
	if (handle == nullptr)
	{
		return;
	}

	try
	{
		//handle->instance.Infer();
	}
	catch (...)
	{
		
	}
}
