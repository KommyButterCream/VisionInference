#include "VisionInference.h"

#include "InferenceInstance.h"

#include <new>     // for std::nothrow
#include <limits> // for std::numeric_limits

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

void VI_CALL VI_OpenModel(VI_Handle handle) noexcept
{
	if (handle == nullptr)
	{
		return;
	}

	try
	{
		//handle->instance.Open();
	}
	catch (...)
	{

	}
}

void VI_CALL VI_CloseModel(VI_Handle handle) noexcept
{
	if (handle == nullptr)
	{
		return;
	}

	try
	{
		//handle->instance.Close();
	}
	catch (...)
	{

	}
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
