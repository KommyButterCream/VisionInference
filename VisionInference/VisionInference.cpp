#include "VisionInference.h"  // for public C API declarations, VI_Status

#include "InferenceInstance.h" // for InferenceInstance
#include "InferenceTypes.h"

#include <new>        // for std::nothrow, std::bad_alloc
#include <filesystem> // for std::filesystem::path

namespace
{
	VI_Status ToPublicStatus(InferenceStatus status) noexcept
	{
		switch (status)
		{
		case InferenceStatus::Ok:
			return VI_STATUS_OK;

		case InferenceStatus::InvalidArgument:
			return VI_STATUS_INVALID_ARGUMENT;

		case InferenceStatus::ModelNotOpen:
			return VI_STATUS_MODEL_NOT_OPEN;

		case InferenceStatus::PreprocessFailed:
			return VI_STATUS_PREPROCESS_FAILED;

		case InferenceStatus::ExecutionFailed:
			return VI_STATUS_INFERENCE_FAILED;

		case InferenceStatus::PostprocessFailed:
			return VI_STATUS_POSTPROCESS_FAILED;

		case InferenceStatus::FileReadFailed:
			return VI_STATUS_FILE_READ_FAILED;

		case InferenceStatus::UnsupportedModel:
			return VI_STATUS_UNSUPPORTED_MODEL;

		case InferenceStatus::ResourcePreparationFailed:
			return VI_STATUS_RESOURCE_PREPARATION_FAILED;

		case InferenceStatus::ModelOpenFailed:
			return VI_STATUS_MODEL_OPEN_FAILED;

		case InferenceStatus::OutOfMemory:
			return VI_STATUS_OUT_OF_MEMORY;

		case InferenceStatus::InternalError:
			return VI_STATUS_INTERNAL_ERROR;
		}

		return VI_STATUS_INTERNAL_ERROR;
	}
}

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

		const InferenceStatus status = handle->instance.Open(path, deviceId);

		return ToPublicStatus(status);
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

VI_Status VI_CALL VI_InferClassification(VI_Handle handle, const VI_ImageView* image, VI_ClassificationResult* outResult) noexcept
{
	if (outResult == nullptr)
	{
		return VI_STATUS_INVALID_ARGUMENT;
	}

	// 실패했을 때 이전 추론 결과가 남지 않도록 초기화한다.
	*outResult = { -1, 0.0f };

	if (handle == nullptr || image == nullptr)
	{
		return VI_STATUS_INVALID_ARGUMENT;
	}

	const InferenceStatus status = handle->instance.Infer(*image, *outResult);

	return ToPublicStatus(status);
}
