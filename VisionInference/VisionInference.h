#pragma once

#include <stdint.h> // for int32_t
#include <stddef.h> // for wchar_t when included from C

#ifdef VISIONINFERENCE_EXPORTS
#define VI_API __declspec(dllexport)
#else
#define VI_API __declspec(dllimport)
#endif

#define VI_CALL __cdecl

#ifdef __cplusplus
#define VI_NOEXCEPT noexcept
extern "C" {
#else
#define VI_NOEXCEPT
#endif

typedef struct VI_Context* VI_Handle;
typedef int32_t VI_Status;

enum ENUM_VI_STATUS
{
    VI_STATUS_OK = 0,
    VI_STATUS_INVALID_ARGUMENT = 1,
    VI_STATUS_OUT_OF_MEMORY = 2,
    VI_STATUS_MODEL_OPEN_FAILED = 3,
    VI_STATUS_INTERNAL_ERROR = 4,
    VI_STATUS_MODEL_NOT_OPEN = 5,
    VI_STATUS_INFERENCE_FAILED = 6
};

VI_API VI_Status VI_CALL VI_Create(VI_Handle* out_handle) VI_NOEXCEPT;
VI_API void VI_CALL VI_Destroy(VI_Handle handle) VI_NOEXCEPT;

VI_API VI_Status VI_CALL VI_OpenModel(VI_Handle handle, const wchar_t* enginePath, int32_t deviceId) VI_NOEXCEPT;
VI_API void VI_CALL VI_CloseModel(VI_Handle handle) VI_NOEXCEPT;

VI_API VI_Status VI_CALL VI_InferClassification(VI_Handle handle) VI_NOEXCEPT;


#ifdef __cplusplus
}
#endif