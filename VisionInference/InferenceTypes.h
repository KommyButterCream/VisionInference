#pragma once

#include "TensorTypes.h" // for shared tensor types

enum class InferenceStatus
{
    Ok,
    InvalidArgument,
    ModelNotOpen,
    PreprocessFailed,
    ExecutionFailed,
    PostprocessFailed,
    FileReadFailed,
    UnsupportedModel,
    ResourcePreparationFailed,
    ModelOpenFailed,
    OutOfMemory,
    InternalError
};