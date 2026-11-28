#pragma once

#include <cstdint> // for std::int32_t


enum class InferenceTaskKind
{
	Classification,
	Detection,
	Segmentation
};

enum class ProcessingBackend
{
	Cpu,
	Cuda
};

// 모델을 열 때 정하는 옵션
// 실행 중에는 변경하지 않는다.
struct ModelOptions
{
	InferenceTaskKind taskKind = InferenceTaskKind::Classification;

	ProcessingBackend preprocessBackend = ProcessingBackend::Cpu;
	ProcessingBackend postprocessBackend = ProcessingBackend::Cpu;
};

// 실제 전처리에서 사용한 변환 정보를 기록한다.
// 검출·분할에서는 원본 좌표 복원에도 사용한다.
struct ImageTransform
{
	std::int32_t originalWidth = 0;
	std::int32_t originalHeight = 0;

	std::int32_t resizedWidth = 0;
	std::int32_t resizedHeight = 0;

	std::int32_t cropLeft = 0;
	std::int32_t cropTop = 0;

	std::int32_t targetWidth = 0;
	std::int32_t targetHeight = 0;
};
