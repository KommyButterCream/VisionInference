#pragma once

#include <NvInferRuntime.h> // for nvinfer1::Dims, nvinfer1::DataType

#include <cstddef> // for std::size_t
#include <string> // for std::string

struct TensorInfo
{
	std::string name;		// TensorRT 에 버퍼 주소를 등록할 때 사용하는 텐서 이름

	nvinfer1::Dims shape{};	// [1, 3, 224, 224] 같은 차원 정보
	nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT; // FP32 등 자료형

	std::size_t elementCount = 0;	// 전체 원소 개수
	std::size_t byteSize = 0;		// 메모리 할당, 복사에 필요한 바이트 단위 수량
};
