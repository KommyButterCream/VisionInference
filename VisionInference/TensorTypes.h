#pragma once

#include <NvInferRuntime.h> // for nvinfer1::Dims, nvinfer1::DataType

#include <cstddef> // for std::size_t
#include <string> // for std::string
#include <vector> // for std::vector

// 모델의 텐서 정보
struct TensorInfo
{
	std::string name;
	nvinfer1::Dims shape{};
	nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;

	std::size_t elementCount = 0;
	std::size_t byteSize = 0;
};

// 모델의 모든 입력, 출력 목록
struct TensorCatalog
{
	std::vector<TensorInfo> inputs;
	std::vector<TensorInfo> outputs;
};
