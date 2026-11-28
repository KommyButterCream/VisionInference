#pragma once


#include "VisionInference.h"

#include <cstddef>

inline bool IsValidImageView(const VI_ImageView& image) noexcept
{
	if (image.data == nullptr || image.width <= 0 || image.height <= 0)
	{
		return false;
	}

	if (image.pixelFormat != VI_PIXEL_FORMAT_BGR8)
	{
		return false;
	}

	return image.strideBytes / 3 >= static_cast<std::size_t>(image.width);
}