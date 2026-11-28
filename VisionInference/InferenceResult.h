#pragma once

#include <cstdint> // for std::int32_t
#include <span> // for std::span
#include <vector> // for std::vector

struct ClassificationScore
{
	std::int32_t classId = -1;
	float score = 0.0f;
};

struct ClassificationResult
{
	std::vector<ClassificationScore> scores;
};

class ResultBuilder
{
public:
	void Reset() noexcept
	{
		m_classification.clear();
	}

	void AddClassificationResult(std::span<const ClassificationScore> scores)
	{
		ClassificationResult result;
		result.scores.assign(scores.begin(), scores.end());
		m_classification.push_back(result);
	}

	std::span<const ClassificationResult> Classification() const noexcept
	{
		return m_classification;
	}

private:
	std::vector<ClassificationResult> m_classification;
};