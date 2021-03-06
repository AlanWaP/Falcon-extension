
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class SigmoidConfig : public LayerConfig
{
public:
	size_t inputDim = 0;
	size_t batchSize = 0;

	SigmoidConfig(size_t _inputDim, size_t _batchSize)
	:inputDim(_inputDim),
	 batchSize(_batchSize), 
	 LayerConfig("Sigmoid")
	{};
};
