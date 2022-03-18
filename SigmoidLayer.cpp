
#pragma once
#include "SigmoidLayer.h"
#include "Functionalities.h"
using namespace std;

SigmoidLayer::SigmoidLayer(SigmoidConfig* conf, int _layerNum)
:Layer(_layerNum),
 conf(conf->inputDim, conf->batchSize),
 activations(conf->batchSize * conf->inputDim), 
 deltas(conf->batchSize * conf->inputDim)
{}


void SigmoidLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << layerNum+1 << ") Sigmoid Layer\t\t  " << conf.batchSize << " x " << conf.inputDim << endl;
}


void SigmoidLayer::forward(const RSSVectorMyType &inputActivation)
{
	log_print("Sigmoid.forward");

	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

	if (FUNCTION_TIME) {
		cout << "funcSigmoid: " << funcTime(divideByScalar, inputActivation, 2, activations) << endl;
	}
	else {
		//funcRELU(inputActivation, reluPrime, activations, size);
		divideByScalar(inputActivation, 2, activations);
	}
}


void SigmoidLayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("ReLU.computeDelta");

	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

	if (FUNCTION_TIME)
		cout << "funcSelectShares: " << funcTime(divideByScalar, deltas, 2, prevDelta) << endl;
	else
		divideByScalar(deltas, 2, prevDelta);
}


void SigmoidLayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("Sigmoid.updateEquations");
}
