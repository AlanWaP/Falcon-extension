
#pragma once
#include "SigmoidLayer.h"
#include "Functionalities.h"
using namespace std;

//#define DEBUG_PREDICT
#define DEBUG_LOSS

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

	copyVectors<RSSMyType>(inputActivation, activations, size);

	if (FUNCTION_TIME) {
		cout << "funcSigmoid: " << funcTime(funcTruncate, activations, 1, size) << endl;
	}
	else {
		//funcRELU(inputActivation, reluPrime, activations, size);
		funcTruncate(activations, 1, size);
	}

	#ifdef DEBUG_PREDICT
	vector<myType> d_input(size), d_act(size);
	funcReconstruct(inputActivation, d_input, size, "input", false);
	funcReconstruct(activations, d_act, size, "act", false);
	//cout << "input: ";
	//for (int i = 0; i < size; i++) cout << (static_cast<int32_t>(d_input[i])) / (float)(1 << FLOAT_PRECISION) << " ";
	//cout << "\n";
	cout << "predict(just 10): ";
	for (int i = 0; i < 10; i++) cout << (static_cast<int32_t>(d_act[i])) / (float)(1 << FLOAT_PRECISION) << " ";
	cout << "\n";
	#endif
}


void SigmoidLayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("ReLU.computeDelta");

	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t size = rows*columns;

	assert((columns == 1) && "inputDim not 1");

	copyVectors<RSSMyType>(deltas, prevDelta, size);

	if (FUNCTION_TIME)
		cout << "funcSelectShares: " << funcTime(funcTruncate, prevDelta, 1, size) << endl;
	else
		funcTruncate(prevDelta, 1, size);

	// test loss function
	#ifdef DEBUG_LOSS
	vector<myType> data_delta(size), data_act(size), d_prevDelta(size);
	vector<double> f_delta(size), f_act(size), f_prevDelta(size);
	funcReconstruct(deltas, data_delta, size, "delta", false);
	funcReconstruct(activations, data_act, size, "act", false);
	funcReconstruct(prevDelta, d_prevDelta, size, "prevDelta", false);
	for (int i = 0; i < size; i++) {
		f_delta[i] = (static_cast<int32_t>(data_delta[i])) / (double)(1 << FLOAT_PRECISION);
		f_prevDelta[i] = (static_cast<int32_t>(d_prevDelta[i])) / (double)(1 << FLOAT_PRECISION);
		f_act[i] = (static_cast<int32_t>(data_act[i])) / (double)(1<< FLOAT_PRECISION);
	}
	//cout << "g_loss: ";
	//for (int i = 0; i < size; i++) cout << f_prevDelta[i] << " "; cout << "\n";
	//cout << "f_delta(10): ";
	//for (int i = 0; i < 10; i++) cout << f_delta[i] << " "; cout << "\nact(10): ";
	//for (int i = 0; i < 10; i++) cout << f_act[i] << " "; cout << "\n";

	double loss = 0;
	for (int i = 0; i < size ; i++) loss += 0.6931471805599453 - f_act[i] * f_act[i] / 2 + f_delta[i] * f_act[i];
	cout << "loss: " << (loss / MINI_BATCH_SIZE) << "\n";
	#endif
}


void SigmoidLayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("Sigmoid.updateEquations");
}
