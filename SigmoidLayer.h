
#pragma once
#include "SigmoidConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern int partyNum;


class SigmoidLayer : public Layer
{
private:
	SigmoidConfig conf;
    // the output of this layer
	RSSVectorMyType activations; 
    // the delta of input
	RSSVectorMyType deltas; 

public:
	//Constructor and initializer
	SigmoidLayer(SigmoidConfig* conf, int _layerNum);

	//Functions
	void printLayer() override;
    // inputActivation is the input of this layer, i.e. wtx
    // The output of this layer (activations) is approx of adjusted sigmoid(wtx) = approx of 2/(1+exp(wtx))-1 = 0.5 * wtx
	void forward(const RSSVectorMyType& inputActivation) override; 
    // prevDelta is the delta of output
    // delta = 0.25 * wtx - 0.5 * y = 0.5 * activations - 0.5 * (activations - prevDelta) = 0.5 * prevDelta
	void computeDelta(RSSVectorMyType& prevDelta) override; 
	void updateEquations(const RSSVectorMyType& prevActivations) override;

	//Getters
	RSSVectorMyType* getActivation() {return &activations;};
	RSSVectorMyType* getDelta() {return &deltas;};
};