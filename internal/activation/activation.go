package activation

import "gonum.org/v1/gonum/mat"

type ActivationLayer interface {
	Forward(inputs *mat.Dense)
	Backward(dvalues *mat.Dense)
	GetOutput() *mat.Dense
	GetDInputs() *mat.Dense
}

type BaseActivation struct {
	inputs  *mat.Dense
	output  *mat.Dense
	dinputs *mat.Dense
}

func (b *BaseActivation) GetOutput() *mat.Dense {
	return b.output
}

func (b *BaseActivation) GetDInputs() *mat.Dense {
	return b.dinputs
}
