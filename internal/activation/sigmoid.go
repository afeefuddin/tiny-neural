package activation

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SigmoidLayer struct {
	BaseActivation
}

func NewSigmoidLayer() *SigmoidLayer {
	return &SigmoidLayer{}
}

func SigmoidForward(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (layer *SigmoidLayer) Forward(inputs *mat.Dense) {
	layer.inputs = inputs

	r, c := inputs.Dims()
	layer.output = mat.NewDense(r, c, nil)
	layer.output.Apply(func(i, j int, v float64) float64 {
		return SigmoidForward(v)
	}, inputs)
}

func (layer *SigmoidLayer) Backward(dvalues *mat.Dense) {
	r, c := dvalues.Dims()
	layer.dinputs = mat.NewDense(r, c, nil)
	layer.dinputs.Apply(func(i, j int, v float64) float64 {
		output := layer.output.At(i, j)
		return v * output * (1.0 - output)
	}, dvalues)
}
