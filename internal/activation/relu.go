package activation

import "gonum.org/v1/gonum/mat"

type ReLULayer struct {
	BaseActivation
}

func NewReLULayer() *ReLULayer {
	return &ReLULayer{}
}

func (layer *ReLULayer) Forward(inputs *mat.Dense) {
	layer.inputs = inputs

	r, c := inputs.Dims()
	layer.output = mat.NewDense(r, c, nil)

	layer.output.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	}, inputs)
}

func (layer *ReLULayer) Backward(dinputs *mat.Dense) {
	var copy mat.Dense
	copy.CloneFrom(dinputs)
	layer.dinputs = &copy

	layer.dinputs.Apply(func(i, j int, v float64) float64 {
		if v < 0.0 {
			return 0.0
		} else {
			return v
		}
	}, layer.dinputs)
}
