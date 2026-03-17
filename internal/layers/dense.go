package layers

import (
	"tiny-neural/internal/activation"
	"tiny-neural/internal/helper"

	"gonum.org/v1/gonum/mat"
)

type LayerDense struct {
	numberOfNeurons int
	inputSize       int
	activationFn    string
	weights         *mat.Dense
	biases          []float64
}

func NewLayerDense(numberOfNeurons, inputSize int, activationFn string) *LayerDense {
	weights := mat.NewDense(numberOfNeurons, inputSize, helper.RandomArray(numberOfNeurons*inputSize))
	biases := make([]float64, numberOfNeurons)

	return &LayerDense{
		numberOfNeurons: numberOfNeurons,
		inputSize:       inputSize,
		activationFn:    activationFn,
		weights:         weights,
		biases:          biases,
	}
}

func (layer *LayerDense) Forward(data *mat.Dense) *mat.Dense {
	input_rows, _ := data.Dims()
	result := mat.NewDense(input_rows, layer.numberOfNeurons, nil)
	result.Mul(data, layer.weights.T())
	result_rows, result_cols := result.Dims()
	broadcast_bias := make([]float64, result_rows*result_cols)
	for i := 0; i < len(broadcast_bias); i++ {
		broadcast_bias[i] = layer.biases[i%len(layer.biases)]
	}
	bias_matrix := mat.NewDense(result_rows, result_cols, broadcast_bias)

	result.Add(result, bias_matrix)

	// apply activationFn
	if layer.activationFn == "sigmoid" {
		result.Apply(func(i, j int, v float64) float64 { return activation.SigmoidForward(v) }, result)
	}

	return result
}
