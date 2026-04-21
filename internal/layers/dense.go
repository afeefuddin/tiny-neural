package layers

import (
	"errors"
	"tiny-neural/internal/activation"
	"tiny-neural/internal/helper"

	"gonum.org/v1/gonum/mat"
)

type LayerDense struct {
	numberOfNeurons int
	inputSize       int
	activationLayer activation.ActivationLayer

	// matrix of numberOfNeurons * inputSize
	weights *mat.Dense
	// array of size of numberOfNeurons
	biases *mat.VecDense

	// matrix of anyNumberOfInputs * inputSize
	inputs *mat.Dense
	// matrix of anyNumberOfInputs * numberOfNeurons
	output *mat.Dense

	// gradients
	dweights *mat.Dense
	dbiases  *mat.VecDense
	dinputs  *mat.Dense
}

func NewLayerDense(numberOfNeurons, inputSize int, activationFn string) *LayerDense {
	weights := mat.NewDense(numberOfNeurons, inputSize, helper.RandomArray(numberOfNeurons*inputSize))
	biases := mat.NewVecDense(numberOfNeurons, nil)

	var activationLayer activation.ActivationLayer

	if activationFn == "relu" {
		activationLayer = activation.NewReLULayer()
	} else if activationFn == "sigmoid" {
		activationLayer = activation.NewSigmoidLayer()
	}

	return &LayerDense{
		numberOfNeurons: numberOfNeurons,
		inputSize:       inputSize,
		weights:         weights,
		biases:          biases,
		activationLayer: activationLayer,
	}
}

func (layer *LayerDense) Forward(data *mat.Dense) (*mat.Dense, error) {
	input_rows, c := data.Dims()
	if c != layer.inputSize {
		return nil, errors.New("Invalid input size")
	}

	// cache the inputs
	layer.inputs = mat.DenseCopyOf(data)

	result := mat.NewDense(input_rows, layer.numberOfNeurons, nil)
	result.Mul(data, layer.weights.T())

	result.Apply(func(i, j int, v float64) float64 {
		return v + layer.biases.AtVec(j)
	}, result)

	layer.activationLayer.Forward(result)
	layer.output = layer.activationLayer.GetOutput()
	return layer.output, nil
}

func (layer *LayerDense) Backward(dvalues *mat.Dense) {
	layer.activationLayer.Backward(dvalues)
	activatedGradients := layer.activationLayer.GetDInputs()

	dweights := mat.NewDense(layer.numberOfNeurons, layer.inputSize, nil)
	dweights.Mul(activatedGradients.T(), layer.inputs)
	layer.dweights = dweights

	dbiases := mat.NewVecDense(layer.numberOfNeurons, nil)
	for i := 0; i < layer.numberOfNeurons; i++ {
		dbiases.SetVec(i, mat.Sum(activatedGradients.ColView(i)))
	}
	layer.dbiases = dbiases

	input_rows, input_cols := layer.inputs.Dims()
	dinputs := mat.NewDense(input_rows, input_cols, nil)
	dinputs.Mul(activatedGradients, layer.weights)
	layer.dinputs = dinputs
}

func (layer *LayerDense) DInputs() *mat.Dense {
	return layer.dinputs
}

func (layer *LayerDense) Update(learningRate float64) {
	rows, cols := layer.weights.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			layer.weights.Set(i, j, layer.weights.At(i, j)-learningRate*layer.dweights.At(i, j))
		}
	}

	for i := 0; i < layer.numberOfNeurons; i++ {
		layer.biases.SetVec(i, layer.biases.AtVec(i)-learningRate*layer.dbiases.AtVec(i))
	}
}
