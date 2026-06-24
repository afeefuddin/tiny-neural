package loss

import (
	"math"
	"tiny-neural/internal/helper"
	"tiny-neural/internal/preprocessing"

	"gonum.org/v1/gonum/mat"
)

type CategoricalCrossEntropyLoss struct {
	BaseLoss
}

func NewCategoricalCrossEntropyLoss() *CategoricalCrossEntropyLoss {
	return &CategoricalCrossEntropyLoss{}
}

// expected must have the class targets
func (loss *CategoricalCrossEntropyLoss) Forward(output *mat.Dense, expected []int) float64 {
	extractedData := make([]float64, len(expected))

	for i, classIndex := range expected {
		prob := helper.Clip(output.At(i, classIndex), 1e-7, 1-1e-7)
		extractedData[i] = -math.Log(prob)
	}

	mean := helper.Mean(extractedData)
	return mean
}

// here y_true is one hot encoded that is basically
// for every sample the input will be something like this [0, 1, 0, 0] given that their are 4 labels
func (loss *CategoricalCrossEntropyLoss) Backward(dvalues *mat.Dense, y_true *mat.VecDense) {
	// Do one hot encoding
	samples, labels := dvalues.Dims()
	y_encoded := preprocessing.OneHotEncode(y_true, labels)

	dinputs := mat.NewDense(samples, labels, nil)
	dinputs.Apply(func(i, j int, value float64) float64 {
		predictedProb := helper.Clip(dvalues.At(i, j), 1e-7, 1-1e-7)
		return -(value / predictedProb) / float64(samples)
	}, y_encoded)

	loss.dvalues = dinputs
}
