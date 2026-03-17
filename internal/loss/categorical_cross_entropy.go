package loss

import (
	"math"
	"tiny-neural/internal/helper"

	"gonum.org/v1/gonum/mat"
)

type CategoricalCrossEntropyLoss struct {
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
