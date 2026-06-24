package optimizer

import "tiny-neural/internal/layers"

type BatchGradientDescentOptimizer struct {
	learning_rate float64
}

func NewBatchGradientDescentOptimizer(lr float64) *BatchGradientDescentOptimizer {
	return &BatchGradientDescentOptimizer{
		learning_rate: lr,
	}
}

func (o *BatchGradientDescentOptimizer) Update(layer *layers.LayerDense) {
	weights := layer.Weights()
	rows, cols := weights.Dims()

	for i := range rows {
		for j := range cols {
			weights.Set(i, j, weights.At(i, j)-o.learning_rate*layer.DWeights().At(i, j))
		}
	}

	for i := 0; i < layer.NumberOfNeurons; i++ {
		layer.Biases().SetVec(i, layer.Biases().AtVec(i)-o.learning_rate*layer.DBiases().AtVec(i))
	}
}
