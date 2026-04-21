package model

import (
	"tiny-neural/internal/layers"
	"tiny-neural/internal/loss"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	layers []*layers.LayerDense
	lossFn string
}

func NewModel(layers []*layers.LayerDense) *Model {
	return &Model{
		layers: layers,
	}
}

func (m *Model) Fit(xTrain *mat.Dense, yTrain []float64, epochs int) {
	learningRate := 0.1

	for i := 0; i < epochs; i++ {
		current := xTrain

		for _, layer := range m.layers {
			c, err := layer.Forward(current)
			if err != nil {
				panic(err)
			}
			current = c
		}

		l := loss.NewMeanSquaredLoss()

		dvalues := l.Backward(current, yTrain)

		for layerIndex := len(m.layers) - 1; layerIndex >= 0; layerIndex-- {
			layer := m.layers[layerIndex]
			layer.Backward(dvalues)
			dvalues = layer.DInputs()
		}

		for _, layer := range m.layers {
			layer.Update(learningRate)
		}
	}
}
