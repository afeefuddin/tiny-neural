package model

import (
	"tiny-neural/internal/helper"
	"tiny-neural/internal/layers"

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
	current := xTrain

	for _, layer := range m.layers {
		helper.Print(*current)
		current = layer.Forward(current)
	}

	helper.Print(*current)
}
