package main

import (
	"tiny-neural/internal/layers"
	"tiny-neural/internal/model"

	"gonum.org/v1/gonum/mat"
)

func main() {
	l1 := layers.NewLayerDense(1, 2, "relu")
	model := model.NewModel([]*layers.LayerDense{l1})
	model.Fit(mat.NewDense(4, 2, []float64{0, 0, 0, 1, 1, 0, 1, 1}), []float64{0, 1, 1, 1}, 10)
}
