package main

import (
	"fmt"
	"tiny-neural/internal/layers"
	"tiny-neural/internal/model"

	"gonum.org/v1/gonum/mat"
)

func main() {
	inputs := mat.NewDense(4, 2, []float64{0, 0, 0, 1, 1, 0, 1, 1})
	targets := []float64{0, 1, 1, 1}

	l1 := layers.NewLayerDense(1, 2, "sigmoid")
	model := model.NewModel([]*layers.LayerDense{l1})
	model.Fit(inputs, targets, 10000)

	output, err := l1.Forward(inputs)
	if err != nil {
		panic(err)
	}

	for i := 0; i < 4; i++ {
		fmt.Printf("%.6f\n", output.At(i, 0))
	}
}
