package main

import (
	// "tiny-neural/internal/layers"
	// "tiny-neural/internal/model"

	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	ws := mat.NewDense(4, 3, []float64{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6})
	biases := mat.NewVecDense(3, []float64{1, 1, 1})

	ws.Add(ws, biases)

	fmt.Println(ws)

	// bias := mat.
	// l1 := layers.NewLayerDense(2, 2, "sigmoid")
	// l2 := layers.NewLayerDense(1, 2, "sigmoid")
	// model := model.NewModel([]*layers.LayerDense{l1, l2})
	// model.Fit(mat.NewDense(4, 2, []float64{0, 0, 0, 1, 1, 0, 1, 1}), []float64{0, 1, 1, 0}, 10)
}
