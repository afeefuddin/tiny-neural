package model

import (
	"fmt"
	"tiny-neural/internal/layers"
	"tiny-neural/internal/loss"
	"tiny-neural/internal/optimizer"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	layers    []*layers.LayerDense
	lossFn    string
	optimizer optimizer.Optimizer
}

func NewModel(layers []*layers.LayerDense, lossFn string) *Model {
	return &Model{
		layers: layers,
		lossFn: lossFn,
	}
}

func classTargets(yTrain *mat.VecDense) []int {
	targets := make([]int, yTrain.Len())
	for i := 0; i < yTrain.Len(); i++ {
		targets[i] = int(yTrain.AtVec(i))
	}
	return targets
}

func meanSquaredLoss(predictions *mat.Dense, expected *mat.VecDense) float64 {
	rows, cols := predictions.Dims()
	if rows == 0 || cols == 0 {
		return 0
	}

	total := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := predictions.At(i, j) - expected.AtVec(i)
			total += diff * diff
		}
	}

	return total / float64(rows*cols)
}

func (m *Model) calculateLoss(predictions *mat.Dense, yTrain *mat.VecDense) float64 {
	if m.lossFn == "categorical-cross-entropy" {
		l := loss.NewCategoricalCrossEntropyLoss()
		return l.Forward(predictions, classTargets(yTrain))
	}

	if m.lossFn == "mse" {
		return meanSquaredLoss(predictions, yTrain)
	}

	return 0
}

func (m *Model) Fit(xTrain *mat.Dense, yTrain *mat.VecDense, epochs int) {
	// learningRate := 0.1
	trainingRows := m.optimizer.GetRows(yTrain)

	for i := 0; i < epochs; i++ {
		current := xTrain

		for _, layer := range m.layers {
			c, err := layer.Forward(current)
			if err != nil {
				panic(fmt.Sprintf("forward failed: %v", err))
			}
			current = c
		}

		currentLoss := m.calculateLoss(current, yTrain)
		fmt.Printf("epoch %d/%d loss: %.6f\n", i+1, epochs, currentLoss)

		var lossFunction loss.Loss

		if m.lossFn == "categorical-cross-entropy" {
			lossFunction = loss.NewCategoricalCrossEntropyLoss()
		} else if m.lossFn == "mse" {
			lossFunction = loss.NewMeanSquaredLoss()
		}

		lossFunction.Backward(current, trainingRows)
		dvalues := lossFunction.GetDValues()

		for layerIndex := len(m.layers) - 1; layerIndex >= 0; layerIndex-- {
			layer := m.layers[layerIndex]
			layer.Backward(dvalues)
			dvalues = layer.DInputs()
		}

		// o := optimizer.NewSGD(learningRate,)
		// o := optimizer.NewBatchGradientDescentOptimizer(learningRate)

		for _, layer := range m.layers {
			o.Update(layer)
		}
	}
}
