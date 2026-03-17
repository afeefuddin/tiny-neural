package internal

import (
	"math"
	"math/rand"
)

type Neuron struct {
	Weights *Tensor
	Bias    float64
	outputA []float64
	input   []*Tensor
}

func NewNeuron(rows, cols int) *Neuron {
	Weights := NewTensor(rows, cols)
	Bias := rand.Float64()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			Weights.Data[i][j] = rand.Float64()
		}
	}

	return &Neuron{
		Weights: Weights,
		Bias:    Bias,
	}
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (p *Neuron) Forward(data []*Tensor) error {
	p.input = data
	n := len(data)
	// p.outputZ = make([]float64, n)
	p.outputA = make([]float64, n)
	for i := 0; i < len(data); i++ {
		// dataPoint := data[i]
		dotProd, err := data[i].Dot(p.Weights)

		if err != nil {
			return err
		}
		p.outputA[i] = Sigmoid(dotProd + p.Bias)
	}
	return nil
}

func (p *Neuron) Backward(expected []float64, dA []float64) []float64 {
	// loss := calculateLoss(expected, output)
	data := p.input
	lr := 0.1
	descent := NewTensor(data[0].Rows, data[0].Cols)
	db := 0.0

	errors := make([]float64, len(expected))

	for k := 0; k < len(expected); k++ {
		// dL_da =
		a := p.outputA[k]
		sigmoidPrime := a * (1 - a)

		dZ := 0.0
		if len(dA) == 0 {
			dZ = 2 * (p.outputA[k] - expected[k]) * sigmoidPrime
		} else {
			dZ = dA[k] * sigmoidPrime
		}

		errors[k] = dZ

		for i := 0; i < data[k].Rows; i++ {
			for j := 0; j < data[k].Cols; j++ {
				descent.Data[i][j] += (dZ * data[k].Data[i][j])
			}
		}

		db += dZ
	}

	m := float64(len(data))
	for i := 0; i < descent.Rows; i++ {
		for j := 0; j < descent.Cols; j++ {
			p.Weights.Data[i][j] -= lr * (descent.Data[i][j] / m)
		}
	}

	p.Bias -= lr * (db / m)

	return errors
}
