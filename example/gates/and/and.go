package main

import (
	"fmt"
	"tiny-neural/internal"
)

func main() {
	neuron := internal.NewNeuron(1, 2)
	tensors := make([]*internal.Tensor, 4)
	for i := 0; i < 4; i++ {
		tensors[i] = internal.NewTensor(1, 2)
	}
	expected := []float64{0, 0, 0, 1}

	tensors[0].Data = [][]float64{
		{0, 0},
	}
	tensors[1].Data = [][]float64{
		{0, 1},
	}
	tensors[2].Data = [][]float64{
		{1, 0},
	}
	tensors[3].Data = [][]float64{
		{1, 1},
	}

	epochs := 10000

	fmt.Println("Starting AND gate training")
	for i := 0; i < epochs; i++ {
		err := neuron.Forward(tensors)
		if err != nil {
			fmt.Printf("Error %v", err)
			return
		}

		neuron.Backward(expected, nil)
	}

	neuron.Weights.Print()

	for i := 0; i < 4; i++ {
		op, err := tensors[i].Dot(neuron.Weights)
		if err != nil {
			fmt.Printf("Error %v", err)
			return
		}
		fmt.Printf("%f \n", internal.Sigmoid(op+neuron.Bias))
	}
}
