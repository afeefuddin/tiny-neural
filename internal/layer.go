package internal

type DenseLayer struct {
	Neurons []*Neuron
}

func NewDenseLayer(numberOfNeurons int, rows int, cols int) *DenseLayer {
	neurons := make([]*Neuron, numberOfNeurons)
	for i := 0; i < numberOfNeurons; i++ {
		neurons[i] = NewNeuron(rows, cols)
	}

	return &DenseLayer{
		Neurons: neurons,
	}
}

func (l *DenseLayer) Forward(data *Tensor) ([]float64, error) {
	outputs := make([]float64, len(l.Neurons))
	for i := 0; i < len(l.Neurons); i++ {
		err := l.Neurons[i].Forward([]*Tensor{data})
		if err != nil {
			return nil, err
		}
		outputs[i] = l.Neurons[i].outputA[0]
	}

	return outputs, nil
}

func (l *DenseLayer) Backward() {

}
