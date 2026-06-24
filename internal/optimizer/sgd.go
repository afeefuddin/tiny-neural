package optimizer

type SGD struct {
	learning_rate float64
	batch_size    int
}

func NewSGD(lr float64, batch_size int) *SGD {
	return &SGD{
		learning_rate: lr,
	}
}

// func (s *SGD) Update(layer *layers) {
// }
