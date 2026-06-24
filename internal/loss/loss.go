package loss

import "gonum.org/v1/gonum/mat"

type Loss interface {
	// Forward(output *mat.Dense, expected []int)
	Backward(dvalues *mat.Dense, y_true *mat.VecDense)
	GetDValues() *mat.Dense
	// GetOutput() *mat.Dense
	// GetDInputs() *mat.Dense
}

type BaseLoss struct {
	dvalues *mat.Dense
}

func (l *BaseLoss) GetDValues() *mat.Dense {
	return l.dvalues
}
