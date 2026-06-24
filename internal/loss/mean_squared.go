package loss

import "gonum.org/v1/gonum/mat"

type MeanSquaredLoss struct {
	BaseLoss
}

func NewMeanSquaredLoss() *MeanSquaredLoss {
	return &MeanSquaredLoss{}
}

func (l *MeanSquaredLoss) Backward(dinputs *mat.Dense, expected *mat.VecDense) {
	rows, cols := dinputs.Dims()
	dvalues := mat.NewDense(rows, cols, nil)

	dvalues.Apply(func(i, j int, prediction float64) float64 {
		return 2.0 * (prediction - expected.AtVec(i)) / float64(rows)
	}, dinputs)

	l.dvalues = dvalues
}
