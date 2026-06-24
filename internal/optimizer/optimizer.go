package optimizer

import (
	"gonum.org/v1/gonum/mat"
)

type Optimizer interface {
	GetRows(yTrain *mat.VecDense) *mat.VecDense
}
