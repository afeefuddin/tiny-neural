package activation

import "gonum.org/v1/gonum/mat"

type SoftMaxLayer struct {
	Output *mat.Dense
}

func NewSoftMaxLayer() *SoftMaxLayer {
	return &SoftMaxLayer{}
}

func (layer *SoftMaxLayer) Forward(input mat.Dense) {
	r, c := input.Dims()

	rowMax := make([]float64, r)

	for i := 0; i < r; i++ {
		rowMax[i] = mat.Max(input.RowView(i))
	}

	normalized := mat.NewDense(r, c, nil)
	normalized.Apply(func(i, j int, v float64) float64 {
		return v - rowMax[i]
	}, &input)

	// exponent of everything
	normalized.Exp(normalized)

	rowWiseSum := make([]float64, r)

	for i := 0; i < r; i++ {
		rowWiseSum[i] = mat.Sum(normalized.RowView(i))
	}

	normalized.Apply(func(i, j int, v float64) float64 {
		return v / rowWiseSum[i]
	}, normalized)

	layer.Output = normalized
}
