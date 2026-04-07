package activation

import "gonum.org/v1/gonum/mat"

type SoftMaxLayer struct {
	Output  *mat.Dense
	dinputs *mat.Dense
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

func (layer *SoftMaxLayer) Backward(dvalues *mat.Dense) {
	r, c := dvalues.Dims()
	dinputs := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		singleOutput := layer.Output.RowView(i)

		s := mat.NewDense(c, 1, nil)
		for j := 0; j < c; j++ {
			s.Set(j, 0, singleOutput.At(j, 0))
		}

		// --- Compute Jacobian ---
		// J = diag(s) - s * s^T

		diag := mat.NewDense(c, c, nil)
		for j := 0; j < c; j++ {
			diag.Set(j, j, s.At(j, 0))
		}

		var outer mat.Dense
		outer.Mul(s, s.T())

		var jacobian mat.Dense
		jacobian.Sub(diag, &outer)

		singleDvalues := dvalues.RowView(i)

		dv := mat.NewDense(c, 1, nil)
		for j := 0; j < c; j++ {
			dv.Set(j, 0, singleDvalues.At(j, 0))
		}

		var result mat.Dense
		result.Mul(&jacobian, dv)

		for j := 0; j < c; j++ {
			dinputs.Set(i, j, result.At(j, 0))
		}
	}

	layer.dinputs = dinputs
}
