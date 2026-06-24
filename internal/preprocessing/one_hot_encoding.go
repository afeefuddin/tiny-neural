package preprocessing

import "gonum.org/v1/gonum/mat"

func OneHotEncode(input *mat.VecDense, labelCount int) *mat.Dense {
	output := mat.NewDense(input.Len(), labelCount, nil)
	for i := 0; i < input.Len(); i++ {
		output.Set(i, int(input.AtVec(i)), 1.0)
	}
	
	return output
}
