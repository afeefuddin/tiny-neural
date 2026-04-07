package helper

import "gonum.org/v1/gonum/mat"

func SumVector(v mat.Vector) float64 {
	sum := 0.0
	for i := 0; i < v.Len(); i++ {
		sum += v.AtVec(i)
	}
	return sum
}
