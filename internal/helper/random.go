package helper

import "math/rand"

func RandomArray(size int) []float64 {
	arr := make([]float64, size)

	for i := 0; i < len(arr); i++ {
		arr[i] = rand.Float64()
	}
	return arr
}
