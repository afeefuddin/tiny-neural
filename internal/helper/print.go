package helper

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Print(mat mat.Dense) {
	rows, cols := mat.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fmt.Printf("%v ", mat.At(i, j))
		}
		fmt.Println()
	}
}
