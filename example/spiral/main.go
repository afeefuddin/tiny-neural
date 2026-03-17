package main

import (
	"math"
	"math/rand"
	"tiny-neural/internal/helper"

	"gonum.org/v1/gonum/mat"
)

func buildDataSet() (*mat.Dense, []float64) {

	pointsPerClass := 100
	classes := 3
	total := pointsPerClass * classes

	Xdata := make([]float64, total*2)
	y := make([]float64, total)

	for class := 0; class < classes; class++ {
		for i := 0; i < pointsPerClass; i++ {

			r := float64(i) / float64(pointsPerClass)
			t := float64(class)*4 + 4*r + rand.NormFloat64()*0.2

			x := r * math.Sin(t*2.5)
			yv := r * math.Cos(t*2.5)

			idx := class*pointsPerClass + i

			Xdata[idx*2] = x
			Xdata[idx*2+1] = yv

			y[idx] = float64(class)
		}
	}

	X := mat.NewDense(total, 2, Xdata)

	return X, y
}

func main() {

	X, y := buildDataSet()

	r, c := X.Dims()

	println("Samples:", r)
	println("Features:", c)
	println("Labels:", len(y))
	helper.Print(*X)
}
