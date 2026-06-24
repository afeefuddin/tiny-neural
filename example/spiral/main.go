package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"tiny-neural/internal/layers"
	"tiny-neural/internal/model"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const (
	pointsPerClass = 100
	classes        = 3
	epochs         = 10000
)

func buildDataSet() (*mat.Dense, []float64) {
	total := pointsPerClass * classes
	rng := rand.New(rand.NewSource(42))

	Xdata := make([]float64, total*2)
	y := make([]float64, total)

	for class := range classes {
		for i := range pointsPerClass {

			r := float64(i) / float64(pointsPerClass)
			t := float64(class)*4 + 4*r + rng.NormFloat64()*0.2

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

func plotDataSet(X *mat.Dense, labels []float64, outputPath string) error {
	p := plot.New()
	p.Title.Text = "Spiral Dataset"
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"

	colors := []color.RGBA{
		{R: 239, G: 68, B: 68, A: 255},
		{R: 59, G: 130, B: 246, A: 255},
		{R: 34, G: 197, B: 94, A: 255},
	}

	classPoints := make(map[int]plotter.XYs)
	rows, _ := X.Dims()
	for i := range rows {
		classID := int(labels[i])
		classPoints[classID] = append(classPoints[classID], plotter.XY{
			X: X.At(i, 0),
			Y: X.At(i, 1),
		})
	}

	for classID, points := range classPoints {
		scatter, err := plotter.NewScatter(points)
		if err != nil {
			return err
		}

		scatter.GlyphStyle.Color = colors[classID%len(colors)]
		scatter.GlyphStyle.Radius = vg.Points(2)
		p.Add(scatter)
		p.Legend.Add(fmt.Sprintf("Class %d", classID), scatter)
	}

	return p.Save(6*vg.Inch, 6*vg.Inch, outputPath)
}

func predict(network []*layers.LayerDense, inputs *mat.Dense) (*mat.Dense, []int, error) {
	output := inputs
	for _, layer := range network {
		next, err := layer.Forward(output)
		if err != nil {
			return nil, nil, err
		}
		output = next
	}

	rows, cols := output.Dims()
	predictions := make([]int, rows)
	for i := 0; i < rows; i++ {
		bestClass := 0
		bestProbability := output.At(i, 0)
		for j := 1; j < cols; j++ {
			if output.At(i, j) > bestProbability {
				bestClass = j
				bestProbability = output.At(i, j)
			}
		}
		predictions[i] = bestClass
	}

	return output, predictions, nil
}

func accuracy(predictions []int, labels []float64) float64 {
	correct := 0
	for i, prediction := range predictions {
		if prediction == int(labels[i]) {
			correct++
		}
	}

	return float64(correct) / float64(len(labels))
}

func plotDecisionBoundary(network []*layers.LayerDense, X *mat.Dense, labels []float64, outputPath string) error {
	const (
		gridSize = 160
		minAxis  = -1.1
		maxAxis  = 1.1
	)

	grid := make([]float64, gridSize*gridSize*2)
	step := (maxAxis - minAxis) / float64(gridSize-1)
	for row := 0; row < gridSize; row++ {
		for col := 0; col < gridSize; col++ {
			idx := row*gridSize + col
			grid[idx*2] = minAxis + float64(col)*step
			grid[idx*2+1] = minAxis + float64(row)*step
		}
	}

	_, predictions, err := predict(network, mat.NewDense(gridSize*gridSize, 2, grid))
	if err != nil {
		return err
	}

	p := plot.New()
	p.Title.Text = "Spiral Decision Boundary"
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"
	p.X.Min = minAxis
	p.X.Max = maxAxis
	p.Y.Min = minAxis
	p.Y.Max = maxAxis

	colors := []color.RGBA{
		{R: 239, G: 68, B: 68, A: 255},
		{R: 59, G: 130, B: 246, A: 255},
		{R: 34, G: 197, B: 94, A: 255},
	}

	boundaryPoints := make(map[int]plotter.XYs)
	for i, prediction := range predictions {
		boundaryPoints[prediction] = append(boundaryPoints[prediction], plotter.XY{
			X: grid[i*2],
			Y: grid[i*2+1],
		})
	}

	for classID, points := range boundaryPoints {
		scatter, err := plotter.NewScatter(points)
		if err != nil {
			return err
		}

		scatter.GlyphStyle.Color = color.RGBA{
			R: colors[classID%len(colors)].R,
			G: colors[classID%len(colors)].G,
			B: colors[classID%len(colors)].B,
			A: 45,
		}
		scatter.GlyphStyle.Radius = vg.Points(1)
		p.Add(scatter)
	}

	classPoints := make(map[int]plotter.XYs)
	rows, _ := X.Dims()
	for i := 0; i < rows; i++ {
		classID := int(labels[i])
		classPoints[classID] = append(classPoints[classID], plotter.XY{
			X: X.At(i, 0),
			Y: X.At(i, 1),
		})
	}

	for classID, points := range classPoints {
		scatter, err := plotter.NewScatter(points)
		if err != nil {
			return err
		}

		scatter.GlyphStyle.Color = colors[classID%len(colors)]
		scatter.GlyphStyle.Radius = vg.Points(2.5)
		p.Add(scatter)
		p.Legend.Add(fmt.Sprintf("Class %d", classID), scatter)
	}

	return p.Save(6*vg.Inch, 6*vg.Inch, outputPath)
}

func main() {
	X, y := buildDataSet()
	if err := plotDataSet(X, y, "spiral.png"); err != nil {
		panic(err)
	}

	l1 := layers.NewLayerDense(64, 2, "relu")
	l2 := layers.NewLayerDense(64, 64, "relu")
	l3 := layers.NewLayerDense(3, 64, "softmax")
	network := []*layers.LayerDense{l1, l2, l3}

	model := model.NewModel(network, "categorical-cross-entropy")

	model.Fit(X, mat.NewVecDense(len(y), y), epochs)

	probabilities, predictions, err := predict(network, X)
	if err != nil {
		panic(err)
	}

	fmt.Printf("training accuracy: %.2f%%\n", accuracy(predictions, y)*100)
	fmt.Println("sample predictions:")
	for i := 0; i < 10; i++ {
		fmt.Printf(
			"point %03d expected=%d predicted=%d probabilities=[%.3f %.3f %.3f]\n",
			i,
			int(y[i]),
			predictions[i],
			probabilities.At(i, 0),
			probabilities.At(i, 1),
			probabilities.At(i, 2),
		)
	}

	if err := plotDecisionBoundary(network, X, y, "spiral_decision_boundary.png"); err != nil {
		panic(err)
	}

	fmt.Fprintln(os.Stderr, "wrote spiral.png and spiral_decision_boundary.png")
}
