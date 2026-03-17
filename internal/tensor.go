package internal

import "fmt"

type Tensor struct {
	Rows int
	Cols int
	Data [][]float64
}

func NewTensor(Rows, Cols int) *Tensor {
	Data := make([][]float64, Rows)
	for i := range Data {
		Data[i] = make([]float64, Cols)
	}

	return &Tensor{
		Rows: Rows,
		Cols: Cols,
		Data: Data,
	}
}

func (t *Tensor) Multiply(other *Tensor) (*Tensor, error) {
	if t.Cols != other.Rows {
		return nil, fmt.Errorf("incompatible dimensions for dot product: %d x %d and %d x %d", t.Rows, t.Cols, other.Rows, other.Cols)
	}

	newTensor := NewTensor(t.Rows, other.Cols)
	for i := 0; i < newTensor.Rows; i++ {
		for j := 0; j < newTensor.Cols; j++ {
			sum := 0.0
			for a := 0; a < t.Cols; a++ {
				sum += (t.Data[i][a] * other.Data[a][j])
			}
			newTensor.Data[i][j] = sum
		}
	}
	return newTensor, nil
}

func (t *Tensor) Sum(other *Tensor) (*Tensor, error) {
	if other.Rows != t.Rows && other.Cols != t.Cols {
		return nil, fmt.Errorf("incompatible dimensions for sum: %d x %d and %d x %d", t.Rows, t.Cols, other.Rows, other.Cols)
	}

	newTensor := NewTensor(t.Rows, t.Cols)

	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			newTensor.Data[i][j] = t.Data[i][j] + other.Data[i][j]
		}
	}

	return newTensor, nil
}

func (t *Tensor) Dot(other *Tensor) (float64, error) {
	if t.Rows != other.Rows || t.Cols != other.Cols {
		return 0.0, fmt.Errorf(
			"incompatible dimensions: %d x %d and %d x %d",
			t.Rows, t.Cols, other.Rows, other.Cols,
		)
	}

	sum := 0.0
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			sum += t.Data[i][j] * other.Data[i][j]
		}
	}

	return sum, nil
}

func (t *Tensor) Transpose() *Tensor {
	newTensor := NewTensor(t.Cols, t.Rows)
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			newTensor.Data[j][i] = t.Data[i][j]
		}
	}

	return newTensor
}

func (t *Tensor) Print() {
	fmt.Printf("The dimesions are %d * %d \n", t.Rows, t.Cols)
	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			fmt.Printf("%f, ", t.Data[i][j])
		}
		fmt.Print("\n")
	}
}
