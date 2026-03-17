package internal

import (
	"testing"
)

func TestMultiply(t *testing.T) {
	tensor1 := NewTensor(2, 3)
	tensor1.Data = [][]float64{
		{1, 2, 3},
		{4, 3, 6},
	}
	tensor2 := NewTensor(3, 4)

	tensor2.Data = [][]float64{
		{2, 6, 3, 2},
		{1, 2, 3, 5},
		{3, 4, 5, 6},
	}

	result, err := tensor1.Multiply(tensor2)
	if err != nil {
		t.Fatalf("unexpected error from Dot: %v", err)
	}

	if result.Rows != 2 || result.Cols != 4 {
		t.Fatalf("unexpected result dimensions: got %d x %d, want %d x %d", result.Rows, result.Cols, 2, 4)
	}

	expected := [][]float64{
		{13, 22, 24, 30},
		{29, 54, 51, 59},
	}

	for i := 0; i < result.Rows; i++ {
		for j := 0; j < result.Cols; j++ {
			if result.Data[i][j] != expected[i][j] {
				t.Fatalf("result.Data[%d][%d] = %v; want %v", i, j, result.Data[i][j], expected[i][j])
			}
		}
	}
}

func TestSum(t *testing.T) {
	tensor1 := NewTensor(2, 2)
	tensor1.Data = [][]float64{
		{1, 2},
		{3, 2},
	}
	tensor2 := NewTensor(2, 2)
	tensor2.Data = [][]float64{
		{3, 2},
		{2, 3},
	}
	result, err := tensor1.Sum(tensor2)

	if err != nil {
		t.Fatalf("unexpected error from Sum: %v", err)
	}
	expected := [][]float64{
		{4, 4},
		{5, 5},
	}

	for i := range expected {
		if len(result.Data[i]) != len(expected[i]) {
			t.Fatalf("expected %d Cols in row %d, got %d",
				len(expected[i]), i, len(result.Data[i]))
		}

		for j := range expected[i] {
			if result.Data[i][j] != expected[i][j] {
				t.Errorf("expected result[%d][%d] = %v, got %v",
					i, j, expected[i][j], result.Data[i][j])
			}
		}
	}

}
