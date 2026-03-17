package activation

import "math"

func SigmoidForward(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
