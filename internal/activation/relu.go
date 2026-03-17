package activation

func ReLUForward(value float64) float64 {
	if value > 0 {
		return value
	}
	return 0.0
}
