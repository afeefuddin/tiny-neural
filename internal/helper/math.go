package helper

func Mean(arr []float64) float64 {
	// mean := 0
	var mean float64

	if len(arr) == 0 {
		return 0.0
	}

	for _, val := range arr {
		mean += val
	}

	return mean / float64(len(arr))
}

func Clip(value, min, max float64) float64 {
	if value > max {
		return max
	} else if value < min {
		return min
	} else {
		return value 
	}
}
