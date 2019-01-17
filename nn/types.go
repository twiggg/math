package nn

//ActivationFunc signature
type ActivationFunc func(x float64) float64

//FeedForward represents a simple feed forward neural network
type FeedForward struct {
	inSize  int
	outSize int
	levels  []Level
}

//Level represents connections between two layers of neurons, defined by W (weights), b (bias) and f (activation function).
//
type Level struct {
}
