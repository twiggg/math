package nn

import (
	"fmt"

	"github.com/twiggg/math/mat64"
)

//FeedForward represents a simple feed forward neural network
type FeedForward struct {
	inSize     int
	outSize    int
	layers     []*layer
	states     []*mat.M64
	keepStates bool
}

//NewFeedForward returns a new instance of FeedForward Neural Network, with no layers
func NewFeedForward(inSize int, keepStates bool) (*FeedForward, error) {
	if inSize < 1 {
		return nil, fmt.Errorf("minimum input size is 1")
	}
	ff := &FeedForward{
		inSize:     inSize,
		outSize:    inSize,
		keepStates: keepStates,
	}
	return ff, nil
}

//SetLayers sets neuron layers connected via w,b,fn. Must have at least 1 layer
func (ff *FeedForward) SetLayers(configs ...*LayerConfig) error {
	var err error
	n := len(configs)
	if n < 1 {
		return fmt.Errorf("must have at least one layer")
	}
	n2 := 0
	if ff.keepStates {
		n2 = n
	}
	prevSize := ff.inSize
	layers := make([]*layer, n)

	for i, l := range configs {
		if err = l.Validate(); err != nil {
			return fmt.Errorf("configs[%d]: %s", i, err.Error())
		}
		layers[i] = newLayer(prevSize, l.Size, l.Fn)
		prevSize = l.Size
	}
	ff.layers = layers
	ff.states = make([]*mat.M64, n2)
	return nil
}
