package nn

import (
	"fmt"

	mat "github.com/twiggg/math/mat64"
)

//ActivationFunc signature
type ActivationFunc func(x float64) float64

//LayerConfig holds info to define a new layer
type LayerConfig struct {
	//InSize  int
	Size int
	Fn   ActivationFunc
}

//Validate checks configuration data
func (l *LayerConfig) Validate() error {
	if l == nil {
		return fmt.Errorf("level config is nil")
	}
	if l.Size <= 0 {
		return fmt.Errorf("size must be >0")
	}
	return nil
}

//newLayer returns a new Level
func newLayer(inSize int, outSize int, fn ActivationFunc) *layer {
	return &layer{
		inSize:  inSize,
		outSize: outSize,
		w:       mat.NewM64(outSize, inSize, nil),
		b:       mat.NewM64(outSize, 1, nil),
		fn:      fn,
	}
}

//layer represents a layer of neurons, defined by Y=fn(w*X+b) where X is the input, Y the output,fn the activation function, W the weights matrix and b the bias.
type layer struct {
	inSize  int
	outSize int
	w       *mat.M64
	b       *mat.M64
	fn      ActivationFunc
}

func (l *layer) dataSize() int {
	//should be the size of w + size of b
	//which is out*in +out*1=out*(in+1)
	if l == nil {
		return 0
	}
	return l.outSize * (l.inSize + 1) //l.w.Size() + l.b.Size()
}

func (l *layer) UpdateData(data []float64) error {
	if l == nil {
		return fmt.Errorf("layer is nil")
	}
	size := l.dataSize()
	if len(data) != size {
		return fmt.Errorf("expected %d values", size)
	}
	//each group of <in+1> values is <in> values for 1 line of w and <1> value for b
	ind := 0
	i, j := 0, 0
	for i = 0; i < l.outSize; i++ {
		for j = 0; j < l.inSize; j++ {
			l.w.Set(i, j, data[ind])
			ind++
			if j == l.inSize-1 {
				l.b.Set(i, 0, data[ind])
				ind++
			}
		}
	}

	return nil
}

func (l *layer) ComputeWith(input *mat.M64) (*mat.M64, error) {
	res, err := wxpb(l.w, input, l.b)
	if err != nil {
		return nil, err
	}
	return mat.MapElem(res, l.fn)
}

//wxpb computes the dot product of w and x then adds b
func wxpb(w, x, b *mat.M64) (*mat.M64, error) {
	res, err := mat.DotProduct(w, x)
	if err != nil {
		return nil, fmt.Errorf("w*x failed: %s", err.Error())
	}
	res, err = mat.Add(res, b)
	if err != nil {
		return nil, fmt.Errorf("w*x +b failed: %s", err.Error())
	}

	return res, nil
}
