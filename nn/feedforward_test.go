package nn

import (
	"fmt"
	"testing"

	mat "github.com/twiggg/math/mat64"
	"github.com/twiggg/math/nn/activation"

	"github.com/twiggg/tester"
)

func TestNewFeedForward(t *testing.T) {
	te := tester.New(t)
	tests := []struct {
		inSize     int
		keepStates bool
		ff         *FeedForward
		err        error
	}{
		{
			inSize:     3,
			keepStates: true,
			ff: &FeedForward{
				inSize:     3,
				outSize:    3,
				keepStates: true,
				layers:     nil,
				states:     nil,
			},
			err: nil,
		},
	}
	for ind, test := range tests {
		ff, err := NewFeedForward(test.inSize, test.keepStates)
		te.CompareError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "network", test.ff, ff)
		}
	}
}

var getFF = func(inSize int, keepStates bool) *FeedForward {
	f, _ := NewFeedForward(3, false)
	return f
}

var getFF2 = func(inSize int, keepStates bool, configs []*LayerConfig) *FeedForward {
	f, _ := NewFeedForward(3, false)
	f.SetLayers(configs...)
	return f
}

func TestSetLayers(t *testing.T) {
	f1 := activation.Sigmoid

	te := tester.New(t)
	tests := []struct {
		ff      *FeedForward
		configs []*LayerConfig
		exp     []*layer
		err     error
	}{
		{
			ff: getFF(3, false),
			configs: []*LayerConfig{
				&LayerConfig{Size: 3, Fn: f1},
			},
			exp: []*layer{
				newLayer(3, 3, f1),
			},
			err: nil,
		},
		{
			ff:      getFF(3, false),
			configs: []*LayerConfig{},
			exp:     nil,
			err:     fmt.Errorf("must have at least one layer"),
		},
	}

	for ind, test := range tests {
		err := test.ff.SetLayers(test.configs...)
		te.CompareError(ind, test.err, err)
		if err == nil {
			l1 := len(test.exp)
			l2 := len(test.ff.layers)
			if l1 != l2 {
				t.Errorf("test %d: expected %d layers received %d", ind, l1, l2)
				continue
			}
			for i, l := range test.ff.layers {
				le := test.exp[i]
				w := l.w
				we := le.w
				b := l.b
				be := le.b

				te.DeepEqual(ind, fmt.Sprintf("layer[%d].w", i), we, w)
				te.DeepEqual(ind, fmt.Sprintf("layer[%d].b", i), be, b)
			}
		}
	}
}

func TestGetState(t *testing.T) {

	te := tester.New(t)
	tests := []struct {
		ff       *FeedForward
		layerInd int
		state    *mat.M64
		err      error
	}{
		{
			ff:       getFF2(3, true, []*LayerConfig{{Size: 3, Fn: activation.Sigmoid}}),
			layerInd: 0,
			state:    mat.NewM64(3, 3, nil),
			err:      nil,
		},
	}
	for ind, test := range tests {
		state, err := test.ff.GetState(test.layerInd)
		te.CompareError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "state", test.state, state)
		}
	}
}
