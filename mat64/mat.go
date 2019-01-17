package m64

import (
	"fmt"
	//mat "gonum.org/v1/gonum/mat"
)

/*
func useless() *mat.Dense {
	return mat.NewDense()
}
*/

//M64 represents a float64 matrix with r rows and c colomns
type M64 struct {
	r    int
	c    int
	data []float64
}

//NewM64 returns a new M64 instance, initialized with data if len==r*c
func NewM64(r, c int, data []float64) (*M64, error) {
	if r <= 0 {
		return nil, fmt.Errorf("r must be >=1")
	}
	if c <= 0 {
		return nil, fmt.Errorf("c must be >=1")
	}
	m := &M64{r: r, c: c}
	if len(data) == r*c {
		m.data = data
	} else {
		m.data = make([]float64, int(r*c))
	}
	return m, nil
}

//Valid returns false if m is nil, and initiates with empty data of size=r*c if invalid size
func (m *M64) Valid() bool {
	if m == nil {
		return false
	}
	if m.r <= 0 {
		m.r = 1
	}
	if m.c <= 0 {
		m.c = 1
	}
	s := m.r * m.c
	if len(m.data) != s {
		m.data = make([]float64, s)
	}
	return true
}

func (m *M64) index(i, j int) int {
	return m.c*i + j
}

//At returns the value at position row=i,col=j. panics if m is nil or index out of range
func (m *M64) At(i, j int) float64 {
	return m.data[m.index(i, j)]
}

//Set sets val at position row=i,col=j. panics if m is nil or index out of range
func (m *M64) Set(i, j int, val float64) {
	m.data[m.index(i, j)] = val
}

//Add adds n to m (element by element)
func (m *M64) Add(n *M64) error {
	return add(m, n, m)
}

//Sub substracts n to m (element by element)
func (m *M64) Sub(n *M64) error {
	return sub(m, n, m)
}
