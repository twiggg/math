package mat

//Add returns a new matrix as m+n (element by element)
func Add(m, n *M64) (*M64, error) {
	r, c := m.Dims()
	res := NewM64(r, c, nil)
	if err := add(m, n, res); err != nil {
		return nil, err
	}
	return res, nil
}

//Sub returns a new matrix as m-n (element by element)
func Sub(m, n *M64) (*M64, error) {
	r, c := m.Dims()
	res := NewM64(r, c, nil)
	if err := sub(m, n, res); err != nil {
		return nil, err
	}
	return res, nil
}

//DotProduct returns a new matrix as the dot product of m and n
func DotProduct(m, n *M64) (*M64, error) {
	r, _ := m.Dims()
	_, c1 := n.Dims()
	res := NewM64(r, c1, nil)
	if err := dotprod(m, n, res); err != nil {
		return nil, err
	}
	return res, nil
}

//MapElem applies function fn to each elem of m
func MapElem(m *M64, fn func(x float64) float64) (*M64, error) {
	res := NewM64(m.r, m.c, nil)
	if err := mapElemVal(m, res, fn); err != nil {
		return nil, err
	}
	return res, nil
}
