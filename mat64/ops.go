package m64

//Add returns a new matrix as m+n (element by element)
func Add(m, n *M64) (*M64, error) {
	res := NewM64(m.r, m.c, nil)
	if err := add(m, n, res); err != nil {
		return nil, err
	}
	return res, nil
}

//Sub returns a new matrix as m-n (element by element)
func Sub(m, n *M64) (*M64, error) {
	res := NewM64(m.r, m.c, nil)
	if err := sub(m, n, res); err != nil {
		return nil, err
	}
	return res, nil
}

func add(m, n, dest *M64) error {
	if err := sameSize(m, n, dest); err != nil {
		return err
	}
	for i := range m.data {
		dest.data[i] = m.data[i] + n.data[i]
	}
	return nil
}

func sub(m, n, dest *M64) error {
	if err := sameSize(m, n, dest); err != nil {
		return err
	}
	for i := range m.data {
		dest.data[i] = m.data[i] - n.data[i]
	}
	return nil
}

func mul(m, n, dest *M64) error {
	if err := dotSize(m, n, dest); err != nil {
		return err
	}
	sum := 0.0
	i, j, k := 0, 0, 0
	for i = 0; i < m.r; i++ {
		for j = 0; j < n.c; j++ {
			sum = 0.0
			for k = 0; k < m.c; k++ {
				sum += m.At(i, k) * m.At(k, j)
			}
			dest.Set(i, j, sum)
		}
	}
	return nil
}

func mapElem(m, n, dest *M64, fn func(valm, valn float64) float64) error {
	if err := sameSize(m, n, dest); err != nil {
		return err
	}
	for i := range m.data {
		dest.data[i] = fn(m.data[i], n.data[i])
	}
	return nil
}

//mapElemIJ sets dest.data[i,j] to fn(i,j)
func mapElemIJ(m, dest *M64, fn func(i, j int) float64) error {
	if err := sameSize2(m, dest); err != nil {
		return err
	}
	for i := 0; i < m.r; i++ {
		for j := 0; j < m.c; j++ {
			dest.Set(i, j, fn(i, j))
		}
	}
	return nil
}

//mapElemVal sets dest.data[i] to fn(m.data[i])
func mapElemVal(m, dest *M64, fn func(val float64) float64) error {
	if err := sameSize2(m, dest); err != nil {
		return err
	}
	for i := range m.data {
		dest.data[i] = fn(m.data[i])
	}
	return nil
}
