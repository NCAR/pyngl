PyObject *fplib_betainc(PyObject *self, PyObject *args)
{
  PyObject *xar = NULL;
  PyObject *aar = NULL;
  PyObject *bar = NULL;
  PyArrayObject *arr = NULL;
  double *x, *a, *b;
  double fill_value_x;

/*
 * Output variables
 */
  double *alpha;
  int i, ndims_x, ndims_a, ndims_b, size_alpha;
  npy_intp *dsizes_x, *dsizes_a, *dsizes_b;

/*
 *  Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args, "OOOd:betainc", &xar, &aar, &bar, &fill_value_x)) {
    printf("betainc: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 *  Extract array information.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (xar,PyArray_DOUBLE,0,0);
  x = (double *)arr->data;
  ndims_x  = arr->nd;
  dsizes_x = (npy_intp *) calloc(ndims_x,sizeof(npy_intp));
  for(i = 0; i < ndims_x; i++ ) {
    dsizes_x[i] = (npy_intp)arr->dimensions[i];
  }

  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (aar,PyArray_DOUBLE,0,0);
  a = (double *)arr->data;
  ndims_a  = arr->nd;
  dsizes_a = (npy_intp *) calloc(ndims_a,sizeof(npy_intp));
  for(i = 0; i < ndims_a; i++ ) {
    dsizes_a[i] = (npy_intp)arr->dimensions[i];
  }

  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (bar,PyArray_DOUBLE,0,0);
  b = (double *)arr->data;
  ndims_b  = arr->nd;
  dsizes_b = (npy_intp *) calloc(ndims_b,sizeof(npy_intp));
  for(i = 0; i < ndims_b; i++ ) {
    dsizes_b[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * The x, a, and b arrays must be the same dimension sizes.
 */
  if (ndims_x != ndims_a || ndims_x != ndims_b) {
    printf("betainc: The three input arrays must have the same number of dimensions");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for( i = 0; i < ndims_x; i++ ) {
    if (dsizes_x[i] != dsizes_a[i] || dsizes_x[i] != dsizes_b[i]) {
      printf("betainc: The three input arrays must have the same dimension sizes");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 *  Compute total size of output array and allocate space for it.
 */
  size_alpha = 1;
  for( i = 0; i < ndims_x; i++ ) {
    size_alpha *= dsizes_x[i];
  }
  alpha = (double *) calloc(size_alpha,sizeof(double));
  if(alpha == NULL ) {
    printf("betainc: Unable to allocate memory for output array");
    Py_INCREF(Py_None);
    return Py_None;
  }

  for( i = 0; i < size_alpha; i++ ) {
    if(x[i] == fill_value_x) {
      alpha[i] = fill_value_x;
    }
    else {
/*
 * Call the f77 version of 'betainc' if x[i] is not missing
 */
      NGCALLF(betainc,BETAINC)(&x[i],&a[i],&b[i],&alpha[i]);
    }
  }
  return ((PyObject *) PyArray_SimpleNewFromData(ndims_x,dsizes_x,
                                                 PyArray_DOUBLE,
                                                 (void *) alpha));
}
