extern void NGCALLF(chisub,CHISUB)(double *, double *, double *);

PyObject *fplib_chiinv(PyObject *self, PyObject *args)
{
/*
 * Argument # 0
 */
  PyObject *arr_p = NULL;
  double *p;
  int ndims_p;
  npy_intp *dsizes_p;
/*
 * Argument # 1
 */
  PyObject *arr_df = NULL;
  double *df;
  int ndims_df;
  npy_intp *dsizes_df;
/*
 * Return variable
 */
  PyArrayObject *ret_chi;
  double *chi;

/*
 * Define some PyArray objects for input and output.
 */
  PyArrayObject *arr;

/*
 * Various
 */
  npy_intp np, inpy;
  npy_intp size_leftmost, size_chi;
  int i, ndims_leftmost;

/*
 * Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args,(char *)"OO:chiinv",&arr_p,&arr_df)) {
    printf("chiinv: fatal: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Start extracting array information.
 */
/*
 * Get argument # 0
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_p,PyArray_DOUBLE,0,0);
  p        = (double *)arr->data;
  ndims_p  = arr->nd;
  dsizes_p = (npy_intp *)malloc(ndims_p * sizeof(npy_intp));
  for(i = 0; i < ndims_p; i++ ) dsizes_p[i] = (npy_intp)arr->dimensions[i];

/*
 * Check dimension sizes.
 */
  if(ndims_p < 1) {
    printf("chiinv: fatal: The p array must have at least 1 dimension\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  np = dsizes_p[ndims_p-1];
/*
 * Get argument # 1
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromObject
                        (arr_df,PyArray_DOUBLE,0,0);
  df        = (double *)arr->data;
  ndims_df  = arr->nd;
  dsizes_df = (npy_intp *)malloc(ndims_df * sizeof(npy_intp));
  for(i = 0; i < ndims_df; i++ ) dsizes_df[i] = (npy_intp)arr->dimensions[i];

/*
 * Check dimension sizes.
 */
  if(ndims_df < 1) {
    printf("chiinv: fatal: The df array must have at least 1 dimension\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(dsizes_df[ndims_df-1] != np) {
    printf("chiinv: fatal: The ndims-1 dimension of df must be of length np\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Calculate size of leftmost dimensions.
 */
  size_leftmost  = 1;
  ndims_leftmost = ndims_p-1;
  for(i = 0; i < ndims_leftmost; i++) {
    if(dsizes_df[i] != dsizes_p[i]) {
      printf("chiinv: fatal: The leftmost dimensions of p and df must be the same\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
    size_leftmost *= dsizes_p[i];
  }


/*
 * Calculate size of output array.
 */
  size_chi = size_leftmost * np;

/* 
 * Allocate space for output array.
 */
  chi = (double *)calloc(size_chi, sizeof(double));
  if(chi == NULL) {
    printf("chiinv: fatal: Unable to allocate memory for output array\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Loop across leftmost dimensions and call the Fortran routine for each
 * subsection of the input arrays.
 */
  for(inpy = 0; inpy < size_leftmost; inpy++) {
/*
 * Loop across leftmost dimensions and call the Fortran routine.
 */
    NGCALLF(chisub,CHISUB)(&p[i], &df[i], &chi[i]);
  }

/*
 * Return value back to PyNGL script.
 */
  ret_chi = (PyArrayObject *) PyArray_SimpleNewFromData(ndims_p,dsizes_p,
                                                 PyArray_DOUBLE, (void *) chi);
  PyArray_Return(ret_chi);
}
