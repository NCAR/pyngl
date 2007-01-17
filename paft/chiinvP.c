PyObject *fplib_chiinv(PyObject *self, PyObject *args)
{
  PyObject *obj1 = NULL ;
  PyObject *obj2 = NULL ;

  int i, tsize, ndims_p, ndims_df, *len_dims_p, *len_dims_df;
  double *p, *df, *rval, *tmp_chi;
 
  PyArrayObject *arr, *aret;

/*
 *  Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args,(char *)"OO:chiinv",&obj1,&obj2)) {
    printf("chiinv: argument parsing failed\n");
    return Py_None;
  }

/*
 *  Extract array information.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromObject \
                        (obj1,PyArray_DOUBLE,0,0);
  p = (double *)arr->data;
  ndims_p = arr->nd;
  len_dims_p = (int *)malloc(ndims_p*sizeof(int));
  for(i = 0; i < ndims_p; i++ ) {
    len_dims_p[i] = arr->dimensions[i];
  }

  arr = (PyArrayObject *) PyArray_ContiguousFromObject \
                        (obj2,PyArray_DOUBLE,0,0);
  df = (double *)arr->data;
  ndims_df = arr->nd;
  len_dims_df = (int *)malloc(ndims_df*sizeof(int));
  for(i = 0; i < ndims_df; i++ ) {
    len_dims_df[i] = arr->dimensions[i];
  }

/*
 * Check dimensions and dimension sizes.
 */
  if (ndims_p != ndims_df) {
    NhlPError(NhlFATAL,NhlEUNKNOWN,"chiinv: The two input arrays must have the same number of dimensions");
    Py_INCREF(Py_None);
    return(Py_None);
  }
  for( i = 0; i < ndims_p; i++ ) {
    if (len_dims_p[i] != len_dims_df[i]) {
      NhlPError(NhlFATAL,NhlEUNKNOWN,"chiinv: The two input arrays must have the same dimension sizes");
      Py_INCREF(Py_None);
      return(Py_None);
    }
  }

/*
 *  Compute total size of output array and allocate space.
 */
  tsize = 1;
  for( i = 0; i < ndims_p; i++ ) {
    tsize *= len_dims_p[i];
  }
  rval = (double *) calloc(tsize,sizeof(double));

/*
 * Call the Fortran.
 */
  tmp_chi = (double *)calloc(1,sizeof(double));
  for (i = 0; i < tsize; i++) {
    NGCALLF(chisub,CHISUB)(&p[i],&df[i],tmp_chi);
    rval[i] = *tmp_chi;
  }

  aret = (PyArrayObject *) PyArray_FromDimsAndData(ndims_p, len_dims_p, PyArray_DOUBLE, (void *) rval);
  PyArray_Return(aret);
   
}

