PyObject *fplib_linmsg(PyObject *self, PyObject *args)
{
  PyObject *xar = NULL;
  PyObject *max_msg = NULL;
  double fill_value;

/*
 *  Output variable.
 */
  double *xlinmsg;
/*
 *  Other variables
 */
  int i, index_x, npts, total_size_x, total_size_x1;
  int ndims_x, *dsizes_x, mflag, nflag;
  double *x;

  PyArrayObject *arr;

/*
 *  Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args, "OiOd:linmsg", &xar, &nflag, 
                          &max_msg, &fill_value)) {
    printf("linmsg: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 *  Extract array information.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromObject \
                        (xar,PyArray_DOUBLE,0,0);
  x = (double *)arr->data;
  ndims_x = arr->nd;
  dsizes_x = (int *) calloc(ndims_x,sizeof(int));
  for(i = 0; i < ndims_x; i++ ) {
    dsizes_x[i] = arr->dimensions[i];
  }
  npts = dsizes_x[ndims_x-1];

/*
 *  Check on max_msg.  If it is any string, then set mflag to npts,
 *  otherwise max_msg must be an integer in which case set
 *  mflag to that.
 */
  if (PyString_Check(max_msg)) {
    mflag = npts;     
  }
  else if (PyInt_Check(max_msg)) {
    mflag = (int) (PyInt_AsLong(max_msg));
  }
  else {
    printf("linmsg: max_msg must be default or an integer\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 *  Compute total size of output array and allocate space
 *  for output array.
 */
  total_size_x1 = 1;
  for( i = 0; i < ndims_x-1; i++ ) {
    total_size_x1 *= dsizes_x[i];
  }
  total_size_x = total_size_x1*npts;
  xlinmsg = (double *) calloc(total_size_x,sizeof(double));
  memcpy(xlinmsg, x, total_size_x*sizeof(double));

/*
 *  Call Fortran.
 */
  index_x = 0;
  for (i = 0; i < total_size_x1; i++) {
    NGCALLF(dlinmsg,DLINMSG) (xlinmsg+index_x, &npts, &fill_value, 
                              &nflag, &mflag);
    index_x += npts;
  }

/*
 *  Return.
 */
  return ((PyObject *) PyArray_FromDimsAndData(ndims_x,dsizes_x,
                                        PyArray_DOUBLE, (void *) xlinmsg));
}
