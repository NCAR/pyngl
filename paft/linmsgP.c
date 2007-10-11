PyObject *fplib_linmsg(PyObject *self, PyObject *args)
{
  PyObject *xar = NULL;
  double fill_value;

/*
 *  Output variable.
 */
  double *xlinmsg;
/*
 *  Other variables
 */
  int i, inpts, ndims_x, mflag, nflag, max_msg;
  npy_intp inpy, index_x, npts, total_size_x, total_size_x1, *dsizes_x;
  double *x;

  PyArrayObject *arr;

/*
 *  Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args, "Oiid:linmsg", &xar, &nflag, 
                          &max_msg, &fill_value)) {
    printf("linmsg: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 *  Extract array information.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                        (xar,PyArray_DOUBLE,0,0);
  x = (double *)arr->data;
  ndims_x = arr->nd;
  dsizes_x = (npy_intp *) calloc(ndims_x,sizeof(npy_intp));
  for(i = 0; i < ndims_x; i++ ) {
    dsizes_x[i] = (npy_intp)arr->dimensions[i];
  }
  npts  = dsizes_x[ndims_x-1];
  inpts = (int)npts;   /* inpts may not be big enough to hold value of npts */
/*
 *  Check on max_msg.  If it is zero, then set mflag to the
 *  maximum.
 */
  if (max_msg == 0) {
    mflag = inpts;     
  }
  else {
    mflag = max_msg;
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
  for (inpy = 0; inpy < total_size_x1; inpy++) {
    NGCALLF(dlinmsg,DLINMSG) (xlinmsg+index_x, &inpts, &fill_value, 
                              &nflag, &mflag);
    index_x += npts;
  }

/*
 *  Return.
 */
  return ((PyObject *) PyArray_SimpleNewFromData(ndims_x,dsizes_x,
                                        PyArray_DOUBLE, (void *) xlinmsg));
}
