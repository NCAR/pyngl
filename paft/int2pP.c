#include <stdio.h>

PyObject *fplib_int2p(PyObject *self, PyObject *args)
{
/*
 * Input array variables
 */
  PyObject *pinar = NULL;
  PyObject *xinar = NULL;
  PyObject *poutar = NULL;
  PyArrayObject *arr = NULL;

  double *pin, *xin, *pout, fill_value_x;
  int linlog;
  double *tmp_pin, *tmp_pout;

  int ndims_pin;
  npy_intp *dsizes_pin;
  int ndims_xin;
  npy_intp *dsizes_xin;
  int ndims_pout;
  npy_intp *dsizes_pout;

/*
 * work arrays
 */
  double *p, *x;
/*
 * output variable 
 */
  double *xout;
  npy_intp size_leftmost, *dsizes_xout, size_xout;

/*
 * Declare various variables for random purposes.
 */
  npy_intp i, j, index_in, index_out;
  npy_intp npin, npout;
  int ier = 0, inpin, inpout;
  int nmiss = 0, nmono = 0;

/*
 *  Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args, "OOOid:int2p", &pinar, &xinar, &poutar, &linlog, &fill_value_x)) {
    printf("int2p: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Retrieve parameters
 *
 * Note that any of the pointer parameters can be set to NULL,
 * which implies you don't care about its value.
 */

/*
 * Read argument #1
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (pinar,PyArray_DOUBLE,0,0);
  pin = (double *)arr->data;
  ndims_pin  = arr->nd;
  dsizes_pin = (npy_intp *) calloc(ndims_pin,sizeof(npy_intp));
  for(i = 0; i < ndims_pin; i++ ) {
    dsizes_pin[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Read argument #2
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (xinar,PyArray_DOUBLE,0,0);
  xin = (double *)arr->data;
  ndims_xin  = arr->nd;
  dsizes_xin = (npy_intp *) calloc(ndims_xin,sizeof(npy_intp));
  for(i = 0; i < ndims_xin; i++ ) {
    dsizes_xin[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Read argument #3
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (poutar,PyArray_DOUBLE,0,0);
  pout = (double *)arr->data;
  ndims_pout  = arr->nd;
  dsizes_pout = (npy_intp *) calloc(ndims_pout,sizeof(npy_intp));
  for(i = 0; i < ndims_pout; i++ ) {
    dsizes_pout[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Check dimension sizes for pin and xin. If any of them are multiple
 * dimensions, then all but the last (rightmost) dimension must be the same.
 */
  npin = dsizes_pin[ndims_pin-1];
  if (npin < 2) {
    printf("int2p: The rightmost dimension of pin must be at least two");
    Py_INCREF(Py_None);
    return Py_None;
  }

  if(ndims_pin != ndims_xin && ndims_pin != 1) {
    printf("int2p: pin must either be a one-dimensional array or an array the same size as xin");
    Py_INCREF(Py_None);
    return Py_None;
  }

  if(ndims_pin == ndims_xin) {
    for( i = 0; i < ndims_pin; i++ ) {
      if (dsizes_pin[i] != dsizes_xin[i]) {
        printf("int2p: If xin and pin have the same number of dimensions, then they must be the same dimension sizes");
	Py_INCREF(Py_None);
	return Py_None;
      }
    }
  }
  else {
    if (dsizes_xin[ndims_xin-1] != npin) {
      printf("int2p: If pin is a one-dimensional array, then it must be the same size as the righmost dimension of xin");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Check dimension sizes for pout. If it is multi-dimensional, then
 * it must have the same rightmost dimensions as xin.
 */
  npout = dsizes_pout[ndims_pout-1];

/*
 * Test dimension sizes.
 */
  if((npin > INT_MAX) || (npout > INT_MAX)){
    printf("int2p: npin and/or npout is greater than INT_MAX");
    Py_INCREF(Py_None);
    return Py_None;
  }
  inpin  = (int) npin;
  inpout = (int) npout;

  if(ndims_pout != ndims_xin && ndims_pout != 1) {
    printf("int2p: pout must either be a one-dimensional array or an array with the same number of dimensions as xin");
    Py_INCREF(Py_None);
    return Py_None;
  }

  if(ndims_pout > 1) {
    for( i = 0; i < ndims_pout-1; i++ ) {
      if (dsizes_pout[i] != dsizes_xin[i]) {
        printf("int2p: If xin and pout have the same number of dimensions, then all but their last dimension must be the same size");
	Py_INCREF(Py_None);
	return Py_None;
      }
    }
  }
/*
 * Calculate the size of the leftmost dimensions of xin (if any).
 */
  size_leftmost = 1;
  for( i = 0; i < ndims_xin-1; i++ ) size_leftmost *= dsizes_xin[i];

/*
 * Allocate space for output.
 */
  dsizes_xout = (ng_size_t*)calloc(ndims_xin,sizeof(ng_size_t));
  if(dsizes_xout == NULL) {
    printf("int2p: Unable to allocate memory for holding dimension sizes");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_xin-1; i++ ) {
    dsizes_xout[i] = dsizes_xin[i];
  }
  dsizes_xout[ndims_xin-1] = npout;
  size_xout = size_leftmost * npout;

/*
 * Allocate space for output and work arrays.
 */
  xout  = (double*)calloc(size_xout,sizeof(double));
  p     = (double*)calloc(npin,sizeof(double));
  x     = (double*)calloc(npin,sizeof(double));
  if (xout == NULL || p == NULL || x == NULL) {
    printf("int2p: Unable to allocate space for output and/or work arrays\n" );
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Call the Fortran version of this routine.
 */
  index_in = index_out = 0;
  if(ndims_pin  == 1) tmp_pin  = pin;
  if(ndims_pout == 1) tmp_pout = pout;

  for( i = 0; i < size_leftmost; i++ ) {
    if(ndims_pin > 1) {
      tmp_pin = &pin[index_in];
    }
    if(ndims_pout > 1) {
      tmp_pout = &pin[index_out];
    }

    NGCALLF(dint2p,DINT2P)(tmp_pin,&xin[index_in],&p[0],&x[0],&inpin,
                           tmp_pout,&xout[index_out],&inpout,&linlog,
                           &fill_value_x,&ier);
    if (ier) {
      if (ier >= 1000) nmiss++;
      else             nmono++;
      for (j=0; j < npout; j++) xout[index_out+j] = fill_value_x;
    }
    index_in  += npin;
    index_out += npout;
  }
  if (nmiss) {
    printf("int2p: %d input array(s) contained all missing data. No interpolation performed on these arrays",nmiss);
  }
  if (nmono) {
    printf("int2p: %d pin array(s) were in a different order than the corresponding pout array(s). No interpolation performed on these arrays",nmono);
  }
/*
 * Free memory.
 */
  free(p);
  free(x);

/*
 * Return value.
 */
  return ((PyObject *) PyArray_SimpleNewFromData(ndims_xin,dsizes_xout,
                                                 PyArray_DOUBLE,
                                                 (void *) xout));

  free(dsizes_xout);
}

