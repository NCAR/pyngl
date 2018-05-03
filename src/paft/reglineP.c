PyObject *fplib_regline(PyObject *self, PyObject *args)
{
  PyObject *xar = NULL;
  PyObject *yar = NULL;
  PyArrayObject *arr = NULL;
  double *x, *y;
  double fill_value_x, fill_value_y;
  int return_info;

/*
 * Output variables
 */
  double *rcoef, tval, rstd, xave, yave, yint;
  int inpts, nptxy, ier = 0;
  PyObject *pdict, *rc, *result;
  npy_intp npts, dsizes_x[1], dsizes_y[1], dsizes_rcoef[1];

/*
 *  Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args, "OOddi:regline", &xar, &yar, 
                         &fill_value_x, &fill_value_y, &return_info)) {
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 *  Extract array information.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (xar,PyArray_DOUBLE,0,0);
  x = (double *)arr->data;
  dsizes_x[0] = arr->dimensions[0];

  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (yar,PyArray_DOUBLE,0,0);
  y = (double *)arr->data;
  dsizes_y[0] = arr->dimensions[0];

/*
 * The x and y arrays coming in must have the same length.
 */
  if( dsizes_x[0] != dsizes_y[0] ) {
    PyErr_SetString(PyExc_StandardError, "regline: The input arrays must be the same length.");
    Py_INCREF(Py_None);
    return Py_None;
   }

/*
 * Get and check number of input points.
 */
  npts  = dsizes_x[0];
  inpts = (int)npts;   /* inpts may not be big enough to hold value of npts */
  if( npts < 2 ) {
    PyErr_SetString(PyExc_StandardError, "regline: The length of x and y must be at least 2.");
    Py_INCREF(Py_None);
    return Py_None;
  }

  rcoef = (double *)malloc(sizeof(double));

/*
 * Call the f77 version of 'regline' with the full argument list.
 */
  NGCALLF(dregcoef,DREGCOEF)(x, y, &inpts, &fill_value_x, &fill_value_y,
                             rcoef, &tval, &nptxy, &xave, &yave, &rstd, &ier);
  if (ier == 5) {
    PyErr_SetString(PyExc_StandardError, "regline: The x and/or y array contains all missing values.");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if (ier == 6) {
    PyErr_SetString(PyExc_StandardError, "regline: The x and/or y array contains less than 3 non-missing values.");
    Py_INCREF(Py_None);
    return Py_None;
  }

  yint  = yave - *rcoef*(xave);

/*
 * Return extra calculations only if return_info is True (1).
 */
  if(return_info) {
/*
 *  Create return tuple.
 */
    rc = PyFloat_FromDouble(*rcoef);

    pdict = PyDict_New();
    PyDict_SetItem(pdict, PyString_FromString("xave"), PyFloat_FromDouble(xave));
    PyDict_SetItem(pdict, PyString_FromString("yave"), PyFloat_FromDouble(yave));
    PyDict_SetItem(pdict, PyString_FromString("tval"), PyFloat_FromDouble(tval));
    PyDict_SetItem(pdict, PyString_FromString("rstd"), PyFloat_FromDouble(rstd));
    PyDict_SetItem(pdict, PyString_FromString("yintercept"), PyFloat_FromDouble(yint));
    PyDict_SetItem(pdict, PyString_FromString("nptxy"), PyInt_FromLong((long) nptxy));

/*
 *  pdict = Py_BuildValue("{s:f, s:f, s:f, s:f, s:f, s:i}",
 *                 "xave",xave, 
 *                 "yave",yave, 
 *                 "tval",tval, 
 *                 "rstd",rstd, 
 *                 "yintercept",yint,
 *                 "nptxy",nptxy);
 */
    result = Py_None;
    result = t_output_helper(result,rc);
    result = t_output_helper(result,pdict);
    if (result == Py_None) Py_INCREF(Py_None);
    return result;
  }
  else {
    dsizes_rcoef[0] = 1;
    return ((PyObject *) PyArray_SimpleNewFromData(1,dsizes_rcoef,
  						   PyArray_DOUBLE,
						   (void *) rcoef));
  }
}
