PyObject *fplib_regline(PyObject *self, PyObject *args)
{
  PyObject *xar = NULL;
  PyObject *yar = NULL;
  PyArrayObject *arr = NULL;
  double *x, *y;
  double fill_value_x, fill_value_y;

/*
 * Output array variables
 */
  double rcoef, tval, rstd, xave, yave, yint;
  int nptxy, ier = 0;
  PyObject *pdict, *rc;

/*
 *  Output variable.
 */
  int npts, dsizes_x[1], dsizes_y[1];

/*
 *  Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args, "OOdd:regline", &xar, &yar, 
                         &fill_value_x, &fill_value_y)) {
    printf("regline: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 *  Extract array information.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromObject \
                            (xar,PyArray_DOUBLE,0,0);
  x = (double *)arr->data;
  dsizes_x[0] = arr->dimensions[0];

  arr = (PyArrayObject *) PyArray_ContiguousFromObject \
                            (yar,PyArray_DOUBLE,0,0);
  y = (double *)arr->data;
  dsizes_y[0] = arr->dimensions[0];

/*
 * The x and y arrays coming in must have the same length.
 */
  if( dsizes_x[0] != dsizes_y[0] ) {
    printf ("regline: The input arrays must be the same length\n");
    Py_INCREF(Py_None);
    return Py_None;
   }

/*
 * Get and check number of input points.
 */
  npts = dsizes_x[0];
  if( npts < 2 ) {
    printf ("regline: The length of x and y must be at least 2\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Call the f77 version of 'regline' with the full argument list.
 */
  NGCALLF(dregcoef,DREGCOEF)(x, y, &npts, &fill_value_x, &fill_value_y,
                             &rcoef, &tval, &nptxy, &xave, &yave, &rstd, &ier);
  if (ier == 5) {
    printf ("regline: The x and/or y array contains all missing values\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if (ier == 6) {
    printf ("regline: The x and/or y array contains less than 3 non-missing values\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

  yint  = yave - rcoef*(xave);

/*
 *  Create return tuple.
 */
  pdict = PyDict_New();
  PyDict_SetItem(pdict, PyString_FromString("fill_value"), PyFloat_FromDouble(fill_value_y));
  PyDict_SetItem(pdict, PyString_FromString("tval"), PyFloat_FromDouble(tval));
  PyDict_SetItem(pdict, PyString_FromString("xave"), PyFloat_FromDouble(xave));
  PyDict_SetItem(pdict, PyString_FromString("yave"), PyFloat_FromDouble(yave));
  PyDict_SetItem(pdict, PyString_FromString("rstd"), PyFloat_FromDouble(rstd));
  PyDict_SetItem(pdict, PyString_FromString("nptxy"), PyInt_FromLong((long) nptxy));
  PyDict_SetItem(pdict, PyString_FromString("yintercept"), PyFloat_FromDouble(yint));

/*
 *  pdict = Py_BuildValue("{s:f, s:f, s:f, s:f, s:f, s:i, s:f}",  
 *                 "fill_value",fill_value_y, 
 *                 "tval",tval, 
 *                 "xave",xave, 
 *                 "yave",yave, 
 *                 "rstd",rstd, 
 *                 "nptxy",nptxy, 
 *                 "yintercept",yint);
 */

  rc = PyFloat_FromDouble(rcoef);
  return t_output_helper(rc,pdict);
}
