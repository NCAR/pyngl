PyObject *fplib_wrf_tk(PyObject *self, PyObject *args)
{
  PyObject *par = NULL;
  PyObject *tar = NULL;
  double *p = NULL;
  double *theta = NULL;
  PyArrayObject *arr = NULL;
  double *tk;
/*
 * Various
 */
  int ndims_p, inx;
  npy_intp i, nx, *dsizes_p, size_leftmost, size_tk, index_p;

  if (!PyArg_ParseTuple(args, "OO:wrf_tk", &par, &tar)) {
    printf("wrf_tk: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 *  Extract array information.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (par,PyArray_DOUBLE,0,0);
  p        = (double *)arr->data;
  ndims_p  = arr->nd;
  dsizes_p = (npy_intp *) calloc(ndims_p,sizeof(npy_intp));
  for(i = 0; i < ndims_p; i++ ) {
    dsizes_p[i] = (npy_intp)arr->dimensions[i];
  }

  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (tar,PyArray_DOUBLE,0,0);
  theta = (double *)arr->data;
/*
 * Error checking. Input variables must be same size.
 */
  if(ndims_p != arr->nd) {
    printf("wrf_tk: p and theta must be the same dimensionality");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_p; i++) {
    if(dsizes_p[i] != (npy_intp)arr->dimensions[i]) {
      printf("wrf_tk: p and theta must be the same dimensionality");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Calculate size of leftmost dimensions.
 */
  size_leftmost = 1;
  for(i = 0; i < ndims_p-1; i++) size_leftmost *= dsizes_p[i];
  nx      = dsizes_p[ndims_p-1];

/*
 * Test dimension sizes.
 */
  if(nx > INT_MAX) {
    printf("wrf_tk: nx = %ld is greater than INT_MAX", nx);
    Py_INCREF(Py_None);
    return Py_None;
  }
  inx = (int) nx;

  size_tk = size_leftmost * nx;

  tk = (double *)calloc(size_tk,sizeof(double));
  if(tk == NULL) {
    printf("wrf_tk: Unable to allocate memory for output array");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Loop across leftmost dimensions and call the Fortran routine for each
 * one-dimensional subsection.
 */
  index_p = 0;
  for(i = 0; i < size_leftmost; i++) {
    NGCALLF(dcomputetk,DCOMPUTETK)(&tk[index_p],&p[index_p],
				   &theta[index_p],&inx);
    index_p += nx;    /* Increment index */
  }
  return ((PyObject *) PyArray_SimpleNewFromData(ndims_p,dsizes_p,
                                                 PyArray_DOUBLE,
                                                 (void *) tk));
}


PyObject *fplib_wrf_rh(PyObject *self, PyObject *args)
{
  PyObject *qvar = NULL;
  PyObject *tar = NULL;
  PyObject *par = NULL;
  double *qv = NULL;
  double *p = NULL;
  double *t = NULL;
  PyArrayObject *arr = NULL;
  double *rh;
/*
 * Various
 */
  int ndims_qv, inx;
  npy_intp i, nx, size_leftmost, size_rh, index_qv, *dsizes_qv;

  if (!PyArg_ParseTuple(args, "OOO:wrf_rh", &qvar, &par, &tar)) {
    printf("wrf_rh: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 *  Extract qv
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (qvar,PyArray_DOUBLE,0,0);
  qv        = (double *)arr->data;
  ndims_qv  = arr->nd;
  dsizes_qv = (npy_intp *) calloc(ndims_qv,sizeof(npy_intp));
  for(i = 0; i < ndims_qv; i++ ) {
    dsizes_qv[i] = (npy_intp)arr->dimensions[i];
  }

/*
 *  Extract p
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (par,PyArray_DOUBLE,0,0);
  p = (double *)arr->data;

/*
 * Error checking. Input variables must be same size.
 */
  if(ndims_qv != arr->nd) {
    printf("wrf_rh: qv, p, t must be the same dimensionality");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_qv; i++) {
    if(dsizes_qv[i] != (npy_intp)arr->dimensions[i]) {
      printf("wrf_rh: qv, p, t must be the same dimensionality");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 *  Extract t
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (tar,PyArray_DOUBLE,0,0);
  t = (double *)arr->data;

/*
 * Error checking. Input variables must be same size.
 */
  if(ndims_qv != arr->nd) {
    printf("wrf_rh: qv, p, t must be the same dimensionality");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_qv; i++) {
    if(dsizes_qv[i] != (npy_intp)arr->dimensions[i]) {
      printf("wrf_rh: qv, p, t must be the same dimensionality");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Calculate size of leftmost dimensions.
 */
  size_leftmost = 1;
  for(i = 0; i < ndims_qv-1; i++) size_leftmost *= dsizes_qv[i];
  nx      = dsizes_qv[ndims_qv-1];
/*
 * Test dimension sizes.
 */
  if(nx > INT_MAX) {
    printf("wrf_rh: nx = %ld is greater than INT_MAX", nx);
    Py_INCREF(Py_None);
    return Py_None;
  }
  inx = (int) nx;

  size_rh = size_leftmost * nx;

  rh = (double *)calloc(size_rh,sizeof(double));
  if(rh == NULL) {
    printf("wrf_rh: Unable to allocate memory for output array");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Loop across leftmost dimensions and call the Fortran routine for each
 * one-dimensional subsection.
 */
  index_qv = 0;
  for(i = 0; i < size_leftmost; i++) {
    NGCALLF(dcomputerh,DCOMPUTERH)(&qv[index_qv],&p[index_qv],
				   &t[index_qv],&rh[index_qv],&inx);
    index_qv += nx;    /* Increment index */
  }
  return ((PyObject *) PyArray_SimpleNewFromData(ndims_qv,dsizes_qv,
                                                 PyArray_DOUBLE,
                                                 (void *) rh));
}


PyObject *fplib_wrf_dbz(PyObject *self, PyObject *args)
{
  PyObject *par = NULL;
  PyObject *tar = NULL;
  PyObject *qvar = NULL;
  PyObject *qrar = NULL;
  PyObject *qsar = NULL;
  PyObject *qgar = NULL;
  double *p = NULL;
  double *t = NULL;
  double *qv = NULL;
  double *qr = NULL;
  double *qs = NULL;
  double *qg = NULL;
  double *tmp_qg, *tmp_qs;
  int ivarint, iliqskin;
  PyArrayObject *arr = NULL;
  double *dbz;
  int ndims_p, ndims_t, ndims_qv, ndims_qr, ndims_qs, ndims_qg;
  npy_intp *dsizes_p, *dsizes_t, *dsizes_qv;
  npy_intp *dsizes_qr, *dsizes_qs, *dsizes_qg;
/*
 *
 */
  npy_intp btdim, sndim, wedim, nbtsnwe, index_dbz;
  npy_intp i, j, size_leftmost, size_output;
  int iwedim, isndim, ibtdim, sn0 = 0;
  int is_scalar_qs, is_scalar_qg;

  if (!PyArg_ParseTuple(args, "OOOOOOii:wrf_dbz", &par, &tar, &qvar, 
			&qrar, &qsar, &qgar, &ivarint, &iliqskin)) {
    printf("wrf_dbz: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Extract array information.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (par,PyArray_DOUBLE,0,0);
  p        = (double *)arr->data;
  ndims_p  = arr->nd;
  dsizes_p = (npy_intp *) calloc(ndims_p,sizeof(npy_intp));
  for(i = 0; i < ndims_p; i++ ) {
    dsizes_p[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Check dimension sizes.
 */
  if(ndims_p < 3) {
    printf("wrf_dbz: The p array must have at least 3 dimensions");
    Py_INCREF(Py_None);
    return Py_None;
  }
  btdim = dsizes_p[ndims_p-3];
  sndim = dsizes_p[ndims_p-2];
  wedim = dsizes_p[ndims_p-1];

/*
 * Test dimension sizes.
 */
  if((wedim > INT_MAX) || (sndim > INT_MAX) || (btdim > INT_MAX)) {
    printf("wrf_dbz: one or more dimension sizes is greater than INT_MAX");    
    Py_INCREF(Py_None);
    return Py_None;
  }
  iwedim = (int) wedim;
  isndim = (int) sndim;
  ibtdim = (int) btdim;

  nbtsnwe = btdim * sndim * wedim;

/*
 * Extract t
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (tar,PyArray_DOUBLE,0,0);
  t        = (double *)arr->data;
  ndims_t  = arr->nd;
  dsizes_t = (npy_intp *) calloc(ndims_t,sizeof(npy_intp));
  for(i = 0; i < ndims_t; i++ ) {
    dsizes_t[i] = (npy_intp)arr->dimensions[i];
  }

  if(ndims_t != ndims_p) {
    printf("wrf_dbz: The t array must have the same number of dimensions as the p array");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Extract qv
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (qvar,PyArray_DOUBLE,0,0);
  qv        = (double *)arr->data;
  ndims_qv  = arr->nd;
  dsizes_qv = (npy_intp *) calloc(ndims_qv,sizeof(npy_intp));
  for(i = 0; i < ndims_qv; i++ ) {
    dsizes_qv[i] = (npy_intp)arr->dimensions[i];
  }
  if(ndims_qv != ndims_p) {
    printf("wrf_dbz: The qv array must have the same number of dimensions as the p array");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Extract qr
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (qrar,PyArray_DOUBLE,0,0);
  qr = (double *)arr->data;
  ndims_qr  = arr->nd;
  dsizes_qr = (npy_intp *) calloc(ndims_qr,sizeof(npy_intp));
  for(i = 0; i < ndims_qr; i++ ) {
    dsizes_qr[i] = (npy_intp)arr->dimensions[i];
  }

  if(ndims_qr != ndims_p) {
    printf("wrf_dbz: The qr array must have the same number of dimensions as the p array");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Extract qs. This is an optional argument, so it will have been
 * set to a scalar by the calling routine.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (qsar,PyArray_DOUBLE,0,0);
  qs        = (double *)arr->data;
  ndims_qs  = arr->nd;
  dsizes_qs = (npy_intp *) calloc(ndims_qs,sizeof(npy_intp));
  for(i = 0; i < ndims_qs; i++ ) {
    dsizes_qs[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Check dimension sizes.
 */
  is_scalar_qs = is_scalar(ndims_qs,dsizes_qs);
  if(!is_scalar_qs && ndims_qs != ndims_p) {
    printf("wrf_dbz: qs must either be a scalar or have the same number of dimensions as the p array");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Extract qg
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (qgar,PyArray_DOUBLE,0,0);
  qg        = (double *)arr->data;
  ndims_qg  = arr->nd;
  dsizes_qg = (npy_intp *) calloc(ndims_qg,sizeof(npy_intp));
  for(i = 0; i < ndims_qg; i++ ) {
    dsizes_qg[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Check dimension sizes.
 */
  is_scalar_qg = is_scalar(ndims_qg,dsizes_qg);
  if(!is_scalar_qg && ndims_qg != ndims_p) {
    printf("wrf_dbz: qg must either be a scalar or have the same number of dimensions as the p array");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Check that the first 6 input arrays all have the same dimensionality.
 */
  for(i = 0; i < ndims_p; i++) {
    if( dsizes_t[i] != dsizes_p[i] || 
       dsizes_qv[i] != dsizes_p[i] || 
       dsizes_qr[i] != dsizes_p[i] || 
      (!is_scalar_qs && dsizes_qs[i] != dsizes_p[i]) || 
      (!is_scalar_qg && dsizes_qg[i] != dsizes_p[i])) {
      printf("wrf_dbz: The p, t, qv, qr, qs, and qg arrays must have the same dimensions (qs and qg are optional)");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Calculate size of leftmost dimensions.
 */
  size_leftmost  = 1;
  for(i = 0; i < ndims_p-3; i++) size_leftmost *= dsizes_p[i];

/*
 * If qs and/or qg are scalars, we need to propagate them to full
 * arrays. We'll do this later inside the do loop where the Fortran
 * routine is called.  Allocate the space for that here.
 */
  if(is_scalar_qs) {
    tmp_qs = (double *)calloc(nbtsnwe,sizeof(double));
    if(tmp_qs == NULL) {
      printf("wrf_dbz: Unable to allocate memory for promoting qs array");
      Py_INCREF(Py_None);
      return Py_None;
    }
    if(qs[0] == 0.) sn0 = 0;
    else            sn0 = 1;
    for(j = 0; j < nbtsnwe; j++) tmp_qs[j] = qs[0];
  }
  if(is_scalar_qg) {
    tmp_qg = (double *)calloc(nbtsnwe,sizeof(double));
    if(tmp_qg == NULL) {
      printf("wrf_dbz: Unable to allocate memory for promoting qg array");
      Py_INCREF(Py_None);
      return Py_None;
    }
    for(i = 0; i < nbtsnwe; i++) tmp_qg[i] = qg[0];
  }

/* 
 * Allocate space for output array.
 */
  size_output = size_leftmost * nbtsnwe;

  dbz = (double *)calloc(size_output, sizeof(double));
  if(dbz == NULL) {
    printf("wrf_dbz: Unable to allocate memory for output array");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Loop across leftmost dimensions and call the Fortran routine for each
 * subsection of the input arrays.
 */
  index_dbz = 0;
  for(i = 0; i < size_leftmost; i++) {
    if(!is_scalar_qg) {
      tmp_qg = &qs[index_dbz];
    }

/*
 * Check values for qs array. If all zero, then set sn0 to 0. Otherwise
 * set sn0 to 1.
 */
    if(!is_scalar_qs) {
      j   = 0;
      sn0 = 0;
      while( (j < nbtsnwe) && !sn0) {
	if(qs[index_dbz+j] != 0.) sn0 = 1;
        j++;
      }
      tmp_qs = &qs[index_dbz];
    }
/*
 * Call the Fortran routine.
 */
    NGCALLF(calcdbz,CALCDBZ)(&dbz[index_dbz], &p[index_dbz], &t[index_dbz], 
			     &qv[index_dbz], &qr[index_dbz], tmp_qs, tmp_qg,
			     &iwedim,&isndim, &ibtdim, &sn0, 
			     &ivarint, &iliqskin);
    index_dbz += nbtsnwe;
  }

/*
 * Free unneeded memory.
 */
  if(is_scalar_qs) free(tmp_qs);
  if(is_scalar_qg) free(tmp_qg);
 
  return ((PyObject *) PyArray_SimpleNewFromData(ndims_p,dsizes_p,
                                                 PyArray_DOUBLE,
                                                 (void *) dbz));
}

/*
 * Checks if a variable is a scalar or not.
 * Returns 1 if it is, and a 0 if it isn't.
 */
int is_scalar(
int        ndims_x,
npy_intp *dsizes_x
)
{
  if(ndims_x == 1 && dsizes_x[0] == 1) return(1);
  else                                 return(0);
}

