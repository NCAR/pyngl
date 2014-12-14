PyObject *fplib_wrf_avo(PyObject *self, PyObject *args)
{
  PyArrayObject *arr = NULL;
/*
 * Input variables
 */
  double *u; 
  PyObject *uar = NULL;
  int ndims_u;
  npy_intp *dsizes_u;

  double *v; 
  PyObject *var = NULL;
  int ndims_v;
  npy_intp *dsizes_v;

  double *msfu; 
  PyObject *msfuar = NULL;
  int ndims_msfu;
  npy_intp *dsizes_msfu;

  double *msfv; 
  PyObject *msfvar = NULL;
  int ndims_msfv;
  npy_intp *dsizes_msfv;

  double *msft; 
  PyObject *msftar = NULL;
  int ndims_msft;
  npy_intp *dsizes_msft;

  double *cor; 
  PyObject *corar = NULL;
  int ndims_cor;
  npy_intp *dsizes_cor;

  double *dx; 
  PyObject *dxar = NULL;
  int ndims_dx;
  npy_intp *dsizes_dx;

  double *dy; 
  PyObject *dyar = NULL;
  int ndims_dy;
  npy_intp *dsizes_dy;

  int *opt;

/*
 * Return variable
 */
  double *av;
  npy_intp *dsizes_av;

/*
 * Various
 */
  npy_intp nx, ny, nz, nxp1, nyp1;
  npy_intp nznynxp1, nznyp1nx, nznynx, nynxp1, nyp1nx, nynx;
  npy_intp i, size_av, size_leftmost;
  npy_intp index_u, index_v, index_msfu, index_msfv, index_msft, index_av;
  int inx, iny, inz, inxp1, inyp1;

  if (!PyArg_ParseTuple(args, "OOOOOOOOi:wrf_avo", &uar, &var, &msfuar, 
                        &msfvar,&msftar, &corar, &dxar, &dyar, &opt)) {
    printf("wrf_avo: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Retrieve parameters.
 *
 * Note any of the pointer parameters can be set to NULL, which
 * implies you don't care about its value.
 */
/*
 * Extract u
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (uar,PyArray_DOUBLE,0,0);
  u        = (double *)arr->data;
  ndims_u  = arr->nd;
  dsizes_u = (npy_intp *) calloc(ndims_u,sizeof(npy_intp));
  for(i = 0; i < ndims_u; i++ ) {
    dsizes_u[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_u < 3) {
    printf("wrf_avo: u must have at least 3 dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  nz   = dsizes_u[ndims_u-3];
  ny   = dsizes_u[ndims_u-2];
  nxp1 = dsizes_u[ndims_u-1];

/*
 * Extract v
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (var,PyArray_DOUBLE,0,0);
  v        = (double *)arr->data;
  ndims_v  = arr->nd;
  dsizes_v = (npy_intp *) calloc(ndims_v,sizeof(npy_intp));
  for(i = 0; i < ndims_v; i++ ) {
    dsizes_v[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_v != ndims_u) {
    printf("wrf_avo: u and v must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(dsizes_v[ndims_v-3] != nz) {
    printf("wrf_avo: The third-from-the-right dimension of v must be the same as the third-from-the-right dimension of u\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Error checking on leftmost dimension sizes.
 */
  for(i = 0; i < ndims_u-3; i++) {
    if(dsizes_u[i] != dsizes_v[i]) {
      printf("wrf_avo: The leftmost dimensions of u and v must be the same\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

  nyp1 = dsizes_v[ndims_v-2];
  nx   = dsizes_v[ndims_v-1];

/*
 * Extract msfu
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (msfuar,PyArray_DOUBLE,0,0);
  msfu        = (double *)arr->data;
  ndims_msfu  = arr->nd;
  dsizes_msfu = (npy_intp *) calloc(ndims_msfu,sizeof(npy_intp));
  for(i = 0; i < ndims_msfu; i++ ) {
    dsizes_msfu[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_msfu < 2) {
    printf("wrf_avo: msfu must have at least 2 dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(ndims_msfu !=2 && ndims_msfu != (ndims_u-1)) {
    printf("wrf_avo: msfu must be 2D or have one fewer dimensions than u\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(dsizes_msfu[ndims_msfu-2] != ny || dsizes_msfu[ndims_msfu-1] != nxp1) {
    printf("wrf_avo: The rightmost 2 dimensions of msfu must be the same as the rightmost 2 dimensions of u\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Error checking on leftmost dimension sizes. msfu, msfv, msft, and 
 * cor can be 2D or nD.  If they are nD, they must have same leftmost
 * dimensions as other input arrays.
 */
  if(ndims_msfu > 2) {
    for(i = 0; i < ndims_u-3; i++) {
      if(dsizes_msfu[i] != dsizes_u[i]) {
        printf("wrf_avo: If msfu is not 2-dimensional, then the leftmost dimensions of msfu and u must be the same\n");
        Py_INCREF(Py_None);
        return(Py_None);
      }
    }
  }

/*
 * Extract msfv
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (msfvar,PyArray_DOUBLE,0,0);
  msfv        = (double *)arr->data;
  ndims_msfv  = arr->nd;
  dsizes_msfv = (npy_intp *) calloc(ndims_msfv,sizeof(npy_intp));
  for(i = 0; i < ndims_msfv; i++ ) {
    dsizes_msfv[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_msfv != ndims_msfu) {
    printf("wrf_avo: msfu, msfv, msft, and cor must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(dsizes_msfv[ndims_msfv-2] != nyp1 || dsizes_msfv[ndims_msfv-1] != nx) {
    printf("wrf_avo: The rightmost 2 dimensions of msfv must be the same as the rightmost 2 dimensions of v\n");
    Py_INCREF(Py_None);
    return(Py_None);
  }

/*
 * Error checking on leftmost dimension sizes.
 */
  for(i = 0; i < ndims_msfu-2; i++) {
    if(dsizes_msfv[i] != dsizes_msfu[i]) {
      printf("wrf_avo: The leftmost dimensions of msfv and msfu must be the same\n");
      Py_INCREF(Py_None);
      return(Py_None);
    }
  }

/*
 * Extract msft
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (msftar,PyArray_DOUBLE,0,0);
  msft        = (double *)arr->data;
  ndims_msft  = arr->nd;
  dsizes_msft = (npy_intp *) calloc(ndims_msft,sizeof(npy_intp));
  for(i = 0; i < ndims_msft; i++ ) {
    dsizes_msft[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_msft != ndims_msfu) {
    printf("wrf_avo: msfu, msfv, msft, and cor must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return(Py_None);
  }
  if(dsizes_msft[ndims_msft-2] != ny || dsizes_msft[ndims_msft-1] != nx) {
    printf("wrf_avo: The rightmost 2 dimensions of msft must be the same as the rightmost 2 dimensions of th\n");
    Py_INCREF(Py_None);
    return(Py_None);
  }

/*
 * Error checking on leftmost dimension sizes.
 */
  for(i = 0; i < ndims_msfu-2; i++) {
    if(dsizes_msft[i] != dsizes_msfu[i]) {
      printf("wrf_avo: The leftmost dimensions of msft and msfu must be the same\n");
      Py_INCREF(Py_None);
      return(Py_None);
    }
  }

/*
 * Extract cor
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (corar,PyArray_DOUBLE,0,0);
  cor        = (double *)arr->data;
  ndims_cor  = arr->nd;
  dsizes_cor = (npy_intp *) calloc(ndims_cor,sizeof(npy_intp));
  for(i = 0; i < ndims_cor; i++ ) {
    dsizes_cor[i] = (npy_intp)arr->dimensions[i];
  }


/*
 * Error checking on dimensions.
 */
  if(ndims_cor != ndims_msft) {
    printf("wrf_avo: msfu, msfv, msft, and cor must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return(Py_None);
  }

/*
 * Error checking on dimension sizes.
 */
  for(i = 0; i < ndims_msft; i++) {
    if(dsizes_cor[i] != dsizes_msft[i]) {
      printf("wrf_avo: The dimensions of cor and msft must be the same\n");
      Py_INCREF(Py_None);
      return(Py_None);
    }
  }

/*
 * Extract dx and dy (scalars)
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (dxar,PyArray_DOUBLE,0,0);
  dx        = (double *)arr->data;
  ndims_dx  = arr->nd;
  dsizes_dx = (npy_intp *) calloc(ndims_dx,sizeof(npy_intp));
  for(i = 0; i < ndims_dx; i++ ) {
    dsizes_dx[i] = (npy_intp)arr->dimensions[i];
  }

  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (dyar,PyArray_DOUBLE,0,0);
  dy        = (double *)arr->data;
  ndims_dy  = arr->nd;
  dsizes_dy = (npy_intp *) calloc(ndims_dy,sizeof(npy_intp));
  for(i = 0; i < ndims_dy; i++ ) {
    dsizes_dy[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if((ndims_dx != 1 && dsizes_dx[0] != 1) ||
     (ndims_dy != 1 && dsizes_dy[0] != 1)) {
    printf("wrf_avo: dx and dy must be scalars\n");
    Py_INCREF(Py_None);
    return(Py_None);
  }


  nynx     = ny * nx;
  nznynx   = nz * nynx;
  nynxp1   = ny * nxp1;
  nyp1nx   = nyp1 * nx;
  nznynxp1 = nz * nynxp1;
  nznyp1nx = nz * nyp1nx;

/*
 * Test dimension sizes.
 */
  if((nxp1 > INT_MAX) || (nyp1 > INT_MAX) || (nz > INT_MAX) || 
     (nx > INT_MAX) ||(ny > INT_MAX)) {
    printf("wrf_avo: one or more dimension sizes is greater than INT_MAX\n");
    Py_INCREF(Py_None);
    return(Py_None);
  }
  inx = (int) nx;
  iny = (int) ny;
  inz = (int) nz;
  inxp1 = (int) nxp1;
  inyp1 = (int) nyp1;

/*
 * Calculate size of leftmost dimensions, and set
 * dimension sizes for output array.
 */
  dsizes_av = (npy_intp*)calloc(ndims_u,sizeof(npy_intp));  
  if( dsizes_av == NULL) {
    printf("wrf_avo: Unable to allocate memory for holding dimension sizes\n");
    Py_INCREF(Py_None);
    return(Py_None);
  }

  size_leftmost = 1;
  for(i = 0; i < ndims_u-3; i++) {
    size_leftmost *= dsizes_u[i];
    dsizes_av[i] = dsizes_u[i];
  }
  size_av = size_leftmost * nznynx;
  dsizes_av[ndims_u-1] = nx;
  dsizes_av[ndims_u-2] = ny;
  dsizes_av[ndims_u-3] = nz;

/* 
 * Allocate space for output array.
 */
  av = (double *)calloc(size_av, sizeof(double));
  if(av == NULL) {
    printf("wrf_avo: Unable to allocate memory for output array\n");
    Py_INCREF(Py_None);
    return(Py_None);
  }

/*
 * Call the Fortran routine.
 */
  index_u = index_v = index_msfu = index_msfv = index_msft = index_av = 0;
  for(i = 0; i < size_leftmost; i++) {
    NGCALLF(dcomputeabsvort,DCOMPUTEABSVORT)(&av[index_av], &u[index_u], 
                                             &v[index_v], &msfu[index_msfu],
                                             &msfv[index_msfv], 
                                             &msft[index_msft], 
                                             &cor[index_msft],
                                             &dx[0], &dy[0], &inx, &iny, &inz,
                                             &inxp1, &inyp1);
    index_u    += nznynxp1;
    index_v    += nznyp1nx;
    index_av   += nznynx;
    if(ndims_msfu > 2) {
      index_msfu += nynxp1;
      index_msfv += nyp1nx;
      index_msft += nynx;
    }
  }

  /* Free memory */
  free(dsizes_v);
  free(dsizes_cor);
  free(dsizes_msfu);
  free(dsizes_msfv);
  free(dsizes_msft);

  return ((PyObject *) PyArray_SimpleNewFromData(ndims_u,dsizes_av,
                                                 PyArray_DOUBLE,
                                                 (void *) av));
}

PyObject *fplib_wrf_pvo(PyObject *self, PyObject *args)
{
  PyArrayObject *arr = NULL;
/*
 * Input variables
 *
 * Argument # 0
 */
  double *u;
  PyObject *uar = NULL;
  int ndims_u;
  npy_intp *dsizes_u;

/*
 * Argument # 1
 */
  double *v;
  PyObject *var = NULL;
  int ndims_v;
  npy_intp *dsizes_v;

/*
 * Argument # 2
 */
  double *th;
  PyObject *thar = NULL;
  int ndims_th;
  npy_intp *dsizes_th;

/*
 * Argument # 3
 */
  double *p;
  PyObject *par = NULL;
  int ndims_p;
  npy_intp *dsizes_p;

/*
 * Argument # 4
 */
  double *msfu;
  PyObject *msfuar = NULL;
  int ndims_msfu;
  npy_intp *dsizes_msfu;

/*
 * Argument # 5
 */
  double *msfv;
  PyObject *msfvar = NULL;
  int ndims_msfv;
  npy_intp *dsizes_msfv;

/*
 * Argument # 6
 */
  double *msft;
  PyObject *msftar = NULL;
  int ndims_msft;
  npy_intp *dsizes_msft;

/*
 * Argument # 7
 */
  double *cor;
  PyObject *corar = NULL;
  int ndims_cor;
  npy_intp *dsizes_cor;

/*
 * Argument # 8
 */
  double *dx;
  PyObject *dxar = NULL;
  int ndims_dx;
  npy_intp *dsizes_dx;

/*
 * Argument # 9
 */
  double *dy;
  PyObject *dyar = NULL;
  int ndims_dy;
  npy_intp *dsizes_dy;

/*
 * Argument # 10
 */
  int *opt;

/*
 * Return variable
 */
  double *pv;

/*
 * Various
 */
  npy_intp nx, ny, nz, nxp1, nyp1;
  npy_intp nznynxp1, nznyp1nx, nznynx, nynxp1, nyp1nx, nynx;
  npy_intp i, size_pv, size_leftmost;
  npy_intp index_u, index_v, index_th, index_msfu, index_msfv, index_msft;
  int inx, iny, inz, inxp1, inyp1;

/*
 * Retrieve parameters.
 *
 * Note any of the pointer parameters can be set to NULL, which
 * implies you don't care about its value.
 */

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOi:wrf_pvo", &uar, &var, &thar, &par,
                        &msfuar, &msfvar, &msftar, &corar, &dxar, &dyar, &opt)) {
    printf("wrf_avo: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Extract u
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (uar,PyArray_DOUBLE,0,0);
  u        = (double *)arr->data;
  ndims_u  = arr->nd;
  dsizes_u = (npy_intp *) calloc(ndims_u,sizeof(npy_intp));
  for(i = 0; i < ndims_u; i++ ) {
    dsizes_u[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_u < 3) {
    printf("wrf_pvo: u must have at least 3 dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  nz   = dsizes_u[ndims_u-3];
  ny   = dsizes_u[ndims_u-2];
  nxp1 = dsizes_u[ndims_u-1];

/*
 * Extract v
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (var,PyArray_DOUBLE,0,0);
  v        = (double *)arr->data;
  ndims_v  = arr->nd;
  dsizes_v = (npy_intp *) calloc(ndims_v,sizeof(npy_intp));
  for(i = 0; i < ndims_v; i++ ) {
    dsizes_v[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_v != ndims_u) {
    printf("wrf_pvo: u, v, th, and p must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(dsizes_v[ndims_v-3] != nz) {
    printf("wrf_pvo: The third-from-the-right dimension of v must be the same as the third-from-the-right dimension of u\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Error checking on leftmost dimension sizes.
 */
  for(i = 0; i < ndims_u-3; i++) {
    if(dsizes_u[i] != dsizes_v[i]) {
      printf("wrf_pvo: The leftmost dimensions of u and v must be the same\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

  nyp1 = dsizes_v[ndims_v-2];
  nx   = dsizes_v[ndims_v-1];

/*
 * Extract th
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (thar,PyArray_DOUBLE,0,0);
  th        = (double *)arr->data;
  ndims_th  = arr->nd;
  dsizes_th = (npy_intp *) calloc(ndims_th,sizeof(npy_intp));
  for(i = 0; i < ndims_th; i++ ) {
    dsizes_th[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_th != ndims_u) {
    printf("wrf_pvo: u, v, th, and p must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

  if(dsizes_th[ndims_th-3] != nz || dsizes_th[ndims_th-2] != ny ||
     dsizes_th[ndims_th-1] != nx) {
    printf("wrf_pvo: The rightmost dimensions of th must be a combination of the dimensions of u and v (see documentation)\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Error checking on leftmost dimension sizes.
 */
  for(i = 0; i < ndims_u-3; i++) {
    if(dsizes_th[i] != dsizes_u[i]) {
      printf("wrf_pvo: The leftmost dimensions of th and u must be the same\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Extract p
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
 * Error checking on dimensions.
 */
  if(ndims_p != ndims_u) {
    printf("wrf_pvo: u, v, th, and p must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Error checking on dimension sizes.
 */
  for(i = 0; i < ndims_th; i++) {
    if(dsizes_p[i] != dsizes_th[i]) {
      printf("wrf_pvo: The dimensions of p and th must be the same\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Extract msfu
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (msfuar,PyArray_DOUBLE,0,0);
  msfu        = (double *)arr->data;
  ndims_msfu  = arr->nd;
  dsizes_msfu = (npy_intp *) calloc(ndims_msfu,sizeof(npy_intp));
  for(i = 0; i < ndims_msfu; i++ ) {
    dsizes_msfu[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_msfu < 2) {
    printf("wrf_pvo: msfu must have at least 2 dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(ndims_msfu !=2 && ndims_msfu != (ndims_u-1)) {
    printf("wrf_pvo: msfu must be 2D or have one fewer dimensions than u\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(dsizes_msfu[ndims_msfu-2] != ny || dsizes_msfu[ndims_msfu-1] != nxp1) {
    printf("wrf_pvo: The rightmost 2 dimensions of msfu must be the same as the rightmost 2 dimensions of u\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Error checking on leftmost dimension sizes. msfu, msfv, msft, and 
 * cor can be 2D or nD.  If they are nD, they must have same leftmost
 * dimensions as other input arrays.
 */
  if(ndims_msfu > 2) {
    for(i = 0; i < ndims_u-3; i++) {
      if(dsizes_msfu[i] != dsizes_u[i]) {
        printf("wrf_pvo: If msfu is not 2-dimensional, then the leftmost dimensions of msfu and u must be the same\n");
        Py_INCREF(Py_None);
        return Py_None;
      }
    }
  }

/*
 * Extract msfv
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (msfvar,PyArray_DOUBLE,0,0);
  msfv        = (double *)arr->data;
  ndims_msfv  = arr->nd;
  dsizes_msfv = (npy_intp *) calloc(ndims_msfv,sizeof(npy_intp));
  for(i = 0; i < ndims_msfv; i++ ) {
    dsizes_msfv[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_msfv != ndims_msfu) {
    printf("wrf_pvo: msfu, msfv, msft, and cor must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(dsizes_msfv[ndims_msfv-2] != nyp1 || dsizes_msfv[ndims_msfv-1] != nx) {
    printf("wrf_pvo: The rightmost 2 dimensions of msfv must be the same as the rightmost 2 dimensions of v\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Error checking on leftmost dimension sizes.
 */
  for(i = 0; i < ndims_msfu-2; i++) {
    if(dsizes_msfv[i] != dsizes_msfu[i]) {
      printf("wrf_pvo: The leftmost dimensions of msfv and msfu must be the same\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Extract msft
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (msftar,PyArray_DOUBLE,0,0);
  msft        = (double *)arr->data;
  ndims_msft  = arr->nd;
  dsizes_msft = (npy_intp *) calloc(ndims_msft,sizeof(npy_intp));
  for(i = 0; i < ndims_msft; i++ ) {
    dsizes_msft[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_msft != ndims_msfu) {
    printf("wrf_pvo: msfu, msfv, msft, and cor must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(dsizes_msft[ndims_msft-2] != ny || dsizes_msft[ndims_msft-1] != nx) {
    printf("wrf_pvo: The rightmost 2 dimensions of msft must be the same as the rightmost 2 dimensions of th\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Error checking on leftmost dimension sizes.
 */
  for(i = 0; i < ndims_msfu-2; i++) {
    if(dsizes_msft[i] != dsizes_msfu[i]) {
      printf("wrf_pvo: The leftmost dimensions of msft and msfu must be the same\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Extract cor
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (corar,PyArray_DOUBLE,0,0);
  cor        = (double *)arr->data;
  ndims_cor  = arr->nd;
  dsizes_cor = (npy_intp *) calloc(ndims_cor,sizeof(npy_intp));
  for(i = 0; i < ndims_cor; i++ ) {
    dsizes_cor[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if(ndims_cor != ndims_msft) {
    printf("wrf_pvo: msfu, msfv, msft, and cor must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Error checking on dimension sizes.
 */
  for(i = 0; i < ndims_msft; i++) {
    if(dsizes_cor[i] != dsizes_msft[i]) {
      printf("wrf_pvo: The dimensions of cor and msft must be the same\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Extract dx and dy (scalars)
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (dxar,PyArray_DOUBLE,0,0);
  dx        = (double *)arr->data;
  ndims_dx  = arr->nd;
  dsizes_dx = (npy_intp *) calloc(ndims_dx,sizeof(npy_intp));
  for(i = 0; i < ndims_dx; i++ ) {
    dsizes_dx[i] = (npy_intp)arr->dimensions[i];
  }

  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (dyar,PyArray_DOUBLE,0,0);
  dy        = (double *)arr->data;
  ndims_dy  = arr->nd;
  dsizes_dy = (npy_intp *) calloc(ndims_dy,sizeof(npy_intp));
  for(i = 0; i < ndims_dy; i++ ) {
    dsizes_dy[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Error checking on dimensions.
 */
  if((ndims_dx != 1 && dsizes_dx[0] != 1) ||
     (ndims_dy != 1 && dsizes_dy[0] != 1)) {
    printf("wrf_avo: dx and dy must be scalars\n");
    Py_INCREF(Py_None);
    return(Py_None);
  }


  nynx     = ny * nx;
  nznynx   = nz * nynx;
  nynxp1   = ny * nxp1;
  nyp1nx   = nyp1 * nx;
  nznynxp1 = nz * nynxp1;
  nznyp1nx = nz * nyp1nx;

/*
 * Test dimension sizes.
 */
    if((nxp1 > INT_MAX) || (nyp1 > INT_MAX) || (nz > INT_MAX) || 
       (nx > INT_MAX) ||(ny > INT_MAX)) {
      printf("wrf_pvo: one or more dimension sizes is greater than INT_MAX\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
    inx = (int) nx;
    iny = (int) ny;
    inz = (int) nz;
    inxp1 = (int) nxp1;
    inyp1 = (int) nyp1;

/*
 * Calculate size of leftmost dimensions. The dimension
 * sizes of the output array are exactly the same
 * as th's dimension sizes.
 */
  size_leftmost = 1;
  for(i = 0; i < ndims_u-3; i++) size_leftmost *= dsizes_u[i];
  size_pv = size_leftmost * nznynx;

/* 
 * Allocate space for output array.
 */
  pv = (double *)calloc(size_pv, sizeof(double));
  if(pv == NULL) {
    printf("wrf_pvo: Unable to allocate memory for output array\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Call the Fortran routine.
 */
  index_u = index_v = index_th = index_msfu = index_msfv = index_msft = 0;
  for(i = 0; i < size_leftmost; i++) {
    NGCALLF(dcomputepv,DCOMPUTEPV)(&pv[index_th], &u[index_u], &v[index_v], 
                                   &th[index_th], &p[index_th], 
                                   &msfu[index_msfu], &msfv[index_msfv], 
                                   &msft[index_msft], &cor[index_msft], 
                                   &dx[0], &dy[0], &inx, &iny, &inz, &inxp1, &inyp1);
    index_u    += nznynxp1;
    index_v    += nznyp1nx;
    index_th   += nznynx;
    if(ndims_msfu > 2) {
      index_msfu += nynxp1;
      index_msfv += nyp1nx;
      index_msft += nynx;
    }
  }

/*
 * Free unneeded memory.
 */
  /* Free memory */
  free(dsizes_v);
  free(dsizes_p);
  free(dsizes_cor);
  free(dsizes_msfu);
  free(dsizes_msfv);
  free(dsizes_msft);

  return ((PyObject *) PyArray_SimpleNewFromData(ndims_th,dsizes_th,
                                                 PyArray_DOUBLE,
                                                 (void *) pv));
}

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
    printf("wrf_tk: p and theta must be the same dimensionality\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_p; i++) {
    if(dsizes_p[i] != (npy_intp)arr->dimensions[i]) {
      printf("wrf_tk: p and theta must be the same dimensionality\n");
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
    printf("wrf_tk: Unable to allocate memory for output array\n");
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


PyObject *fplib_wrf_td(PyObject *self, PyObject *args)
{
  PyObject *par = NULL;
  PyObject *qvar = NULL;
  double *p = NULL;
  double *qv = NULL;
  PyArrayObject *arr = NULL;
  double *td;
/*
 * Various
 */
  int ndims_p, inx;
  npy_intp i, nx, *dsizes_p, size_leftmost, size_td, index_p;

  if (!PyArg_ParseTuple(args, "OO:wrf_td", &par, &qvar)) {
    printf("wrf_td: argument parsing failed\n");
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

                            (qvar,PyArray_DOUBLE,0,0);
  qv = (double *)arr->data;
/*
 * Error checking. Input variables must be same size.
 */
  if(ndims_p != arr->nd) {
    printf("wrf_td: p and qv must be the same dimensionality\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_p; i++) {
    if(dsizes_p[i] != (npy_intp)arr->dimensions[i]) {
      printf("wrf_td: p and qv must be the same dimensionality\n");
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
    printf("wrf_td: nx = %ld is greater than INT_MAX", nx);
    Py_INCREF(Py_None);
    return Py_None;
  }
  inx = (int) nx;

  size_td = size_leftmost * nx;

  td = (double *)calloc(size_td,sizeof(double));
  if(td == NULL) {
    printf("wrf_td: Unable to allocate memory for output array\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Loop across leftmost dimensions and call the Fortran routine for each
 * one-dimensional subsection.
 */
  index_p = 0;
  for(i = 0; i < size_leftmost; i++) {
    convert_to_hPa(&p[index_p],nx);
    var_zero(qv, nx);                   /* Set all values < 0 to 0. */

    NGCALLF(dcomputetd,DCOMPUTETD)(&td[index_p],&p[index_p],
                                   &qv[index_p],&inx);
    index_p += nx;    /* Increment index */}
  
  return ((PyObject *) PyArray_SimpleNewFromData(ndims_p,dsizes_p,
                                                 PyArray_DOUBLE,
                                                 (void *) td));
}


PyObject *fplib_wrf_slp(PyObject *self, PyObject *args)
{
/*
 * Input array variables
 */
  PyArrayObject *arr = NULL;
  PyObject *zar = NULL;
  PyObject *tar = NULL;
  PyObject *par = NULL;
  PyObject *qar = NULL;
  double *z = NULL; 
  double *t = NULL; 
  double *p = NULL; 
  double *q = NULL; 
  int ndims_z, ndims_t, ndims_p, ndims_q;
  npy_intp *dsizes_z;
  npy_intp *dsizes_t;
  npy_intp *dsizes_p;
  npy_intp *dsizes_q;
/*
 * Output variable.
 */
  double *slp;
  int ndims_slp;
  npy_intp *dsizes_slp;
  npy_intp size_slp;
/*
 * Various
 */
  npy_intp i, nx, ny, nz, nxy, nxyz, size_leftmost, index_nxy, index_nxyz;
  double *tmp_t_sea_level, *tmp_t_surf, *tmp_level;
  int inx, iny, inz;

  if (!PyArg_ParseTuple(args, "OOOO:wrf_slp", &zar, &tar, &par, &qar)) {
    printf("wrf_slp: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 *  Extract z.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (zar,PyArray_DOUBLE,0,0);
  z        = (double *)arr->data;
  ndims_z  = arr->nd;
  dsizes_z = (npy_intp *) calloc(ndims_z,sizeof(npy_intp));
  for(i = 0; i < ndims_z; i++ ) {
    dsizes_z[i] = (npy_intp)arr->dimensions[i];
  }

/*
 *  Extract t.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (tar,PyArray_DOUBLE,0,0);
  t        = (double *)arr->data;
  ndims_t  = arr->nd;
  dsizes_t = (npy_intp *) calloc(ndims_t,sizeof(npy_intp));
  for(i = 0; i < ndims_t; i++ ) {
    dsizes_t[i] = (npy_intp)arr->dimensions[i];
  }

/*
 *  Extract p.
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
 *  Extract q.
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                            (qar,PyArray_DOUBLE,0,0);
  q        = (double *)arr->data;
  ndims_q  = arr->nd;
  dsizes_q = (npy_intp *) calloc(ndims_q,sizeof(npy_intp));
  for(i = 0; i < ndims_q; i++ ) {
    dsizes_q[i] = (npy_intp)arr->dimensions[i];
  }


/*
 * Error checking. Input variables must be same size, and must have at least
 * 3 dimensions.
 */
  if(ndims_z != ndims_t || ndims_z != ndims_p || ndims_z != ndims_q) {
    printf("wrf_slp: The z, t, p, and q arrays must have the same number of dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  if(ndims_z < 3) {
    printf("wrf_slp: The z, t, p, and q arrays must have at least 3 dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_z; i++) {
    if(dsizes_z[i] != dsizes_t[i] || dsizes_z[i] != dsizes_p[i] ||
       dsizes_z[i] != dsizes_q[i]) {
      printf("wrf_slp: z, t, p, and q must be the same dimensionality\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }
/*
 * Allocate space to set dimension sizes.
 */
  ndims_slp  = ndims_z-1;
  dsizes_slp = (npy_intp*)calloc(ndims_slp,sizeof(npy_intp));  
  if( dsizes_slp == NULL) {
    printf("wrf_slp: Unable to allocate memory for holding dimension sizes\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Set sizes for output array and calculate size of leftmost dimensions.
 * The output array will have one less dimension than the four input arrays.
 */
  size_leftmost = 1;
  for(i = 0; i < ndims_z-3; i++) {
    dsizes_slp[i] = dsizes_z[i];
    size_leftmost *= dsizes_z[i];
  }
  nx = dsizes_z[ndims_z-1];
  ny = dsizes_z[ndims_z-2];
  nz = dsizes_z[ndims_z-3];
  dsizes_slp[ndims_slp-1] = nx;
  dsizes_slp[ndims_slp-2] = ny;
  nxy  = nx * ny;
  nxyz = nxy * nz;
  size_slp = size_leftmost * nxy;
/*
 * Test dimension sizes.
 */
  if((nx > INT_MAX) || (ny > INT_MAX) || (nz > INT_MAX)) {
    printf("wrf_slp: nx, ny, and/or nz is greater than INT_MAX\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  inx = (int) nx;
  iny = (int) ny;
  inz = (int) nz;

/*
 * Allocate space for work arrays.
 */ 
  tmp_t_sea_level = (double *)calloc(nxy,sizeof(double));
  tmp_t_surf      = (double *)calloc(nxy,sizeof(double));
  tmp_level       = (double *)calloc(nxy,sizeof(double));
  if(tmp_t_sea_level == NULL || tmp_t_surf == NULL || tmp_level == NULL) {
    printf("wrf_slp: Unable to allocate memory for temporary arrays\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Allocate space for output array.
 */ 
  slp = (double *)calloc(size_slp,sizeof(double));
  if(slp == NULL) {
    printf("wrf_slp: Unable to allocate memory for output array\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Loop across leftmost dimensions and call the Fortran routine
 * for each three-dimensional subsection.
 */
  index_nxy = index_nxyz = 0;
  for(i = 0; i < size_leftmost; i++) {
    var_zero(&q[index_nxyz], nxyz);   /* Set all values < 0 to 0. */
/*
 * Call Fortran routine.
 */
    NGCALLF(dcomputeseaprs,DCOMPUTESEAPRS)(&inx,&iny,&inz,&z[index_nxyz],
                                           &t[index_nxyz],&p[index_nxyz],
                                           &q[index_nxyz],&slp[index_nxy],
                                           tmp_t_sea_level,tmp_t_surf,
                                           tmp_level);

    index_nxyz += nxyz;    /* Increment indices */
    index_nxy  += nxy;
  }
/*
 * Free up memory.
 */
  free(dsizes_z);
  free(dsizes_t);
  free(dsizes_p);
  free(dsizes_q);
  free(tmp_t_sea_level);
  free(tmp_t_surf);
  free(tmp_level);

  return ((PyObject *) PyArray_SimpleNewFromData(ndims_slp,dsizes_slp,
                                                 PyArray_DOUBLE,
                                                 (void *) slp));
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
    printf("wrf_rh: qv, p, t must be the same dimensionality\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_qv; i++) {
    if(dsizes_qv[i] != (npy_intp)arr->dimensions[i]) {
      printf("wrf_rh: qv, p, t must be the same dimensionality\n");
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
    printf("wrf_rh: qv, p, t must be the same dimensionality\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_qv; i++) {
    if(dsizes_qv[i] != (npy_intp)arr->dimensions[i]) {
      printf("wrf_rh: qv, p, t must be the same dimensionality\n");
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
    printf("wrf_rh: Unable to allocate memory for output array\n");
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
    printf("wrf_dbz: The p array must have at least 3 dimensions\n");
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
    printf("wrf_dbz: one or more dimension sizes is greater than INT_MAX\n");    
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
    printf("wrf_dbz: The t array must have the same number of dimensions as the p array\n");
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
    printf("wrf_dbz: The qv array must have the same number of dimensions as the p array\n");
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
    printf("wrf_dbz: The qr array must have the same number of dimensions as the p array\n");
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
    printf("wrf_dbz: qs must either be a scalar or have the same number of dimensions as the p array\n");
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
    printf("wrf_dbz: qg must either be a scalar or have the same number of dimensions as the p array\n");
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
      printf("wrf_dbz: The p, t, qv, qr, qs, and qg arrays must have the same dimensions (qs and qg are optional)\n");
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
      printf("wrf_dbz: Unable to allocate memory for promoting qs array\n");
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
      printf("wrf_dbz: Unable to allocate memory for promoting qg array\n");
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
    printf("wrf_dbz: Unable to allocate memory for output array\n");
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
  free(dsizes_p);
  free(dsizes_t);
  free(dsizes_qv);
  free(dsizes_qr);
  free(dsizes_qs);
  free(dsizes_qg);
 
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

/*
 * This routine sets all values of var < 0 to 0.0. This is
 * so you don't have to do this in the NCL script. It's the
 * equivalent of:
 *
 * tmp_var = tmp_var > 0.0
 *
 */
void var_zero(double *tmp_var, npy_intp n)
{
  npy_intp i;

  for(i = 0; i < n; i++) {
    if(tmp_var[i] < 0.0) tmp_var[i] = 0.0;
  }
}

/* Converts from hPa to Pa. */

void convert_to_hPa(double *pp, npy_intp np)
{
  npy_intp i;

  for(i = 0; i < np; i++) pp[i] *= 0.01;
}

