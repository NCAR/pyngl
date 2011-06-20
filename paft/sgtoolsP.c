extern int NGCALLF(gcinout,GCINOUT)(double*,double*,double*,double*,
                                     int*,double*);

PyObject *fplib_gc_inout(PyObject *self, PyObject *args)
{
/*
 * Input variables
 */
  PyObject *arr_plat = NULL, *arr_plon = NULL;
  PyObject *arr_lat = NULL, *arr_lon = NULL;
  double *plat, *plon, *lat, *lon, *tlat, *tlon;

  npy_intp *dsizes_plat, *dsizes_lat;
  int ndims_plat, ndims_lat;

/*
 * output variable 
 */
  int *tfval;
  int size_tfval;
  PyArrayObject *ret_tfval;

/*
 * Define some PyArray objects for input and output.
 */
  PyArrayObject *arr;

/*
 * Various
 */
  int i,itmp,npts,nptsp1,jpol,tsize;
  double *work;

/*
 * Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args,(char *)"OOOO:gc_inout",&arr_plat,&arr_plon,
                        &arr_lat,&arr_lon)) {
    printf("gc_inout: fatal: argument parsing failed\n");
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
                        (arr_plat,PyArray_DOUBLE,0,0);
  plat        = (double *)arr->data;
  ndims_plat  = arr->nd;
  dsizes_plat = (npy_intp *)malloc(ndims_plat * sizeof(npy_intp));
  for(i = 0; i < ndims_plat; i++ ) dsizes_plat[i] = (npy_intp)arr->dimensions[i];

/*
 * Get argument # 1
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_plon,PyArray_DOUBLE,0,0);
  plon = (double *)arr->data;
/*
 * Check dimension sizes.
 */
  if(arr->nd != ndims_plat) {
    printf("gc_inout: fatal: The plat/plon arrays must have the same dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_plat; i++ ) {
    if((npy_intp)arr->dimensions[i] != dsizes_plat[i]) {
      printf("gc_inout: fatal: The plat/plon arrays must have the same dimensions\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Get argument # 2
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_lat,PyArray_DOUBLE,0,0);
  lat        = (double *)arr->data;
  ndims_lat  = arr->nd;
  dsizes_lat = (npy_intp *)malloc(ndims_lat * sizeof(npy_intp));
  for(i = 0; i < ndims_lat; i++ ) dsizes_lat[i] = (npy_intp)arr->dimensions[i];

  if (ndims_lat == 1) {
    if ( ndims_plat != 1 ) {
      printf("gc_inout: if the final two arrays are singly dimensioned, then the first two must be as well.");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }
  else {
    if (ndims_plat != ndims_lat-1) {
      printf("gc_inout: the first two input arrays must have exactly one less dimension than the last two.");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 * Get argument # 3
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_lon,PyArray_DOUBLE,0,0);
  lon = (double *)arr->data;

/*
 * Check dimension sizes.
 */
  if(arr->nd != ndims_lat) {
    printf("gc_inout: fatal: The lat/lon arrays must have the same dimensions\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  for(i = 0; i < ndims_lat; i++ ) {
    if((npy_intp)arr->dimensions[i] != dsizes_lat[i]) {
      printf("gc_inout: fatal: The lat/lon arrays must have the same dimensions\n");
      Py_INCREF(Py_None);
      return Py_None;
    }
  }

/*
 *  Check on dimension sizes of plat/plon versus lat/lon.
 */
  if (ndims_lat > 0) {
    for(i = 0; i < ndims_lat-1; i++) {
      if (dsizes_plat[i] != dsizes_lat[i]) {
        printf("gc_inout: the dimensions sizes for the first two arrays must agree with the dimension sizes of the last two up through the penultimate dimension of the last two.");
        Py_INCREF(Py_None);
        return Py_None;
      }
    }
  }

/*
 * Find the number of points in each polygon and check that it
 * is at least three.
 */
  npts   = dsizes_lat[ndims_lat-1];
  nptsp1 = npts+1;
  if (npts < 3) {
    printf("gc_inout: the polygon must have at least three points.");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Determine size for the return array.
 */
  size_tfval = 1;
  for (i = 0; i < ndims_lat-1; i++) {
    size_tfval *= dsizes_lat[i];
  }

/*
 * Determine total size of input arrays.
 */
  tsize = 1;
  for (i = 0; i < ndims_lat; i++) {
    tsize *= dsizes_lat[i];
  }

/* 
 * Allocate space for output array.
 */
  tfval = (int *)calloc(size_tfval, sizeof(int));
  if(tfval == NULL) {
    printf("gc_inout: fatal: Unable to allocate memory for output array\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Call the Fortran version of this routine.  Close the polygons
 * if they are not closed.
 */
  work = (double *)calloc(4*(npts+1), sizeof(double));
  jpol = 0;
  for( i = 0; i < size_tfval; i++ ) {

/*
 * If the polygon is not closed, close it before calling the
 * Fortran.
 */
    if (lat[jpol] != lat[jpol+npts-1] || lon[jpol] != lon[jpol+npts-1]) {
      tlat = (double *) calloc(npts+1,sizeof(double));
      tlon = (double *) calloc(npts+1,sizeof(double));
      memcpy(tlat,lat+jpol,npts*sizeof(double));
      memcpy(tlon,lon+jpol,npts*sizeof(double));
      tlat[npts] = tlat[0];
      tlon[npts] = tlon[0];
      itmp = NGCALLF(gcinout,GCINOUT)(plat+i,plon+i,tlat,tlon,&nptsp1,work);
      free(tlat);
      free(tlon);
    }
    else {
      itmp = NGCALLF(gcinout,GCINOUT)(plat+i,plon+i,lat+jpol,lon+jpol,
                                      &npts,work);
    }
    if (itmp == 0) tfval[i] = 1;
    else           tfval[i] = 0;
    jpol = jpol+npts;
  }

/*
 * Free memory.
 */
  free(work);

/*
 * Return value back to PyNGL script.
 */
  if (ndims_lat == 1) {
    dsizes_lat[0] = 1;
    ret_tfval = (PyArrayObject *) PyArray_SimpleNewFromData(1,dsizes_lat,
                                                            PyArray_INT,
                                                            (void *) tfval);
  }
  else {
    ret_tfval = (PyArrayObject *) PyArray_SimpleNewFromData(ndims_lat-1,
                                                            dsizes_lat,
                                                            PyArray_INT,
                                                            (void *) tfval);
  }

  PyArray_Return(ret_tfval);
}
