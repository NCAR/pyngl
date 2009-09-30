extern void NGCALLF(vinth2p,VINTH2P)(double *, double *, double *, double *,
                                     double *, double *, double *,int *,
                                     int *, double *, double *, int *,
                                     int *, int *, int *, int *, int *);

PyObject *fplib_vinth2p(PyObject *self, PyObject *args)
{
/*
 * Argument # 0
 */
  PyObject *obj_datai = NULL;
  PyArrayObject *arr_datai = NULL;
  double *datai;
  npy_intp ndims_datai, index_datai;

/*
 * Argument # 1
 */
  PyObject *obj_hbcofa = NULL;
  PyArrayObject *arr_hbcofa = NULL;
  double *hbcofa;
  int ndims_hbcofa;
/*
 * Argument # 2
 */
  PyObject *obj_hbcofb = NULL;
  PyArrayObject *arr_hbcofb = NULL;
  double *hbcofb;
  int ndims_hbcofb;
/*
 * Argument # 3
 */
  PyObject *obj_plevo = NULL;
  PyArrayObject *arr_plevo = NULL;
  double *plevo;
  int ndims_plevo;
/*
 * Argument # 4
 */
  PyObject *obj_psfc = NULL;
  PyArrayObject *arr_psfc = NULL;
  double *psfc;
  int ndims_psfc, index_psfc;
  npy_intp *dsizes_psfc;
/*
 * Argument # 5
 */
  int intyp;
/*
 * Argument # 6
 */
  PyObject *obj_p0 = NULL;
  PyArrayObject *arr_p0 = NULL;
  double *p0;
/*
 * Argument # 7
 */
  int ilev;
/*
 * Argument # 8
 */
  int *kxtrp;
/*
 * Return variable
 */
  PyObject *ret_obj;
  double *datao;
  npy_intp *dsizes_datao;

/*
 * Various
 */
  int i, ntime, nlevi, nlevip1, nlevo, nlat, nlon;
  int nlatlon, nlevilatlon, nlevolatlon,  nlevolatlonsize;
  double *plevi, msg;

/*
 * Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args,(char *)"OOOOOiOii:vinth2p",&obj_datai,
			&obj_hbcofa,&obj_hbcofb,&obj_plevo,&obj_psfc,
			&intyp,&obj_p0,&ilev,&kxtrp)) {
    printf("vinth2p: fatal: argument parsing failed\n");
    goto fail;
  }
/*
 * Start extracting array information.
 */

/*
 * Get argument # 0
 */
  arr_datai   = (PyArrayObject *) PyArray_ContiguousFromAny
                                 (obj_datai,PyArray_DOUBLE,0,0);
  ndims_datai = arr_datai->nd;
/*
 * Check dimension sizes.
 */
  if(ndims_datai != 3 && ndims_datai != 4) {
    printf("vinth2p: fatal: The datai array must be 3 or 4 dimensions\n");
    goto fail;
  }
  nlevi = (npy_intp)arr_datai->dimensions[ndims_datai-3];
  nlat  = (npy_intp)arr_datai->dimensions[ndims_datai-2];
  nlon  = (npy_intp)arr_datai->dimensions[ndims_datai-1];
  if(ndims_datai == 4) ntime = (npy_intp)arr_datai->dimensions[0];
  else                 ntime = 1;
/*
 * Get argument # 1
 */
  arr_hbcofa = (PyArrayObject *) PyArray_ContiguousFromObject
                                (obj_hbcofa,PyArray_DOUBLE,0,0);
  ndims_hbcofa  = arr_hbcofa->nd;

/*
 * Check dimension sizes.
 */
  if(ndims_hbcofa != 1 && (npy_intp)arr_hbcofa->dimensions[0] != nlevi) { 
    printf("vinth2p: fatal: The hbcofa array must be one-dimensional and equal to the level dimension of datai\n");
    goto fail;
  }

/*
 * Get argument # 2
 */
  arr_hbcofb = (PyArrayObject *) PyArray_ContiguousFromObject
                                (obj_hbcofb,PyArray_DOUBLE,0,0);
  ndims_hbcofb  = arr_hbcofb->nd;

/*
 * Check dimension sizes.
 */
  if(ndims_hbcofb != 1 && (npy_intp)arr_hbcofb->dimensions[0] != nlevi) { 
    printf("vinth2p: fatal: The hbcofb array must be one-dimensional and equal to the level dimension of datai\n");
    goto fail;
  }


/*
 * Get argument # 3
 */
  arr_plevo = (PyArrayObject *) PyArray_ContiguousFromObject
                               (obj_plevo,PyArray_DOUBLE,0,0);
  ndims_plevo = arr_plevo->nd;

/*
 * Check dimension sizes.
 */
  if(ndims_plevo != 1) {
    printf("vinth2p: fatal: The plevo array must be one-dimensional\n");
    goto fail;
  }
  nlevo = (npy_intp)arr_plevo->dimensions[0];
  
/*
 * Get argument # 4
 */
  arr_psfc = (PyArrayObject *) PyArray_ContiguousFromAny
                            (obj_psfc,PyArray_DOUBLE,0,0);
  ndims_psfc  = arr_psfc->nd;

/*
 * Check dimension sizes. Must the same size as 'datai', minus the level
 * dimemsion.
 */
  if(ndims_psfc != (ndims_datai-1)) {
    printf("vinth2p: fatal: The psfc array must be one less dimension than datai\n");
    goto fail;
  }
  if( (ndims_datai == 3 && ((npy_intp)arr_psfc->dimensions[0] != nlat && 
                            (npy_intp)arr_psfc->dimensions[1] != nlon)) ||
      (ndims_datai == 4 && ((npy_intp)arr_psfc->dimensions[0] != ntime && 
                            (npy_intp)arr_psfc->dimensions[1] != nlat &&
                            (npy_intp)arr_psfc->dimensions[2] != nlon))){
    printf("vinth2p: fatal: The psfc dimensions must be equal to all but the level dimension of 'datai'\n");
    goto fail;
  }

/*
 * Get argument # 6
 */
  arr_p0 = (PyArrayObject *) PyArray_ContiguousFromAny
                           (obj_p0,PyArray_DOUBLE,0,0);
  if(arr_p0->nd != 0) {
    printf("vinth2p: fatal: p0 must be a scalar\n");
    goto fail;
  }

/* 
 * Allocate space for output array and its dimensions.
 */
  nlatlon     = nlat * nlon;
  nlevilatlon = nlevi * nlatlon;
  nlevolatlon = nlevo * nlatlon;

  dsizes_datao = (npy_intp *)calloc(ndims_datai, sizeof(npy_intp));
  if(dsizes_datao == NULL) {
    printf("vinth2p: fatal: Unable to allocate memory for dimension sizes\n");
    goto fail;
  }
  dsizes_datao[ndims_datai-3] = nlevo;
  dsizes_datao[ndims_datai-2] = nlat;
  dsizes_datao[ndims_datai-1] = nlon; 
  if(ndims_datai == 4) dsizes_datao[0] = ntime;

/* 
 * Allocate space for plevi array, which is calculated inside
 * Fortran routine, but not used here.
 */
  nlevip1 = nlevi + 1;
  plevi   = (double*)calloc(nlevip1,sizeof(double));
  if(plevi == NULL) {
    printf("vinth2p: fatal: Unable to allocate memory for plevi array\n");
    goto fail;
  }

/*
 * Create Python return object.
 */
  ret_obj = (PyObject *) PyArray_SimpleNew(ndims_datai,
					   dsizes_datao,
					   PyArray_DOUBLE);
  if (ret_obj == NULL) {
    free(plevi);
    free(dsizes_datao);
    goto fail;
  }
  nlevolatlonsize = nlevolatlon * PyArray_ITEMSIZE((PyArrayObject*) ret_obj);

/* 
 * Get data pointer to Python object.
 */
  datai  = PyArray_DATA(arr_datai);
  hbcofa = PyArray_DATA(arr_hbcofa);
  hbcofb = PyArray_DATA(arr_hbcofb); 
  p0     = PyArray_DATA(arr_p0);
  plevo  = PyArray_DATA(arr_plevo);
  psfc   = PyArray_DATA(arr_psfc);
  datao  = PyArray_DATA((PyArrayObject*) ret_obj);

/* 
 * Other values we need to pass to Fortran routine.
 */
  msg = 1.e30;

  index_datai = index_psfc = 0;
  for(i = 0; i < ntime; i++) {
    NGCALLF(vinth2p,VINTH2P)(&datai[index_datai], datao, hbcofa, hbcofb,
			     p0, plevi, plevo, &intyp, &ilev, 
			     &psfc[index_psfc], &msg, &kxtrp, &nlon, &nlat,
			     &nlevi, &nlevip1, &nlevo);
    index_datai += nlevilatlon;
    index_psfc  += nlatlon;
    datao       += nlevolatlon;
  }

/*
 * Return value back to Python script.
 */
  free(plevi);
  free(dsizes_datao);

  Py_DECREF(arr_datai);
  Py_DECREF(arr_hbcofa);
  Py_DECREF(arr_hbcofb);
  Py_DECREF(arr_plevo);
  Py_DECREF(arr_psfc);
  Py_DECREF(arr_p0);
/*
 * Supposedly we don't need to decref ret_obj because we're using
 * PyArray_Return. 
*/
  return PyArray_Return(ret_obj);
  
fail:
  Py_XDECREF(arr_datai);
  Py_XDECREF(arr_hbcofa);
  Py_XDECREF(arr_hbcofb);
  Py_XDECREF(arr_plevo);
  Py_XDECREF(arr_psfc);
  Py_XDECREF(arr_p0);
  Py_INCREF(Py_None);
  return Py_None;
}

