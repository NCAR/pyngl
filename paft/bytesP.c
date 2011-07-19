#include <string.h>
#include <stdio.h>

extern void NGCALLF(gbytes,GBYTES)(int *, int *, int *, int *, int *, int *);

PyObject *fplib_dim_gbits(PyObject *self, PyObject *args)

{
/*  
 * Input.
 */
  PyObject *arr_npack = NULL, *arr_ibit = NULL;
  PyObject *arr_nbits = NULL, *arr_nskip = NULL;
  PyObject *arr_iter = NULL;
  int *npack, *ibit, *nbits, *nskip, *iter;
  int ndims_npack;
  npy_intp n, *dsizes_npack;
/*  
 * Output.
 */
  int *isam;
  npy_intp *dsizes_isam;

/*
 * Various.
 */
  npy_intp i, j, size_leftmost, size_isam;
  npy_intp index_npack = 0, index_isam = 0;

/*
 * Define some PyArray objects for input and output.
 */
  PyArrayObject *arr;
  PyArrayObject *ret;

/*
 * Retrieve arguments.
 */
  if (!PyArg_ParseTuple(args,(char *)"OOOOO:dim_gbits",&arr_npack,&arr_ibit,
                        &arr_nbits,&arr_nskip,&arr_iter)) {
    printf("dim_gbits: fatal: argument parsing failed\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Start extracting array information.
 *
 * Get argument # 0
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromObject
                        (arr_npack,PyArray_INT,0,0);
  npack      = (int *)arr->data;
  ndims_npack = arr->nd;
  dsizes_npack = (npy_intp *)malloc(ndims_npack * sizeof(npy_intp));
  for(i = 0; i < ndims_npack; i++) {
    dsizes_npack[i] = (npy_intp)arr->dimensions[i];
  }

/*
 * Get arguments # 1-4
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_ibit,PyArray_INT,0,0);
  ibit = (int *)arr->data;
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_nbits,PyArray_INT,0,0);
  nbits = (int *)arr->data;
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_nskip,PyArray_INT,0,0);
  nskip = (int *)arr->data;
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_iter,PyArray_INT,0,0);
  iter = (int *)arr->data;

/*
 * Compute total number of elements in npack and isam arrays.
 */
  size_leftmost = 1;
  for(i = 0; i < ndims_npack-1; i++) {
    size_leftmost *= dsizes_npack[i];
  }
  size_isam = *iter * size_leftmost;

  if(dsizes_npack[ndims_npack-1] > INT_MAX) {
    printf("dim_gbits: the rightmost dimension of npack is greater than INT_MAX");
    Py_INCREF(Py_None);
    return Py_None;
  }
  n = (int)dsizes_npack[ndims_npack-1];

/*
 * Allocate space for input/output arrays.
 */
  isam = (void*)calloc(size_isam,sizeof(int));
  if(isam == NULL) {
    printf("dim_gbits: Unable to allocate memory for output array");
    Py_INCREF(Py_None);
    return Py_None;
  }
/*
 * Allocate space for dimension sizes of output array.
 */
  dsizes_isam = (npy_intp *)calloc(ndims_npack,sizeof(npy_intp));
  if(dsizes_isam == NULL) {
    printf("dim_gbits: Unable to allocate memory for holding size of output array");
    Py_INCREF(Py_None);
    return Py_None;
  }

  for(i = 0; i < ndims_npack-1; i++) {
    dsizes_isam[i] = dsizes_npack[i];
  }
  dsizes_isam[ndims_npack-1] = *iter;

/*
 * Call the Fortran routine. 
 */

  for(i = 1; i <= size_leftmost; i++) {
    NGCALLF(gbytes,GBYTES)(&npack[index_npack],&isam[index_isam],
			   ibit,nbits,nskip,iter);
    index_npack += n;
    index_isam  += *iter;
  }

/*
 * Return.
 */
  ret = (PyArrayObject *) PyArray_SimpleNewFromData(ndims_npack,
						    dsizes_isam,
						    PyArray_INT,
						    (void *) isam);

  PyArray_Return(ret);
}

