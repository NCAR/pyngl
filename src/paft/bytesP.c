#include <string.h>
#include <stdio.h>

extern void NGCALLF(gbytes,GBYTES)(int *, int *, int *, int *, int *, int *);
extern void NGCALLF(sbytes,SBYTES)(int *, int *, int *, int *, int *, int *);

PyObject *fplib_dim_gbits(PyObject *self, PyObject *args)

{
/*  
 * Input.
 */
  PyObject *arr_npack = NULL, *arr_ibit = NULL;
  PyObject *arr_nbits = NULL, *arr_nskip = NULL;
  PyObject *arr_iter = NULL;
  int *npack, *ibit, *nbits, *nskip, *iter;
  int ndims_npack, type_npack, size_type_npack; 
  npy_intp *dsizes_npack;
  int *tmp_npack, *tmp_isam;
  int n, tmp_ibit, tmp_nbits, tmp_nskip;

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
 * Get the type and type size of the first argument. 
 */
  type_npack      = PyArray_TYPE(arr_npack);
  size_type_npack = PyArray_ITEMSIZE(arr_npack);
/*
 * Check type of npack.
 */
  if(type_npack != PyArray_BYTE  && type_npack != PyArray_UBYTE && 
     type_npack != PyArray_SHORT && type_npack != PyArray_USHORT && 
     type_npack != PyArray_INT   && type_npack != PyArray_INT32) {
    printf("dim_gbits: npack must be of type byte, unsigned byte, short, unsigned short, or int");
    Py_INCREF(Py_None);
    return Py_None;
  }

/*
 * Start extracting array information.
 *
 * Get argument # 0
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromObject
                        (arr_npack,PyArray_INT32,0,0);
  npack      = (int *)arr->data;
  ndims_npack = arr->nd;
/*
 * Check if we have a numpy scalar. If so, fix the number of dimensions.
 */
  if(ndims_npack == 0) {
    ndims_npack = 1;
    dsizes_npack = (npy_intp *)malloc(ndims_npack * sizeof(npy_intp));
    dsizes_npack[0] = 1;
  }
  else {
    dsizes_npack = (npy_intp *)malloc(ndims_npack * sizeof(npy_intp));
    for(i = 0; i < ndims_npack; i++) {
      dsizes_npack[i] = (npy_intp)arr->dimensions[i];
    }
  }
/*
 * Get arguments # 1-4
 */
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_ibit,PyArray_INT32,0,0);
  ibit = (int *)arr->data;
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_nbits,PyArray_INT32,0,0);
  nbits = (int *)arr->data;
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_nskip,PyArray_INT32,0,0);
  nskip = (int *)arr->data;
  arr = (PyArrayObject *) PyArray_ContiguousFromAny
                        (arr_iter,PyArray_INT32,0,0);
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
  if(type_npack != PyArray_INT && type_npack != PyArray_INT32) {
    tmp_npack = (int*)calloc(n,sizeof(int));
    tmp_isam  = (int*)calloc(*iter,sizeof(int));

    if(tmp_npack == NULL || tmp_isam == NULL) {
      printf("dim_gbits: Unable to allocate memory for temporary input/output arrays");
      Py_INCREF(Py_None);
      return Py_None;
    }
    if(type_npack == PyArray_BYTE) {
      isam = (void*)calloc(size_isam,sizeof(char));
    }
    else if(type_npack == PyArray_UBYTE) {
      isam = (void*)calloc(size_isam,sizeof(unsigned char));
    }
    else if(type_npack == PyArray_SHORT) {
      isam = (void*)calloc(size_isam,sizeof(short));
    }
    else {   /* if(type_npack == PyArray_USHORT) */
      isam = (void*)calloc(size_isam,sizeof(unsigned short));
    }
  }
  else {
    isam = (void*)calloc(size_isam,sizeof(int));
  }
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
    if(type_npack != PyArray_INT && type_npack != PyArray_INT32) {
/*
 * Run npack through sbytes to pack it into the leftmost bits of 
 * tmp_npack.
 */ 
      tmp_ibit  = 0;
      tmp_nskip = 0;
      tmp_nbits = 8*size_type_npack;
      NGCALLF(sbytes,SBYTES)(tmp_npack,&((int*)npack)[index_npack],
			     &tmp_ibit,&tmp_nbits,&tmp_nskip,&n);
    }
    else {
/*
 * npack is already an integer.
 */
      tmp_npack = &((int*)npack)[index_npack];
      tmp_isam  = &((int*)isam)[index_isam];
    }

    NGCALLF(gbytes,GBYTES)(tmp_npack,tmp_isam,ibit,nbits,nskip,iter);
/*
 * Coerce back to appropriate type.
 */
    if(type_npack == PyArray_BYTE) {
      for(j = 0; j < *iter; j++ ) {
        ((char*)isam)[index_isam+j] = (char)(tmp_isam[j]);
      }
    }
    else if(type_npack == PyArray_UBYTE) {
      for(j = 0; j < *iter; j++ ) {
        ((unsigned char*)isam)[index_isam+j] = (unsigned char)(tmp_isam[j]);
      }
    }
    else if(type_npack == PyArray_SHORT) {
      for(j = 0; j < *iter; j++ ) {
        ((short*)isam)[index_isam+j] = (short)(tmp_isam[j]);
      }
    }
    else if(type_npack == PyArray_USHORT) {
      for(j = 0; j < *iter; j++ ) {
        ((unsigned short*)isam)[index_isam+j] = (unsigned short)(tmp_isam[j]);
      }
    }
    index_npack += n;
    index_isam  += *iter;
  }

/*
 * Return.
 */
  ret = (PyArrayObject *) PyArray_SimpleNewFromData(ndims_npack,
						    dsizes_isam,
						    type_npack,
						    (void *) isam);

  PyArray_Return(ret);
}

