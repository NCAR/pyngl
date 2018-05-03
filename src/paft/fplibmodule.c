#include "Python.h"
#include <stdio.h>
#include <string.h>

#include <numpy/arrayobject.h>
#include <ncarg/hlu/Error.h>

static PyObject *t_output_helper(PyObject *, PyObject *);

extern void NGCALLF(betainc,BETAINC)(double*,double*,double*,double*);
extern void NGCALLF(dlinmsg,DLINMSG)(double *,int *,double *,int *, int *);
extern void NGCALLF(dlinmsg,DLINMSG)(double *,int *,double *,int *, int *);
extern void NGCALLF(chisub,CHISUB)(double *, double *, double *);
extern void NGCALLF(dregcoef,DREGCOEF)(double *,double *,int *,double *,
                                       double *,double *,double *,int *,
                                       double *,double *,double *,int *);

extern void NGCALLF(dint2p,DINT2P)(double *,double *,double *,double *,
                                   int *,double *,double *,int *,int *,
                                   double *,int*);

extern void NGCALLF(vinth2p,VINTH2P)(double *, double *, double *, double *,
                                     double *, double *, double *,int *,
                                     int *, double *, double *, int *,
                                     int *, int *, int *, int *, int *);

extern int NGCALLF(gcinout,GCINOUT)(double*,double*,double*,double*,
                                     int*,double*);
extern void NGCALLF(gbytes,GBYTES)(int *, int *, int *, int *, int *, int *);
extern void NGCALLF(sbytes,SBYTES)(int *, int *, int *, int *, int *, int *);

/* WRF functions */
extern void NGCALLF(dcomputetk,DCOMPUTETK)(double *,double *,double *,int *);
extern void NGCALLF(dcomputetd,DCOMPUTETD)(double *,double *,double *,int *);
extern void NGCALLF(dcomputerh,DCOMPUTERH)(double *,double *,double *,
                                           double *,int *);
extern void NGCALLF(dcomputeseaprs,DCOMPUTESEAPRS)(int *,int *,int *,
                                                   double *,double *,
                                                   double *,double *,
                                                   double *,double *,
                                                   double *,double *);
extern void NGCALLF(calcdbz,CALCDBZ)(double *, double *, double *, double *,
                                     double *, double *, double *, int *, 
                                     int *, int *, int *, int *, int *);

extern void NGCALLF(dcomputeabsvort,DCOMPUTEABSVORT)(double *, double *,
                                                     double *, double *,
                                                     double *, double *,
                                                     double *, double *,
                                                     double *, int *, int *,
                                                     int *, int *, int *);
extern void NGCALLF(dcomputepv,DCOMPUTEPV)(double *, double *, double *, 
                                           double *, double *, double *, 
                                           double *, double *, double *, 
                                           double *, double *, int *, int *, 
                                           int *, int *, int *);

extern void NGCALLF(dlltoij,DLLTOIJ)(int *, double *, double *, double *, 
                                     double *, double *, double *, double *, 
                                     double *, double *, double *, double *, 
                                     double *, double *, double *, double *, 
                                     double *);

/* WRF utility functions */
extern void convert_to_hPa(double *pp, npy_intp np);
extern void var_zero(double *tmp_var, npy_intp n);
extern int is_scalar(int,npy_intp*);

/*
 * t_output_helper concatenates objects.  That is, 
 * if you have two objects "obj1" and "obj2" 
 * then  calling t_output_helper(obj1, obj2) will create
 * a two-element tuple with the two objects as elements.
 * If obj1 is a tuple to start with, then obj2 is added
 * as an additional final element to that tuple.
 */
static PyObject* t_output_helper(PyObject* target, PyObject* o) {
    PyObject*   o2;
    PyObject*   o3;

    if (!target) {
        target = o;
    } else if (target == Py_None) {
        Py_DECREF(Py_None);
        target = o;
    } else {
        if (!PyTuple_Check(target)) {
            o2 = target;
            target = PyTuple_New(1);
            PyTuple_SetItem(target, 0, o2);
        }
        o3 = PyTuple_New(1);
        PyTuple_SetItem(o3, 0, o);

        o2 = target;
        target = PySequence_Concat(o2, o3);
        Py_DECREF(o2);
        Py_DECREF(o3);
    }
    return target;
}

#ifndef NGCALLF

#define NGCALLF(reg,caps)   reg##_ 

#endif  /* NGCALLF */                                     

#include "betaincP.c"
#include "bytesP.c"
#include "chiinvP.c"
#include "linmsgP.c"
#include "int2pP.c"
#include "reglineP.c"
#include "sgtoolsP.c"
#include "vinth2pP.c"
#include "wrfP.c"

static PyMethodDef fplib_methods[] = {     
    {"betainc", (PyCFunction)fplib_betainc, METH_VARARGS},
    {"dim_gbits",  (PyCFunction)fplib_dim_gbits,  METH_VARARGS},
    {"chiinv",  (PyCFunction)fplib_chiinv,  METH_VARARGS},
    {"gc_inout", (PyCFunction)fplib_gc_inout, METH_VARARGS},
    {"linmsg",  (PyCFunction)fplib_linmsg,  METH_VARARGS},
    {"int2p",  (PyCFunction)fplib_int2p,  METH_VARARGS},
    {"regline", (PyCFunction)fplib_regline, METH_VARARGS},
    {"vinth2p", (PyCFunction)fplib_vinth2p, METH_VARARGS},
    {"wrf_avo", (PyCFunction)fplib_wrf_avo, METH_VARARGS},
    {"wrf_pvo", (PyCFunction)fplib_wrf_pvo, METH_VARARGS},
    {"wrf_dbz", (PyCFunction)fplib_wrf_dbz, METH_VARARGS},
    {"wrf_rh", (PyCFunction)fplib_wrf_rh, METH_VARARGS},
    {"wrf_slp", (PyCFunction)fplib_wrf_slp, METH_VARARGS},
    {"wrf_td", (PyCFunction)fplib_wrf_td, METH_VARARGS},
    {"wrf_tk", (PyCFunction)fplib_wrf_tk, METH_VARARGS},
    {"wrf_ll_to_ij", (PyCFunction)fplib_wrf_ll_to_ij, METH_VARARGS},
    {"wrf_ij_to_ll", (PyCFunction)fplib_wrf_ij_to_ll, METH_VARARGS},
    {NULL,      NULL}        /* Sentinel */
};

void initfplib()
{
    (void) Py_InitModule("fplib", fplib_methods);
    import_array();
}
