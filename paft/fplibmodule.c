#include "Python.h"
#include <stdio.h>
#include <string.h>

#ifdef USE_NUMPY
#include <numpy/arrayobject.h>
#else
#include <Numeric/arrayobject.h>
#endif

#include <ncarg/hlu/Error.h>

static PyObject *t_output_helper(PyObject *, PyObject *);

extern void NGCALLF(dlinmsg,DLINMSG)(double *,int *,double *,int *, int *);
extern void NGCALLF(chisub,CHISUB)(double *, double *, double *);
extern void NGCALLF(dregcoef,DREGCOEF)(double *,double *,int *,double *,
                                       double *,double *,double *,int *,
                                       double *,double *,double *,int *);

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

#include "chiinvP.c"
#include "linmsgP.c"
#include "reglineP.c"

static PyMethodDef fplib_methods[] = {     
    {"chiinv",  (PyCFunction)fplib_chiinv,  METH_VARARGS},
    {"linmsg",  (PyCFunction)fplib_linmsg,  METH_VARARGS},
    {"regline", (PyCFunction)fplib_regline, METH_VARARGS},
    {NULL,      NULL}        /* Sentinel */
};

void initfplib()
{
    (void) Py_InitModule("fplib", fplib_methods);
    import_array();
}
