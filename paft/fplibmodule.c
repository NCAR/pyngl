#include "Python.h"
#include <stdio.h>
#include <string.h>
#include <Numeric/arrayobject.h>
#include "Error.h"

extern void NGCALLF(dlinmsg,DLINMSG)(double *,int *,double *,int *, int *);
extern void NGCALLF(chisub,CHISUB)(double *, double *, double *);

#ifndef NGCALLF

#define NGCALLF(reg,caps)   reg##_ 

#endif  /* NGCALLF */                                     

#include "chiinvP.c"
#include "linmsgP.c"

static PyMethodDef fplib_methods[] = {     
    {"chiinv",  fplib_chiinv, METH_VARARGS | METH_KEYWORDS},
    {"linmsg",  (PyCFunction)fplib_linmsg, METH_VARARGS | METH_KEYWORDS},
    {NULL,      NULL}        /* Sentinel */
};

void initfplib()
{
    (void) Py_InitModule("fplib", fplib_methods);
    import_array();
}
