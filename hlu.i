%module hlu

%{
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <ncarg/hlu/hluP.h>
#include <ncarg/hlu/ResListP.h>
#include <ncarg/hlu/AppI.h>
#include <ncarg/hlu/CoordArraysP.h>
#include <ncarg/hlu/TickMark.h>
#include <ncarg/hlu/Title.h>
#include <ncarg/hlu/Workstation.h>
#include <ncarg/hlu/WorkstationP.h>
#include <ncarg/hlu/XWorkstationP.h>
#include <ncarg/hlu/PSWorkstationP.h>
#include <ncarg/hlu/PDFWorkstationP.h>
#include <ncarg/hlu/LabelBar.h>
#include <ncarg/hlu/Legend.h>
#include <ncarg/hlu/TextItem.h>

/***********************************************************************
 *
 *  Include gsun header
 *
 ***********************************************************************/
#include "gsun.h"

/***********************************************************************
 *
 *  End GSUN functions
 *
 ***********************************************************************/

#include <ncarg/gks.h>

#include <Numeric/arrayobject.h>

#define min(x,y) ((x) < (y) ? (x) : (y) )
#define pow2(x)  ((x)*(x))

extern float *c_natgrids(int, float [], float [], float [],
                          int, int, float [], float [], int *);
int c_ftcurv (int, float [], float [], int, float [], float []);
int c_ftcurvp (int, float [], float [], float, int, float [], float []);
int c_ftcurvpi (float, float, float, int, float [], float [], float *);
double c_dcapethermo(double *, double *, int, double, int, 
                     double **, double, int *, int *, int *);
extern void NGCALLF(dptlclskewt,DPTLCLSKEWT)(double *, double *, double *,
                                             double *, double *);
extern NhlErrorTypes NglGaus(int, double **output);
extern void NglVinth2p (double *dati, int, int, int, double *[], double *,
                 double *, double, double *, double *, int, int, double *,
                 double, int, int, int);
extern void c_nnseti(NhlString,int);
extern void c_nnsetrd(NhlString,double);
extern void c_nnsetc(NhlString,NhlString);

static PyObject* t_output_helper(PyObject *, PyObject *);

struct common1 {
  float pang, plat, plon;
} NGCALLF(pcmp04,PCMP04);

nglRes nglRlist;

void set_nglRes_i(int pos, int ival) {
  if (pos == 0) {
    nglRlist.nglMaximize = ival;
  }
  else if (pos == 1) {
    nglRlist.nglDraw = ival;
  }
  else if (pos == 2) {
    nglRlist.nglFrame = ival;
  }
  else if (pos == 3) {
    nglRlist.nglScale = ival;
  }
  else if (pos == 4) {
    nglRlist.nglDebug = ival;
  }
  else if (pos == 5) {
    nglRlist.nglPaperOrientation = ival;
  }
  else if (pos == 9) {
    nglRlist.nglPanelCenter = ival;
  }
  else if (pos == 10) {
    nglRlist.nglPanelRowSpec = ival;
  }
  else if (pos == 13) {
    nglRlist.nglPanelBoxes = ival;
  }
  else if (pos == 22) {
    nglRlist.nglPanelSave = ival;
  }
  else if (pos == 23) {
    nglRlist.nglSpreadColors = ival;
  }
  else if (pos == 24) {
    nglRlist.nglSpreadColorStart = ival;
  }
  else if (pos == 25) {
    nglRlist.nglSpreadColorEnd = ival;
  }
  else if (pos == 26) {
    nglRlist.nglPanelLabelBarOrientation = ival;
  }
  else if (pos == 27) {
    nglRlist.nglPanelLabelBar = ival;
  }
  else if (pos == 35) {
    nglRlist.nglPanelLabelBarPerimOn = ival;
  }
  else if (pos == 36) {
    nglRlist.nglPanelLabelBarAlignment = ival;
  }
  else if (pos == 37) {
    nglRlist.nglPanelLabelBarLabelAutoStride = ival;
  }
  else if (pos == 39) {
    nglRlist.nglPanelFigureStringsCount = ival;
  }
  else if (pos == 40) {
    nglRlist.nglPanelFigureStringsJust = ival;
  }
  else if (pos == 43) {
    nglRlist.nglPanelFigureStringsPerimOn = ival;
  }
  else if (pos == 44) {
    nglRlist.nglPanelFigureStringsBackgroundFillColor = ival;
  }
  else if (pos == 47) {
    nglRlist.nglXAxisType = ival;
  }
  else if (pos == 48) {
    nglRlist.nglYAxisType = ival;
  }
  else {
    printf("set_nglRes_i: invalid argument %d\n",pos);
  }
}

int get_nglRes_i(int pos) {
  if (pos == 0) {
    return(nglRlist.nglMaximize);
  }
  else if (pos == 1) {
    return(nglRlist.nglDraw);
  }
  else if (pos == 2) {
    return(nglRlist.nglFrame);
  }
  else if (pos == 3) {
    return(nglRlist.nglScale);
  }
  else if (pos == 4) {
    return(nglRlist.nglDebug);
  }
  else if (pos == 5) {
    return(nglRlist.nglPaperOrientation);
  }
  else if (pos == 9) {
    return(nglRlist.nglPanelCenter);
  }
  else if (pos == 10) {
    return(nglRlist.nglPanelRowSpec);
  }
  else if (pos == 13) {
    return(nglRlist.nglPanelBoxes);
  }
  else if (pos == 22) {
    return(nglRlist.nglPanelSave);
  }
  else if (pos == 23) {
    return(nglRlist.nglSpreadColors);
  }
  else if (pos == 24) {
    return(nglRlist.nglSpreadColorStart);
  }
  else if (pos == 25) {
    return(nglRlist.nglSpreadColorEnd);
  }
  else if (pos == 26) {
    return(nglRlist.nglPanelLabelBarOrientation);
  }
  else if (pos == 27) {
    return(nglRlist.nglPanelLabelBar);
  }
  else if (pos == 35) {
    return(nglRlist.nglPanelLabelBarPerimOn);
  }
  else if (pos == 36) {
    return(nglRlist.nglPanelLabelBarAlignment);
  }
  else if (pos == 37) {
    return(nglRlist.nglPanelLabelBarLabelAutoStride);
  }
  else if (pos == 39) {
    return(nglRlist.nglPanelFigureStringsCount);
  }
  else if (pos == 40) {
    return(nglRlist.nglPanelFigureStringsJust);
  }
  else if (pos == 43) {
    return(nglRlist.nglPanelFigureStringsPerimOn);
  }
  else if (pos == 44) {
    return(nglRlist.nglPanelFigureStringsBackgroundFillColor);
  }
  else if (pos == 47) {
    return(nglRlist.nglXAxisType);
  }
  else if (pos == 48) {
    return(nglRlist.nglYAxisType);
  }
  else {
    printf("get_nglRes_i: invalid argument %d\n",pos);
  }
}

void set_nglRes_f(int pos, float ival) {
  if (pos == 6) {
    nglRlist.nglPaperWidth = ival;
  }
  else if (pos == 7) {
    nglRlist.nglPaperHeight = ival;
  }
  else if (pos == 8) {
    nglRlist.nglPaperMargin = ival;
  }
  else if (pos == 11) {
    nglRlist.nglPanelXWhiteSpacePercent = ival;
  }
  else if (pos == 12) {
    nglRlist.nglPanelYWhiteSpacePercent = ival;
  }
  else if (pos == 14) {
    nglRlist.nglPanelLeft = ival;
  }
  else if (pos == 15) {
    nglRlist.nglPanelRight = ival;
  }
  else if (pos == 16) {
    nglRlist.nglPanelBottom = ival;
  }
  else if (pos == 17) {
    nglRlist.nglPanelTop = ival;
  }
  else if (pos == 18) {
    nglRlist.nglPanelInvsblTop = ival;
  }
  else if (pos == 19) {
    nglRlist.nglPanelInvsblLeft = ival;
  }
  else if (pos == 20) {
    nglRlist.nglPanelInvsblRight = ival;
  }
  else if (pos == 21) {
    nglRlist.nglPanelInvsblBottom = ival;
  }
  else if (pos == 28) {
    nglRlist.nglPanelLabelBarXF = ival;
  }
  else if (pos == 29) {
    nglRlist.nglPanelLabelBarYF = ival;
  }
  else if (pos == 30) {
    nglRlist.nglPanelLabelBarLabelFontHeightF = ival;
  }
  else if (pos == 31) {
    nglRlist.nglPanelLabelBarWidthF = ival;
  }
  else if (pos == 32) {
    nglRlist.nglPanelLabelBarHeightF = ival;
  }
  else if (pos == 33) {
    nglRlist.nglPanelLabelBarOrthogonalPosF = ival;
  }
  else if (pos == 34) {
    nglRlist.nglPanelLabelBarParallelPosF = ival;
  }
  else if (pos == 41) {
    nglRlist.nglPanelFigureStringsOrthogonalPosF = ival;
  }
  else if (pos == 42) {
    nglRlist.nglPanelFigureStringsParallelPosF = ival;
  }
  else if (pos == 45) {
    nglRlist.nglPanelFigureStringsFontHeightF = ival;
  }
  else {
    printf ("set_nglRes_f: invalid argument %d\n",pos);
  }
}

float get_nglRes_f(int pos) {
  if (pos == 6) {
    return(nglRlist.nglPaperWidth);
  }
  else if (pos == 7) {
    return(nglRlist.nglPaperHeight);
  }
  else if (pos == 8) {
    return(nglRlist.nglPaperMargin);
  }
  else if (pos == 11) {
    return(nglRlist.nglPanelXWhiteSpacePercent);
  }
  else if (pos == 12) {
    return(nglRlist.nglPanelYWhiteSpacePercent);
  }
  else if (pos == 14) {
    return(nglRlist.nglPanelLeft);
  }
  else if (pos == 15) {
    return(nglRlist.nglPanelRight);
  }
  else if (pos == 16) {
    return(nglRlist.nglPanelBottom);
  }
  else if (pos == 17) {
    return(nglRlist.nglPanelTop);
  }
  else if (pos == 18) {
    return(nglRlist.nglPanelInvsblTop);
  }
  else if (pos == 19) {
    return(nglRlist.nglPanelInvsblLeft);
  }
  else if (pos == 20) {
    return(nglRlist.nglPanelInvsblRight);
  }
  else if (pos == 21) {
    return(nglRlist.nglPanelInvsblBottom);
  }
  else if (pos == 28) {
    return(nglRlist.nglPanelLabelBarXF);
  }
  else if (pos == 29) {
    return(nglRlist.nglPanelLabelBarYF);
  }
  else if (pos == 30) {
    return(nglRlist.nglPanelLabelBarLabelFontHeightF);
  }
  else if (pos == 31) {
    return(nglRlist.nglPanelLabelBarWidthF);
  }
  else if (pos == 32) {
    return(nglRlist.nglPanelLabelBarHeightF);
  }
  else if (pos == 33) {
    return(nglRlist.nglPanelLabelBarOrthogonalPosF);
  }
  else if (pos == 34) {
    return(nglRlist.nglPanelLabelBarParallelPosF);
  }
  else if (pos == 41) {
    return(nglRlist.nglPanelFigureStringsOrthogonalPosF);
  }
  else if (pos == 42) {
    return(nglRlist.nglPanelFigureStringsParallelPosF);
  }
  else if (pos == 45) {
    return(nglRlist.nglPanelFigureStringsFontHeightF);
  }
  else {
    printf ("get_nglRes_f: invalid argument %d\n",pos);
  }
}

void set_nglRes_c (int pos, NhlString *cval) {
  if (pos == 38) {
    nglRlist.nglPanelFigureStrings = cval;    
  }
  else {
    printf ("set_nglRes_c: invalid argument %d\n",pos);
  }
}

NhlString *get_nglRes_c (int pos) {
  if (pos == 38) {
    return(nglRlist.nglPanelFigureStrings);
  }
  else {
    printf ("get_nglRes_c: invalid argument %d\n",pos);
  }
}

void set_nglRes_s (int pos, NhlString sval) {
  if (pos == 46) {
    nglRlist.nglAppResFileName = sval;    
  }
  else {
    printf ("set_nglRes_s: invalid argument %s\n",sval);
  }
}

NhlString get_nglRes_s (int pos) {
  if (pos == 46) {
    return(nglRlist.nglAppResFileName);
  }
  else {
    printf ("get_nglRes_s: invalid argument %d\n",pos);
  }
}

void set_PCMP04(int arg_num, float value)
{
  if (arg_num < 1 || arg_num > 3) {
    printf("Error in argument number in common PCMP04\n");
    exit(70);
  }
  switch(arg_num) {
    case 1:
      NGCALLF(pcmp04,PCMP04).pang = value;
    case 2:
      NGCALLF(pcmp04,PCMP04).plat = value;
    case 3:
      NGCALLF(pcmp04,PCMP04).plon = value;
  }
}

NhlErrorTypes NglGaus_p(int num, int n, int m, double *data_out[]) {
  return NglGaus(num,data_out);
}

PyObject *mapgci(float alat, float alon, float blat, float blon, int npts)
{
  float *rlati,*rloni;
  PyObject *obj1,*obj2,*status,*resultobj;

  int dims[1],ier;

  rlati = (float *) malloc(npts*sizeof(float));
  rloni = (float *) malloc(npts*sizeof(float));
  c_mapgci(alat, alon, blat, blon, npts, rlati, rloni);
  dims[0] = npts;
  obj1 = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,
                                              (char *) rlati);
  obj2 = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,
                                              (char *) rloni);
  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,obj1);
  resultobj = t_output_helper(resultobj,obj2);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

PyObject *dcapethermo(double *penv, double *tenv, int nlvl, double lclmb, 
                      int iprnt, double tmsg)
{
  PyObject *obj1,*obj2,*obj3,*obj4,*obj5,*status,*resultobj;
  int jlcl, jlfc, jcross, dims[1];
  double cape, *tparcel;

  cape = c_dcapethermo(penv, tenv, nlvl, lclmb, iprnt, &tparcel, tmsg,
                       &jlcl, &jlfc, &jcross);                       

  dims[0] = nlvl;
  obj1 = (PyObject *) PyFloat_FromDouble(cape);
  obj2 = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_DOUBLE,
                                              (char *) tparcel);
  obj3 = (PyObject *) PyInt_FromLong((long) jlcl);
  obj4 = (PyObject *) PyInt_FromLong((long) jlfc);
  obj5 = (PyObject *) PyInt_FromLong((long) jcross);

  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,obj1);
  resultobj = t_output_helper(resultobj,obj2);
  resultobj = t_output_helper(resultobj,obj3);
  resultobj = t_output_helper(resultobj,obj4);
  resultobj = t_output_helper(resultobj,obj5);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

PyObject *ftcurvc(int n, float *x, float *y, int m, float *xo)
{
  float *yo;
  PyObject *obj1,*status,*resultobj;

  int dims[1],ier;

  yo = (float *) malloc(m*sizeof(float));
  ier = c_ftcurv(n,x,y,m,xo,yo);
  status = (PyObject *) PyInt_FromLong((long) ier);
  dims[0] = m;
  obj1 = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,
                                              (char *) yo);
  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,status);
  resultobj = t_output_helper(resultobj,obj1);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

PyObject *ftcurvpc(int n, float *x, float *y, float p, int m, float *xo)
{
  float *yo;
  PyObject *obj1,*status,*resultobj;

  int dims[1],ier;

  yo = (float *) malloc(m*sizeof(float));
  ier = c_ftcurvp(n,x,y,p,m,xo,yo);
  status = (PyObject *) PyInt_FromLong((long) ier);
  dims[0] = m;
  obj1 = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,
                                              (char *) yo);
  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,status);
  resultobj = t_output_helper(resultobj,obj1);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

PyObject *ftcurvpic(float xl, float xr, float p, int m, float *xi, float *yi)
{
  float yo;
  PyObject *obj1,*status,*resultobj;

  int dims[1],ier;

  ier = c_ftcurvpi(xl,xr,p,m,xi,yi,&yo);
  status = (PyObject *) PyInt_FromLong((long) ier);
  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,status);
  resultobj = t_output_helper(resultobj,
                 (PyObject *) PyFloat_FromDouble((double) yo));
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}


void natgridc(int n, float *x, float *y, float *z, int nxi, int nyi,
             float *xi, float *yi, int *ier, int nxir, int nyir, float *aout[])
{
    aout[0] = c_natgrids(n,x,y,z,nxi,nyi,xi,yi,ier);
}

int NhlGetInteger(int oid, char *name)
{
    int grlist;
    int valueI;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetInteger(grlist,name,&valueI);
    NhlGetValues(oid,grlist);

    return (valueI);
}

float NhlGetFloat(int oid, char *name)
{
    int grlist;
    float valueF;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetFloat(grlist,name,&valueF);
    NhlGetValues(oid,grlist);

    return (valueF);
}

float NhlGetDouble(int oid, char *name)
{
    int grlist;
    double valueD;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetDouble(grlist,name,&valueD);
    NhlGetValues(oid,grlist);

    return ((float) valueD);
}

NhlString NhlGetString(int oid, NhlString name)
{
    int grlist;
    NhlString valueS;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetString(grlist,name,&valueS);
    NhlGetValues(oid,grlist);

    return (valueS);
}

float *NhlGetFloatArray(int oid, char *name, int *number)
{
    int grlist;
    float *fscales;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetFloatArray(grlist,name,&fscales,number);
    NhlGetValues(oid,grlist);

    return (fscales);
}

double *NhlGetDoubleArray(int oid, char *name, int *number)
{
    int grlist;
    double *dar;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetDoubleArray(grlist,name,&dar,number);
    NhlGetValues(oid,grlist);

    return (dar);
}

int *NhlGetIntegerArray(int oid, char *name, int *number)
{
    int grlist;
    int *iar;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetIntegerArray(grlist,name,&iar,number);
    NhlGetValues(oid,grlist);

    return (iar);
}

NhlString *NhlGetStringArray(int oid, char *name, int *number)
{
    int grlist;
    NhlString *slist;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetStringArray(grlist,name,&slist,number);
    NhlGetValues(oid,grlist);

    return (slist);
}

NhlErrorTypes NhlPGetBB (int pid, float *t, float *b, float *l, float *r) {
  NhlBoundingBox *Box;
  NhlErrorTypes rv;
  Box = (NhlBoundingBox *) malloc(sizeof(NhlBoundingBox));
  rv = NhlGetBB(pid,Box);
  *t = Box->t;
  *b = Box->b;
  *l = Box->l;
  *r = Box->r;
  return(rv);
}

NhlErrorTypes NhlQNDCToData(int pid, float *x, float *y, int n, 
                            float *xout, float *yout, float xmissing, 
                            float ymissing, int ixmissing, int iymissing,
                            int *status, float *out_of_range)
{
  float *mxt, *myt;
  if (ixmissing != 1) {
    mxt = NULL;
  }
  else {
    mxt = malloc(sizeof(float));
    *mxt = xmissing;
  }
  if (iymissing != 1) {
    myt = NULL;
  }
  else {
    myt = malloc(sizeof(float));
    *myt = ymissing;
  }
  return(NhlNDCToData(pid, x, y, n, xout, yout, mxt, myt,
                      status, out_of_range));
}

PyObject *NhlPNDCToData(int pid, float *x, float *y, int n, float xmissing,
                        float ymissing, int ixmissing, int iymissing)
{
  float *xout, *yout;
  PyObject *obj1, *obj2, *nhlerr, *rstatus, *range, *resultobj;
  int dims[1],status,rval;
  float out_of_range;

  xout = (float *) malloc(n*sizeof(float));
  yout = (float *) malloc(n*sizeof(float));
  rval = (int) NhlQNDCToData(pid, x, y, n, xout, yout, xmissing,
                             ymissing, ixmissing, iymissing,
                             &status, &out_of_range);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 
  dims[0] = n;
  obj1 = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,
                                              (char *) xout);
  obj2 = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,
                                              (char *) yout);
  rstatus = (PyObject *) PyInt_FromLong((long) status); 
  range = (PyObject *) PyFloat_FromDouble((double) out_of_range); 

  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,nhlerr);
  resultobj = t_output_helper(resultobj,obj1);
  resultobj = t_output_helper(resultobj,obj2);
  resultobj = t_output_helper(resultobj,rstatus);
  resultobj = t_output_helper(resultobj,range);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

NhlErrorTypes NhlQDataToNDC(int pid, float *x, float *y, int n, 
                            float *xout, float *yout, float xmissing, 
                            float ymissing, int ixmissing, int iymissing,
                            int *status, float *out_of_range)
{
  float *mxt, *myt;
  if (ixmissing != 1) {
    mxt = NULL;
  }
  else {
    mxt = malloc(sizeof(float));
    *mxt = xmissing;
  }
  if (iymissing != 1) {
    myt = NULL;
  }
  else {
    myt = malloc(sizeof(float));
    *myt = ymissing;
  }
  return(NhlDataToNDC(pid, x, y, n, xout, yout, mxt, myt,
                      status, out_of_range));
}

PyObject *NhlPDataToNDC(int pid, float *x, float *y, int n, float xmissing,
                        float ymissing, int ixmissing, int iymissing)
{
  float *xout, *yout;
  PyObject *obj1, *obj2, *nhlerr, *rstatus, *range, *resultobj;
  int dims[1],status,rval;
  float out_of_range;

  xout = (float *) malloc(n*sizeof(float));
  yout = (float *) malloc(n*sizeof(float));
  rval = (int) NhlQDataToNDC(pid, x, y, n, xout, yout, xmissing,
                             ymissing, ixmissing, iymissing,
                             &status, &out_of_range);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 
  dims[0] = n;
  obj1 = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,
                                              (char *) xout);
  obj2 = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,
                                              (char *) yout);
  rstatus = (PyObject *) PyInt_FromLong((long) status); 
  range = (PyObject *) PyFloat_FromDouble((double) out_of_range); 

  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,nhlerr);
  resultobj = t_output_helper(resultobj,obj1);
  resultobj = t_output_helper(resultobj,obj2);
  resultobj = t_output_helper(resultobj,rstatus);
  resultobj = t_output_helper(resultobj,range);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

PyObject *NhlGetMDFloatArray(int pid, char *name) {
  PyObject *obj1, *nhlerr, *resultobj;
  int num_dims, *len_dims, grlist;
  float *bptr;
  NhlErrorTypes rval;

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  rval = NhlRLGetMDFloatArray(grlist,name,&bptr,&num_dims,&len_dims);
  NhlGetValues(pid,grlist);

  obj1 = (PyObject *) PyArray_FromDimsAndData(num_dims,len_dims,PyArray_FLOAT,
                                              (char *) bptr);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 

  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,nhlerr);
  resultobj = t_output_helper(resultobj,obj1);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

PyObject *NhlGetMDDoubleArray(int pid, char *name) {
  PyObject *obj1, *nhlerr, *resultobj;
  int num_dims, *len_dims, grlist;
  double *bptr;
  NhlErrorTypes rval;

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  rval = NhlRLGetMDDoubleArray(grlist,name,&bptr,&num_dims,&len_dims);
  NhlGetValues(pid,grlist);

  obj1 = (PyObject *) PyArray_FromDimsAndData(num_dims,len_dims,PyArray_DOUBLE,
                                              (char *) bptr);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 

  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,nhlerr);
  resultobj = t_output_helper(resultobj,obj1);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

PyObject *NhlGetMDIntegerArray(int pid, char *name) {
  PyObject *obj1, *nhlerr, *resultobj;
  int num_dims, *len_dims, grlist;
  int *bptr;
  NhlErrorTypes rval;

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  rval = NhlRLGetMDIntegerArray(grlist,name,&bptr,&num_dims,&len_dims);
  NhlGetValues(pid,grlist);

  obj1 = (PyObject *) PyArray_FromDimsAndData(num_dims,len_dims,PyArray_INT,
                                              (char *) bptr);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 

  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,nhlerr);
  resultobj = t_output_helper(resultobj,obj1);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

float *d2f(int isize, double *darray) {
  float *farray;
  int i;
  farray = (float *) malloc(isize*sizeof(float));
  for (i = 0; i < isize; i++) {
    farray[i] = (float) darray[i];
  }
  return (farray);
}

void gendat (int idim, int m, int n, int mlow, int mhgh, 
             float dlow, float dhgh, float *data)
{
/*
 * This is a routine to generate test data for two-dimensional graphics
 * routines.  Given an array "DATA", dimensioned "IDIM x 1", it fills
 * the sub-array ((DATA(I,J),I=1,M),J=1,N) with a two-dimensional field
 * of data having approximately "MLOW" lows and "MHGH" highs, a minimum
 * value of exactly "DLOW" and a maximum value of exactly "DHGH".
 *
 * "MLOW" and "MHGH" are each forced to be greater than or equal to 1
 * and less than or equal to 25.
 *
 * The function used is a sum of exponentials.
 */
    float ccnt[3][50], fovm, fovn, dmin, dmax, temp;
    extern float fran();
    int nlow, nhgh, ncnt, i, j, k, ii;

    fovm=9./(float)m;
    fovn=9./(float)n;

    nlow=max(1,min(25,mlow));
    nhgh=max(1,min(25,mhgh));
    ncnt=nlow+nhgh;

    for( k=1; k <= ncnt; k++ ) {
        ccnt[0][k-1]=1.+((float)m-1.)*fran();
        ccnt[1][k-1]=1.+((float)n-1.)*fran();
        if (k <= nlow) {
            ccnt[2][k-1]= -1.;
        }
        else {
            ccnt[2][k-1] = 1.;
        }
    }

    dmin =  1.e36;
    dmax = -1.e36;
    ii = 0;
    for( j = 1; j <= n; j++ ) {
        for( i = 1; i <= m; i++ ) {
            data[ii]=.5*(dlow+dhgh);
            for( k = 1; k <= ncnt; k++ ) {
                temp = -(pow2((fovm*((float)(i)-ccnt[0][k-1])))+
                         pow2(fovn*((float)(j)-ccnt[1][k-1])));
                if (temp >= -20.) data[ii]=data[ii]+.5*(dhgh-dlow)
                                           *ccnt[2][k-1]*exp(temp);
            }
            dmin=min(dmin,data[ii]);
            dmax=max(dmax,data[ii]);
            ii++;
        }
    }

    for( j = 0; j < m*n; j++ ) {
        data[j]=(data[j]-dmin)/(dmax-dmin)*(dhgh-dlow)+dlow;
    }
}

float rseq[] = { .749, .973, .666, .804, .081, .483, .919, .903, .951, .960,
   .039, .269, .270, .756, .222, .478, .621, .063, .550, .798, .027, .569,
   .149, .697, .451, .738, .508, .041, .266, .249, .019, .191, .266, .625,
   .492, .940, .508, .406, .972, .311, .757, .378, .299, .536, .619, .844,
   .342, .295, .447, .499, .688, .193, .225, .520, .954, .749, .997, .693,
   .217, .273, .961, .948, .902, .104, .495, .257, .524, .100, .492, .347,
   .981, .019, .225, .806, .678, .710, .235, .600, .994, .758, .682, .373,
   .009, .469, .203, .730, .588, .603, .213, .495, .884, .032, .185, .127,
   .010, .180, .689, .354, .372, .429 };

float fran()
{
    static int iseq = 0;
    iseq = (iseq % 100) + 1;
    return(rseq[iseq-1]);
}

void bndary()
{
/*
 * Draw a line showing where the edge of the plotter frame is.
 */
    c_plotif (0.,0.,0);
    c_plotif (1.,0.,1);
    c_plotif (1.,1.,1);
    c_plotif (0.,1.,1);
    c_plotif (0.,0.,1);
    c_plotif (0.,0.,2);
}

%}

%constant NhlBACKGROUND = 0;

%constant NhlTOPLEFT      = 0;
%constant NhlCENTERLEFT   = 1;
%constant NhlBOTTOMLEFT   = 2;
%constant NhlTOPCENTER    = 3;
%constant NhlCENTERCENTER = 4;
%constant NhlBOTTOMCENTER = 5;
%constant NhlTOPRIGHT     = 6;
%constant NhlCENTERRIGHT  = 7;
%constant NhlBOTTOMRIGHT  = 8;


%constant NhlNwkOrientation = "wkOrientation";
%constant NhlNamDataXF = "amDataXF";
%constant NhlNamDataYF = "amDataYF";
%constant NhlNamJust = "amJust";
%constant NhlNamOn = "amOn";
%constant NhlNamOrthogonalPosF = "amOrthogonalPosF";
%constant NhlNamParallelPosF = "amParallelPosF";
%constant NhlNamResizeNotify = "amResizeNotify";
%constant NhlNamSide = "amSide";
%constant NhlNamTrackData = "amTrackData";
%constant NhlNamViewId = "amViewId";
%constant NhlNamZone = "amZone";
%constant NhlNappDefaultParent = "appDefaultParent";
%constant NhlNappFileSuffix = "appFileSuffix";
%constant NhlNappResources = "appResources";
%constant NhlNappSysDir = "appSysDir";
%constant NhlNappUsrDir = "appUsrDir";
%constant NhlNcaCopyArrays = "caCopyArrays";
%constant NhlNcaXArray = "caXArray";
%constant NhlNcaXCast = "caXCast";
%constant NhlNcaXMaxV = "caXMaxV";
%constant NhlNcaXMinV = "caXMinV";
%constant NhlNcaXMissingV = "caXMissingV";
%constant NhlNcaYArray = "caYArray";
%constant NhlNcaYCast = "caYCast";
%constant NhlNcaYMaxV = "caYMaxV";
%constant NhlNcaYMinV = "caYMinV";
%constant NhlNcaYMissingV = "caYMissingV";
%constant NhlNcnConpackParams = "cnConpackParams";
%constant NhlNcnConstFLabelAngleF = "cnConstFLabelAngleF";
%constant NhlNcnConstFLabelBackgroundColor = "cnConstFLabelBackgroundColor";
%constant NhlNcnConstFLabelConstantSpacingF = "cnConstFLabelConstantSpacingF";
%constant NhlNcnConstFLabelFont = "cnConstFLabelFont";
%constant NhlNcnConstFLabelFontAspectF = "cnConstFLabelFontAspectF";
%constant NhlNcnConstFLabelFontColor = "cnConstFLabelFontColor";
%constant NhlNcnConstFLabelFontHeightF = "cnConstFLabelFontHeightF";
%constant NhlNcnConstFLabelFontQuality = "cnConstFLabelFontQuality";
%constant NhlNcnConstFLabelFontThicknessF = "cnConstFLabelFontThicknessF";
%constant NhlNcnConstFLabelFormat = "cnConstFLabelFormat";
%constant NhlNcnConstFLabelFuncCode = "cnConstFLabelFuncCode";
%constant NhlNcnConstFLabelJust = "cnConstFLabelJust";
%constant NhlNcnConstFLabelOn = "cnConstFLabelOn";
%constant NhlNcnConstFLabelOrthogonalPosF = "cnConstFLabelOrthogonalPosF";
%constant NhlNcnConstFLabelParallelPosF = "cnConstFLabelParallelPosF";
%constant NhlNcnConstFLabelPerimColor = "cnConstFLabelPerimColor";
%constant NhlNcnConstFLabelPerimOn = "cnConstFLabelPerimOn";
%constant NhlNcnConstFLabelPerimSpaceF = "cnConstFLabelPerimSpaceF";
%constant NhlNcnConstFLabelPerimThicknessF = "cnConstFLabelPerimThicknessF";
%constant NhlNcnConstFLabelSide = "cnConstFLabelSide";
%constant NhlNcnConstFLabelString = "cnConstFLabelString";
%constant NhlNcnConstFLabelTextDirection = "cnConstFLabelTextDirection";
%constant NhlNcnConstFLabelZone = "cnConstFLabelZone";
%constant NhlNcnConstFUseInfoLabelRes = "cnConstFUseInfoLabelRes";
%constant NhlNcnExplicitLabelBarLabelsOn = "cnExplicitLabelBarLabelsOn";
%constant NhlNcnExplicitLegendLabelsOn = "cnExplicitLegendLabelsOn";
%constant NhlNcnExplicitLineLabelsOn = "cnExplicitLineLabelsOn";
%constant NhlNcnFillBackgroundColor = "cnFillBackgroundColor";
%constant NhlNcnFillColor = "cnFillColor";
%constant NhlNcnFillColors = "cnFillColors";
%constant NhlNcnFillDrawOrder = "cnFillDrawOrder";
%constant NhlNcnFillOn = "cnFillOn";
%constant NhlNcnFillPattern = "cnFillPattern";
%constant NhlNcnFillPatterns = "cnFillPatterns";
%constant NhlNcnFillScaleF = "cnFillScaleF";
%constant NhlNcnFillScales = "cnFillScales";
%constant NhlNcnFixFillBleed = "cnFixFillBleed";
%constant NhlNcnGridBoundPerimColor = "cnGridBoundPerimColor";
%constant NhlNcnGridBoundPerimDashPattern = "cnGridBoundPerimDashPattern";
%constant NhlNcnGridBoundPerimOn = "cnGridBoundPerimOn";
%constant NhlNcnGridBoundPerimThicknessF = "cnGridBoundPerimThicknessF";
%constant NhlNcnHighLabelAngleF = "cnHighLabelAngleF";
%constant NhlNcnHighLabelBackgroundColor = "cnHighLabelBackgroundColor";
%constant NhlNcnHighLabelConstantSpacingF = "cnHighLabelConstantSpacingF";
%constant NhlNcnHighLabelFont = "cnHighLabelFont";
%constant NhlNcnHighLabelFontAspectF = "cnHighLabelFontAspectF";
%constant NhlNcnHighLabelFontColor = "cnHighLabelFontColor";
%constant NhlNcnHighLabelFontHeightF = "cnHighLabelFontHeightF";
%constant NhlNcnHighLabelFontQuality = "cnHighLabelFontQuality";
%constant NhlNcnHighLabelFontThicknessF = "cnHighLabelFontThicknessF";
%constant NhlNcnHighLabelFormat = "cnHighLabelFormat";
%constant NhlNcnHighLabelFuncCode = "cnHighLabelFuncCode";
%constant NhlNcnHighLabelPerimColor = "cnHighLabelPerimColor";
%constant NhlNcnHighLabelPerimOn = "cnHighLabelPerimOn";
%constant NhlNcnHighLabelPerimSpaceF = "cnHighLabelPerimSpaceF";
%constant NhlNcnHighLabelPerimThicknessF = "cnHighLabelPerimThicknessF";
%constant NhlNcnHighLabelString = "cnHighLabelString";
%constant NhlNcnHighLabelsOn = "cnHighLabelsOn";
%constant NhlNcnHighLowLabelOverlapMode = "cnHighLowLabelOverlapMode";
%constant NhlNcnHighUseLineLabelRes = "cnHighUseLineLabelRes";
%constant NhlNcnInfoLabelAngleF = "cnInfoLabelAngleF";
%constant NhlNcnInfoLabelBackgroundColor = "cnInfoLabelBackgroundColor";
%constant NhlNcnInfoLabelConstantSpacingF = "cnInfoLabelConstantSpacingF";
%constant NhlNcnInfoLabelFont = "cnInfoLabelFont";
%constant NhlNcnInfoLabelFontAspectF = "cnInfoLabelFontAspectF";
%constant NhlNcnInfoLabelFontColor = "cnInfoLabelFontColor";
%constant NhlNcnInfoLabelFontHeightF = "cnInfoLabelFontHeightF";
%constant NhlNcnInfoLabelFontQuality = "cnInfoLabelFontQuality";
%constant NhlNcnInfoLabelFontThicknessF = "cnInfoLabelFontThicknessF";
%constant NhlNcnInfoLabelFormat = "cnInfoLabelFormat";
%constant NhlNcnInfoLabelFuncCode = "cnInfoLabelFuncCode";
%constant NhlNcnInfoLabelJust = "cnInfoLabelJust";
%constant NhlNcnInfoLabelOn = "cnInfoLabelOn";
%constant NhlNcnInfoLabelOrthogonalPosF = "cnInfoLabelOrthogonalPosF";
%constant NhlNcnInfoLabelParallelPosF = "cnInfoLabelParallelPosF";
%constant NhlNcnInfoLabelPerimColor = "cnInfoLabelPerimColor";
%constant NhlNcnInfoLabelPerimOn = "cnInfoLabelPerimOn";
%constant NhlNcnInfoLabelPerimSpaceF = "cnInfoLabelPerimSpaceF";
%constant NhlNcnInfoLabelPerimThicknessF = "cnInfoLabelPerimThicknessF";
%constant NhlNcnInfoLabelSide = "cnInfoLabelSide";
%constant NhlNcnInfoLabelString = "cnInfoLabelString";
%constant NhlNcnInfoLabelTextDirection = "cnInfoLabelTextDirection";
%constant NhlNcnInfoLabelZone = "cnInfoLabelZone";
%constant NhlNcnLabelBarEndLabelsOn = "cnLabelBarEndLabelsOn";
%constant NhlNcnLabelDrawOrder = "cnLabelDrawOrder";
%constant NhlNcnLabelMasking = "cnLabelMasking";
%constant NhlNcnLabelScaleFactorF = "cnLabelScaleFactorF";
%constant NhlNcnLabelScaleValueF = "cnLabelScaleValueF";
%constant NhlNcnLabelScalingMode = "cnLabelScalingMode";
%constant NhlNcnLegendLevelFlags = "cnLegendLevelFlags";
%constant NhlNcnLevelCount = "cnLevelCount";
%constant NhlNcnLevelFlag = "cnLevelFlag";
%constant NhlNcnLevelFlags = "cnLevelFlags";
%constant NhlNcnLevelSelectionMode = "cnLevelSelectionMode";
%constant NhlNcnLevelSpacingF = "cnLevelSpacingF";
%constant NhlNcnLevels = "cnLevels";
%constant NhlNcnLineColor = "cnLineColor";
%constant NhlNcnLineColors = "cnLineColors";
%constant NhlNcnLineDashPattern = "cnLineDashPattern";
%constant NhlNcnLineDashPatterns = "cnLineDashPatterns";
%constant NhlNcnLineDashSegLenF = "cnLineDashSegLenF";
%constant NhlNcnLineDrawOrder = "cnLineDrawOrder";
%constant NhlNcnLineLabelAngleF = "cnLineLabelAngleF";
%constant NhlNcnLineLabelBackgroundColor = "cnLineLabelBackgroundColor";
%constant NhlNcnLineLabelConstantSpacingF = "cnLineLabelConstantSpacingF";
%constant NhlNcnLineLabelFont = "cnLineLabelFont";
%constant NhlNcnLineLabelFontAspectF = "cnLineLabelFontAspectF";
%constant NhlNcnLineLabelFontColor = "cnLineLabelFontColor";
%constant NhlNcnLineLabelFontColors = "cnLineLabelFontColors";
%constant NhlNcnLineLabelFontHeightF = "cnLineLabelFontHeightF";
%constant NhlNcnLineLabelFontQuality = "cnLineLabelFontQuality";
%constant NhlNcnLineLabelFontThicknessF = "cnLineLabelFontThicknessF";
%constant NhlNcnLineLabelFormat = "cnLineLabelFormat";
%constant NhlNcnLineLabelFuncCode = "cnLineLabelFuncCode";
%constant NhlNcnLineLabelInterval = "cnLineLabelInterval";
%constant NhlNcnLineLabelPerimColor = "cnLineLabelPerimColor";
%constant NhlNcnLineLabelPerimOn = "cnLineLabelPerimOn";
%constant NhlNcnLineLabelPerimSpaceF = "cnLineLabelPerimSpaceF";
%constant NhlNcnLineLabelPerimThicknessF = "cnLineLabelPerimThicknessF";
%constant NhlNcnLineLabelPlacementMode = "cnLineLabelPlacementMode";
%constant NhlNcnLineLabelStrings = "cnLineLabelStrings";
%constant NhlNcnLineLabelsOn = "cnLineLabelsOn";
%constant NhlNcnLineThicknessF = "cnLineThicknessF";
%constant NhlNcnLineThicknesses = "cnLineThicknesses";
%constant NhlNcnLinesOn = "cnLinesOn";
%constant NhlNcnLowLabelAngleF = "cnLowLabelAngleF";
%constant NhlNcnLowLabelBackgroundColor = "cnLowLabelBackgroundColor";
%constant NhlNcnLowLabelConstantSpacingF = "cnLowLabelConstantSpacingF";
%constant NhlNcnLowLabelFont = "cnLowLabelFont";
%constant NhlNcnLowLabelFontAspectF = "cnLowLabelFontAspectF";
%constant NhlNcnLowLabelFontColor = "cnLowLabelFontColor";
%constant NhlNcnLowLabelFontHeightF = "cnLowLabelFontHeightF";
%constant NhlNcnLowLabelFontQuality = "cnLowLabelFontQuality";
%constant NhlNcnLowLabelFontThicknessF = "cnLowLabelFontThicknessF";
%constant NhlNcnLowLabelFormat = "cnLowLabelFormat";
%constant NhlNcnLowLabelFuncCode = "cnLowLabelFuncCode";
%constant NhlNcnLowLabelPerimColor = "cnLowLabelPerimColor";
%constant NhlNcnLowLabelPerimOn = "cnLowLabelPerimOn";
%constant NhlNcnLowLabelPerimSpaceF = "cnLowLabelPerimSpaceF";
%constant NhlNcnLowLabelPerimThicknessF = "cnLowLabelPerimThicknessF";
%constant NhlNcnLowLabelString = "cnLowLabelString";
%constant NhlNcnLowLabelsOn = "cnLowLabelsOn";
%constant NhlNcnLowUseHighLabelRes = "cnLowUseHighLabelRes";
%constant NhlNcnMaxDataValueFormat = "cnMaxDataValueFormat";
%constant NhlNcnMaxLevelCount = "cnMaxLevelCount";
%constant NhlNcnMaxLevelValF = "cnMaxLevelValF";
%constant NhlNcnMaxPointDistanceF = "cnMaxPointDistanceF";
%constant NhlNcnMinLevelValF = "cnMinLevelValF";
%constant NhlNcnMissingValFillColor = "cnMissingValFillColor";
%constant NhlNcnMissingValFillPattern = "cnMissingValFillPattern";
%constant NhlNcnMissingValFillScaleF = "cnMissingValFillScaleF";
%constant NhlNcnMissingValPerimColor = "cnMissingValPerimColor";
%constant NhlNcnMissingValPerimDashPattern = "cnMissingValPerimDashPattern";
%constant NhlNcnMissingValPerimGridBoundOn = "cnMissingValPerimGridBoundOn";
%constant NhlNcnMissingValPerimOn = "cnMissingValPerimOn";
%constant NhlNcnMissingValPerimThicknessF = "cnMissingValPerimThicknessF";
%constant NhlNcnMonoFillColor = "cnMonoFillColor";
%constant NhlNcnMonoFillPattern = "cnMonoFillPattern";
%constant NhlNcnMonoFillScale = "cnMonoFillScale";
%constant NhlNcnMonoLevelFlag = "cnMonoLevelFlag";
%constant NhlNcnMonoLineColor = "cnMonoLineColor";
%constant NhlNcnMonoLineDashPattern = "cnMonoLineDashPattern";
%constant NhlNcnMonoLineLabelFontColor = "cnMonoLineLabelFontColor";
%constant NhlNcnMonoLineThickness = "cnMonoLineThickness";
%constant NhlNcnNoDataLabelOn = "cnNoDataLabelOn";
%constant NhlNcnNoDataLabelString = "cnNoDataLabelString";
%constant NhlNcnOutOfRangePerimColor = "cnOutOfRangePerimColor";
%constant NhlNcnOutOfRangePerimDashPattern = "cnOutOfRangePerimDashPattern";
%constant NhlNcnOutOfRangePerimOn = "cnOutOfRangePerimOn";
%constant NhlNcnOutOfRangePerimThicknessF = "cnOutOfRangePerimThicknessF";
%constant NhlNcnRasterCellSizeF = "cnRasterCellSizeF";
%constant NhlNcnRasterMinCellSizeF = "cnRasterMinCellSizeF";
%constant NhlNcnRasterModeOn = "cnRasterModeOn";
%constant NhlNcnRasterSampleFactorF = "cnRasterSampleFactorF";
%constant NhlNcnRasterSmoothingOn = "cnRasterSmoothingOn";
%constant NhlNcnScalarFieldData = "cnScalarFieldData";
%constant NhlNcnSmoothingDistanceF = "cnSmoothingDistanceF";
%constant NhlNcnSmoothingOn = "cnSmoothingOn";
%constant NhlNcnSmoothingTensionF = "cnSmoothingTensionF";
%constant NhlNctCopyTables = "ctCopyTables";
%constant NhlNctXElementSize = "ctXElementSize";
%constant NhlNctXMaxV = "ctXMaxV";
%constant NhlNctXMinV = "ctXMinV";
%constant NhlNctXMissingV = "ctXMissingV";
%constant NhlNctXTable = "ctXTable";
%constant NhlNctXTableLengths = "ctXTableLengths";
%constant NhlNctXTableType = "ctXTableType";
%constant NhlNctYElementSize = "ctYElementSize";
%constant NhlNctYMaxV = "ctYMaxV";
%constant NhlNctYMinV = "ctYMinV";
%constant NhlNctYMissingV = "ctYMissingV";
%constant NhlNctYTable = "ctYTable";
%constant NhlNctYTableLengths = "ctYTableLengths";
%constant NhlNctYTableType = "ctYTableType";
%constant NhlNdcDelayCompute = "dcDelayCompute";
%constant NhlNerrBuffer = "errBuffer";
%constant NhlNerrFileName = "errFileName";
%constant NhlNerrLevel = "errLevel";
%constant NhlNerrPrint = "errPrint";
%constant NhlNgsClipOn = "gsClipOn";
%constant NhlNgsEdgeColor = "gsEdgeColor";
%constant NhlNgsEdgeDashPattern = "gsEdgeDashPattern";
%constant NhlNgsEdgeDashSegLenF = "gsEdgeDashSegLenF";
%constant NhlNgsEdgeThicknessF = "gsEdgeThicknessF";
%constant NhlNgsEdgesOn = "gsEdgesOn";
%constant NhlNgsFillBackgroundColor = "gsFillBackgroundColor";
%constant NhlNgsFillColor = "gsFillColor";
%constant NhlNgsFillIndex = "gsFillIndex";
%constant NhlNgsFillLineThicknessF = "gsFillLineThicknessF";
%constant NhlNgsFillScaleF = "gsFillScaleF";
%constant NhlNgsFont = "gsFont";
%constant NhlNgsFontAspectF = "gsFontAspectF";
%constant NhlNgsFontColor = "gsFontColor";
%constant NhlNgsFontHeightF = "gsFontHeightF";
%constant NhlNgsFontQuality = "gsFontQuality";
%constant NhlNgsFontThicknessF = "gsFontThicknessF";
%constant NhlNgsLineColor = "gsLineColor";
%constant NhlNgsLineDashPattern = "gsLineDashPattern";
%constant NhlNgsLineDashSegLenF = "gsLineDashSegLenF";
%constant NhlNgsLineLabelConstantSpacingF = "gsLineLabelConstantSpacingF";
%constant NhlNgsLineLabelFont = "gsLineLabelFont";
%constant NhlNgsLineLabelFontAspectF = "gsLineLabelFontAspectF";
%constant NhlNgsLineLabelFontColor = "gsLineLabelFontColor";
%constant NhlNgsLineLabelFontHeightF = "gsLineLabelFontHeightF";
%constant NhlNgsLineLabelFontQuality = "gsLineLabelFontQuality";
%constant NhlNgsLineLabelFontThicknessF = "gsLineLabelFontThicknessF";
%constant NhlNgsLineLabelFuncCode = "gsLineLabelFuncCode";
%constant NhlNgsLineLabelString = "gsLineLabelString";
%constant NhlNgsLineThicknessF = "gsLineThicknessF";
%constant NhlNgsMarkerColor = "gsMarkerColor";
%constant NhlNgsMarkerIndex = "gsMarkerIndex";
%constant NhlNgsMarkerSizeF = "gsMarkerSizeF";
%constant NhlNgsMarkerThicknessF = "gsMarkerThicknessF";
%constant NhlNgsTextAngleF = "gsTextAngleF";
%constant NhlNgsTextConstantSpacingF = "gsTextConstantSpacingF";
%constant NhlNgsTextDirection = "gsTextDirection";
%constant NhlNgsTextFuncCode = "gsTextFuncCode";
%constant NhlNgsTextJustification = "gsTextJustification";
%constant NhlNlbAutoManage = "lbAutoManage";
%constant NhlNlbBottomMarginF = "lbBottomMarginF";
%constant NhlNlbBoxCount = "lbBoxCount";
%constant NhlNlbBoxFractions = "lbBoxFractions";
%constant NhlNlbBoxLineColor = "lbBoxLineColor";
%constant NhlNlbBoxLineDashPattern = "lbBoxLineDashPattern";
%constant NhlNlbBoxLineDashSegLenF = "lbBoxLineDashSegLenF";
%constant NhlNlbBoxLineThicknessF = "lbBoxLineThicknessF";
%constant NhlNlbBoxLinesOn = "lbBoxLinesOn";
%constant NhlNlbBoxMajorExtentF = "lbBoxMajorExtentF";
%constant NhlNlbBoxMinorExtentF = "lbBoxMinorExtentF";
%constant NhlNlbBoxSizing = "lbBoxSizing";
%constant NhlNlbFillBackground = "lbFillBackground";
%constant NhlNlbFillColor = "lbFillColor";
%constant NhlNlbFillColors = "lbFillColors";
%constant NhlNlbFillLineThicknessF = "lbFillLineThicknessF";
%constant NhlNlbFillPattern = "lbFillPattern";
%constant NhlNlbFillPatterns = "lbFillPatterns";
%constant NhlNlbFillScaleF = "lbFillScaleF";
%constant NhlNlbFillScales = "lbFillScales";
%constant NhlNlbJustification = "lbJustification";
%constant NhlNlbLabelAlignment = "lbLabelAlignment";
%constant NhlNlbLabelAngleF = "lbLabelAngleF";
%constant NhlNlbLabelAutoStride = "lbLabelAutoStride";
%constant NhlNlbLabelBarOn = "lbLabelBarOn";
%constant NhlNlbLabelConstantSpacingF = "lbLabelConstantSpacingF";
%constant NhlNlbLabelDirection = "lbLabelDirection";
%constant NhlNlbLabelFont = "lbLabelFont";
%constant NhlNlbLabelFontAspectF = "lbLabelFontAspectF";
%constant NhlNlbLabelFontColor = "lbLabelFontColor";
%constant NhlNlbLabelFontHeightF = "lbLabelFontHeightF";
%constant NhlNlbLabelFontQuality = "lbLabelFontQuality";
%constant NhlNlbLabelFontThicknessF = "lbLabelFontThicknessF";
%constant NhlNlbLabelFuncCode = "lbLabelFuncCode";
%constant NhlNlbLabelJust = "lbLabelJust";
%constant NhlNlbLabelOffsetF = "lbLabelOffsetF";
%constant NhlNlbLabelPosition = "lbLabelPosition";
%constant NhlNlbLabelStride = "lbLabelStride";
%constant NhlNlbLabelStrings = "lbLabelStrings";
%constant NhlNlbLabelsOn = "lbLabelsOn";
%constant NhlNlbLeftMarginF = "lbLeftMarginF";
%constant NhlNlbMaxLabelLenF = "lbMaxLabelLenF";
%constant NhlNlbMinLabelSpacingF = "lbMinLabelSpacingF";
%constant NhlNlbMonoFillColor = "lbMonoFillColor";
%constant NhlNlbMonoFillPattern = "lbMonoFillPattern";
%constant NhlNlbMonoFillScale = "lbMonoFillScale";
%constant NhlNlbOrientation = "lbOrientation";
%constant NhlNlbPerimColor = "lbPerimColor";
%constant NhlNlbPerimDashPattern = "lbPerimDashPattern";
%constant NhlNlbPerimDashSegLenF = "lbPerimDashSegLenF";
%constant NhlNlbPerimFill = "lbPerimFill";
%constant NhlNlbPerimFillColor = "lbPerimFillColor";
%constant NhlNlbPerimOn = "lbPerimOn";
%constant NhlNlbPerimThicknessF = "lbPerimThicknessF";
%constant NhlNlbRightMarginF = "lbRightMarginF";
%constant NhlNlbTitleAngleF = "lbTitleAngleF";
%constant NhlNlbTitleConstantSpacingF = "lbTitleConstantSpacingF";
%constant NhlNlbTitleDirection = "lbTitleDirection";
%constant NhlNlbTitleExtentF = "lbTitleExtentF";
%constant NhlNlbTitleFont = "lbTitleFont";
%constant NhlNlbTitleFontAspectF = "lbTitleFontAspectF";
%constant NhlNlbTitleFontColor = "lbTitleFontColor";
%constant NhlNlbTitleFontHeightF = "lbTitleFontHeightF";
%constant NhlNlbTitleFontQuality = "lbTitleFontQuality";
%constant NhlNlbTitleFontThicknessF = "lbTitleFontThicknessF";
%constant NhlNlbTitleFuncCode = "lbTitleFuncCode";
%constant NhlNlbTitleJust = "lbTitleJust";
%constant NhlNlbTitleOffsetF = "lbTitleOffsetF";
%constant NhlNlbTitleOn = "lbTitleOn";
%constant NhlNlbTitlePosition = "lbTitlePosition";
%constant NhlNlbTitleString = "lbTitleString";
%constant NhlNlbTopMarginF = "lbTopMarginF";
%constant NhlNlgAutoManage = "lgAutoManage";
%constant NhlNlgBottomMarginF = "lgBottomMarginF";
%constant NhlNlgBoxBackground = "lgBoxBackground";
%constant NhlNlgBoxLineColor = "lgBoxLineColor";
%constant NhlNlgBoxLineDashPattern = "lgBoxLineDashPattern";
%constant NhlNlgBoxLineDashSegLenF = "lgBoxLineDashSegLenF";
%constant NhlNlgBoxLineThicknessF = "lgBoxLineThicknessF";
%constant NhlNlgBoxLinesOn = "lgBoxLinesOn";
%constant NhlNlgBoxMajorExtentF = "lgBoxMajorExtentF";
%constant NhlNlgBoxMinorExtentF = "lgBoxMinorExtentF";
%constant NhlNlgDashIndex = "lgDashIndex";
%constant NhlNlgDashIndexes = "lgDashIndexes";
%constant NhlNlgItemCount = "lgItemCount";
%constant NhlNlgItemPlacement = "lgItemPlacement";
%constant NhlNlgItemPositions = "lgItemPositions";
%constant NhlNlgItemType = "lgItemType";
%constant NhlNlgItemTypes = "lgItemTypes";
%constant NhlNlgJustification = "lgJustification";
%constant NhlNlgLabelAlignment = "lgLabelAlignment";
%constant NhlNlgLabelAngleF = "lgLabelAngleF";
%constant NhlNlgLabelAutoStride = "lgLabelAutoStride";
%constant NhlNlgLabelConstantSpacingF = "lgLabelConstantSpacingF";
%constant NhlNlgLabelDirection = "lgLabelDirection";
%constant NhlNlgLabelFont = "lgLabelFont";
%constant NhlNlgLabelFontAspectF = "lgLabelFontAspectF";
%constant NhlNlgLabelFontColor = "lgLabelFontColor";
%constant NhlNlgLabelFontHeightF = "lgLabelFontHeightF";
%constant NhlNlgLabelFontQuality = "lgLabelFontQuality";
%constant NhlNlgLabelFontThicknessF = "lgLabelFontThicknessF";
%constant NhlNlgLabelFuncCode = "lgLabelFuncCode";
%constant NhlNlgLabelJust = "lgLabelJust";
%constant NhlNlgLabelOffsetF = "lgLabelOffsetF";
%constant NhlNlgLabelPosition = "lgLabelPosition";
%constant NhlNlgLabelStride = "lgLabelStride";
%constant NhlNlgLabelStrings = "lgLabelStrings";
%constant NhlNlgLabelsOn = "lgLabelsOn";
%constant NhlNlgLeftMarginF = "lgLeftMarginF";
%constant NhlNlgLegendOn = "lgLegendOn";
%constant NhlNlgLineColor = "lgLineColor";
%constant NhlNlgLineColors = "lgLineColors";
%constant NhlNlgLineDashSegLenF = "lgLineDashSegLenF";
%constant NhlNlgLineDashSegLens = "lgLineDashSegLens";
%constant NhlNlgLineLabelConstantSpacingF = "lgLineLabelConstantSpacingF";
%constant NhlNlgLineLabelFont = "lgLineLabelFont";
%constant NhlNlgLineLabelFontAspectF = "lgLineLabelFontAspectF";
%constant NhlNlgLineLabelFontColor = "lgLineLabelFontColor";
%constant NhlNlgLineLabelFontColors = "lgLineLabelFontColors";
%constant NhlNlgLineLabelFontHeightF = "lgLineLabelFontHeightF";
%constant NhlNlgLineLabelFontHeights = "lgLineLabelFontHeights";
%constant NhlNlgLineLabelFontQuality = "lgLineLabelFontQuality";
%constant NhlNlgLineLabelFontThicknessF = "lgLineLabelFontThicknessF";
%constant NhlNlgLineLabelFuncCode = "lgLineLabelFuncCode";
%constant NhlNlgLineLabelStrings = "lgLineLabelStrings";
%constant NhlNlgLineLabelsOn = "lgLineLabelsOn";
%constant NhlNlgLineThicknessF = "lgLineThicknessF";
%constant NhlNlgLineThicknesses = "lgLineThicknesses";
%constant NhlNlgMarkerColor = "lgMarkerColor";
%constant NhlNlgMarkerColors = "lgMarkerColors";
%constant NhlNlgMarkerIndex = "lgMarkerIndex";
%constant NhlNlgMarkerIndexes = "lgMarkerIndexes";
%constant NhlNlgMarkerSizeF = "lgMarkerSizeF";
%constant NhlNlgMarkerSizes = "lgMarkerSizes";
%constant NhlNlgMarkerThicknessF = "lgMarkerThicknessF";
%constant NhlNlgMarkerThicknesses = "lgMarkerThicknesses";
%constant NhlNlgMonoDashIndex = "lgMonoDashIndex";
%constant NhlNlgMonoItemType = "lgMonoItemType";
%constant NhlNlgMonoLineColor = "lgMonoLineColor";
%constant NhlNlgMonoLineDashSegLen = "lgMonoLineDashSegLen";
%constant NhlNlgMonoLineLabelFontColor = "lgMonoLineLabelFontColor";
%constant NhlNlgMonoLineLabelFontHeight = "lgMonoLineLabelFontHeight";
%constant NhlNlgMonoLineThickness = "lgMonoLineThickness";
%constant NhlNlgMonoMarkerColor = "lgMonoMarkerColor";
%constant NhlNlgMonoMarkerIndex = "lgMonoMarkerIndex";
%constant NhlNlgMonoMarkerSize = "lgMonoMarkerSize";
%constant NhlNlgMonoMarkerThickness = "lgMonoMarkerThickness";
%constant NhlNlgOrientation = "lgOrientation";
%constant NhlNlgPerimColor = "lgPerimColor";
%constant NhlNlgPerimDashPattern = "lgPerimDashPattern";
%constant NhlNlgPerimDashSegLenF = "lgPerimDashSegLenF";
%constant NhlNlgPerimFill = "lgPerimFill";
%constant NhlNlgPerimFillColor = "lgPerimFillColor";
%constant NhlNlgPerimOn = "lgPerimOn";
%constant NhlNlgPerimThicknessF = "lgPerimThicknessF";
%constant NhlNlgRightMarginF = "lgRightMarginF";
%constant NhlNlgTitleAngleF = "lgTitleAngleF";
%constant NhlNlgTitleConstantSpacingF = "lgTitleConstantSpacingF";
%constant NhlNlgTitleDirection = "lgTitleDirection";
%constant NhlNlgTitleExtentF = "lgTitleExtentF";
%constant NhlNlgTitleFont = "lgTitleFont";
%constant NhlNlgTitleFontAspectF = "lgTitleFontAspectF";
%constant NhlNlgTitleFontColor = "lgTitleFontColor";
%constant NhlNlgTitleFontHeightF = "lgTitleFontHeightF";
%constant NhlNlgTitleFontQuality = "lgTitleFontQuality";
%constant NhlNlgTitleFontThicknessF = "lgTitleFontThicknessF";
%constant NhlNlgTitleFuncCode = "lgTitleFuncCode";
%constant NhlNlgTitleJust = "lgTitleJust";
%constant NhlNlgTitleOffsetF = "lgTitleOffsetF";
%constant NhlNlgTitleOn = "lgTitleOn";
%constant NhlNlgTitlePosition = "lgTitlePosition";
%constant NhlNlgTitleString = "lgTitleString";
%constant NhlNlgTopMarginF = "lgTopMarginF";
%constant NhlNmpAreaGroupCount = "mpAreaGroupCount";
%constant NhlNmpAreaMaskingOn = "mpAreaMaskingOn";
%constant NhlNmpAreaNames = "mpAreaNames";
%constant NhlNmpAreaTypes = "mpAreaTypes";
%constant NhlNmpDataBaseVersion = "mpDataBaseVersion";
%constant NhlNmpDataResolution = "mpDataResolution";
%constant NhlNmpDataSetName = "mpDataSetName";
%constant NhlNmpDefaultFillColor = "mpDefaultFillColor";
%constant NhlNmpDefaultFillPattern = "mpDefaultFillPattern";
%constant NhlNmpDefaultFillScaleF = "mpDefaultFillScaleF";
%constant NhlNmpDynamicAreaGroups = "mpDynamicAreaGroups";
%constant NhlNmpFillAreaSpecifiers = "mpFillAreaSpecifiers";
%constant NhlNmpFillBoundarySets = "mpFillBoundarySets";
%constant NhlNmpFillColor = "mpFillColor";
%constant NhlNmpFillColors = "mpFillColors";
%constant NhlNmpFillDrawOrder = "mpFillDrawOrder";
%constant NhlNmpFillOn = "mpFillOn";
%constant NhlNmpFillPattern = "mpFillPattern";
%constant NhlNmpFillPatternBackground = "mpFillPatternBackground";
%constant NhlNmpFillPatterns = "mpFillPatterns";
%constant NhlNmpFillScaleF = "mpFillScaleF";
%constant NhlNmpFillScales = "mpFillScales";
%constant NhlNmpFixedAreaGroups = "mpFixedAreaGroups";
%constant NhlNmpGeophysicalLineColor = "mpGeophysicalLineColor";
%constant NhlNmpGeophysicalLineDashPattern = "mpGeophysicalLineDashPattern";
%constant NhlNmpGeophysicalLineDashSegLenF = "mpGeophysicalLineDashSegLenF";
%constant NhlNmpGeophysicalLineThicknessF = "mpGeophysicalLineThicknessF";
%constant NhlNmpGridAndLimbDrawOrder = "mpGridAndLimbDrawOrder";
%constant NhlNmpGridAndLimbOn = "mpGridAndLimbOn";
%constant NhlNmpGridLatSpacingF = "mpGridLatSpacingF";
%constant NhlNmpGridLineColor = "mpGridLineColor";
%constant NhlNmpGridLineDashPattern = "mpGridLineDashPattern";
%constant NhlNmpGridLineDashSegLenF = "mpGridLineDashSegLenF";
%constant NhlNmpGridLineThicknessF = "mpGridLineThicknessF";
%constant NhlNmpGridLonSpacingF = "mpGridLonSpacingF";
%constant NhlNmpGridMaskMode = "mpGridMaskMode";
%constant NhlNmpGridMaxLatF = "mpGridMaxLatF";
%constant NhlNmpGridPolarLonSpacingF = "mpGridPolarLonSpacingF";
%constant NhlNmpGridSpacingF = "mpGridSpacingF";
%constant NhlNmpInlandWaterFillColor = "mpInlandWaterFillColor";
%constant NhlNmpInlandWaterFillPattern = "mpInlandWaterFillPattern";
%constant NhlNmpInlandWaterFillScaleF = "mpInlandWaterFillScaleF";
%constant NhlNmpLabelDrawOrder = "mpLabelDrawOrder";
%constant NhlNmpLabelFontColor = "mpLabelFontColor";
%constant NhlNmpLabelFontHeightF = "mpLabelFontHeightF";
%constant NhlNmpLabelsOn = "mpLabelsOn";
%constant NhlNmpLandFillColor = "mpLandFillColor";
%constant NhlNmpLandFillPattern = "mpLandFillPattern";
%constant NhlNmpLandFillScaleF = "mpLandFillScaleF";
%constant NhlNmpLimbLineColor = "mpLimbLineColor";
%constant NhlNmpLimbLineDashPattern = "mpLimbLineDashPattern";
%constant NhlNmpLimbLineDashSegLenF = "mpLimbLineDashSegLenF";
%constant NhlNmpLimbLineThicknessF = "mpLimbLineThicknessF";
%constant NhlNmpMaskAreaSpecifiers = "mpMaskAreaSpecifiers";
%constant NhlNmpMonoFillColor = "mpMonoFillColor";
%constant NhlNmpMonoFillPattern = "mpMonoFillPattern";
%constant NhlNmpMonoFillScale = "mpMonoFillScale";
%constant NhlNmpNationalLineColor = "mpNationalLineColor";
%constant NhlNmpNationalLineDashPattern = "mpNationalLineDashPattern";
%constant NhlNmpNationalLineDashSegLenF = "mpNationalLineDashSegLenF";
%constant NhlNmpNationalLineThicknessF = "mpNationalLineThicknessF";
%constant NhlNmpOceanFillColor = "mpOceanFillColor";
%constant NhlNmpOceanFillPattern = "mpOceanFillPattern";
%constant NhlNmpOceanFillScaleF = "mpOceanFillScaleF";
%constant NhlNmpOutlineBoundarySets = "mpOutlineBoundarySets";
%constant NhlNmpOutlineDrawOrder = "mpOutlineDrawOrder";
%constant NhlNmpOutlineOn = "mpOutlineOn";
%constant NhlNmpOutlineSpecifiers = "mpOutlineSpecifiers";
%constant NhlNmpPerimDrawOrder = "mpPerimDrawOrder";
%constant NhlNmpPerimLineColor = "mpPerimLineColor";
%constant NhlNmpPerimLineDashPattern = "mpPerimLineDashPattern";
%constant NhlNmpPerimLineDashSegLenF = "mpPerimLineDashSegLenF";
%constant NhlNmpPerimLineThicknessF = "mpPerimLineThicknessF";
%constant NhlNmpPerimOn = "mpPerimOn";
%constant NhlNmpShapeMode = "mpShapeMode";
%constant NhlNmpSpecifiedFillColors = "mpSpecifiedFillColors";
%constant NhlNmpSpecifiedFillDirectIndexing = "mpSpecifiedFillDirectIndexing";
%constant NhlNmpSpecifiedFillPatterns = "mpSpecifiedFillPatterns";
%constant NhlNmpSpecifiedFillPriority = "mpSpecifiedFillPriority";
%constant NhlNmpSpecifiedFillScales = "mpSpecifiedFillScales";
%constant NhlNmpUSStateLineColor = "mpUSStateLineColor";
%constant NhlNmpUSStateLineDashPattern = "mpUSStateLineDashPattern";
%constant NhlNmpUSStateLineDashSegLenF = "mpUSStateLineDashSegLenF";
%constant NhlNmpUSStateLineThicknessF = "mpUSStateLineThicknessF";
%constant NhlNmpBottomAngleF = "mpBottomAngleF";
%constant NhlNmpBottomMapPosF = "mpBottomMapPosF";
%constant NhlNmpBottomNDCF = "mpBottomNDCF";
%constant NhlNmpBottomNPCF = "mpBottomNPCF";
%constant NhlNmpBottomPointLatF = "mpBottomPointLatF";
%constant NhlNmpBottomPointLonF = "mpBottomPointLonF";
%constant NhlNmpBottomWindowF = "mpBottomWindowF";
%constant NhlNmpCenterLatF = "mpCenterLatF";
%constant NhlNmpCenterLonF = "mpCenterLonF";
%constant NhlNmpCenterRotF = "mpCenterRotF";
%constant NhlNmpEllipticalBoundary = "mpEllipticalBoundary";
%constant NhlNmpGreatCircleLinesOn = "mpGreatCircleLinesOn";
%constant NhlNmpLambertMeridianF = "mpLambertMeridianF";
%constant NhlNmpLambertParallel1F = "mpLambertParallel1F";
%constant NhlNmpLambertParallel2F = "mpLambertParallel2F";
%constant NhlNmpLeftAngleF = "mpLeftAngleF";
%constant NhlNmpLeftCornerLatF = "mpLeftCornerLatF";
%constant NhlNmpLeftCornerLonF = "mpLeftCornerLonF";
%constant NhlNmpLeftMapPosF = "mpLeftMapPosF";
%constant NhlNmpLeftNDCF = "mpLeftNDCF";
%constant NhlNmpLeftNPCF = "mpLeftNPCF";
%constant NhlNmpLeftPointLatF = "mpLeftPointLatF";
%constant NhlNmpLeftPointLonF = "mpLeftPointLonF";
%constant NhlNmpLeftWindowF = "mpLeftWindowF";
%constant NhlNmpLimitMode = "mpLimitMode";
%constant NhlNmpMaxLatF = "mpMaxLatF";
%constant NhlNmpMaxLonF = "mpMaxLonF";
%constant NhlNmpMinLatF = "mpMinLatF";
%constant NhlNmpMinLonF = "mpMinLonF";
%constant NhlNmpProjection = "mpProjection";
%constant NhlNmpRelativeCenterLat = "mpRelativeCenterLat";
%constant NhlNmpRelativeCenterLon = "mpRelativeCenterLon";
%constant NhlNmpRightAngleF = "mpRightAngleF";
%constant NhlNmpRightCornerLatF = "mpRightCornerLatF";
%constant NhlNmpRightCornerLonF = "mpRightCornerLonF";
%constant NhlNmpRightMapPosF = "mpRightMapPosF";
%constant NhlNmpRightNDCF = "mpRightNDCF";
%constant NhlNmpRightNPCF = "mpRightNPCF";
%constant NhlNmpRightPointLatF = "mpRightPointLatF";
%constant NhlNmpRightPointLonF = "mpRightPointLonF";
%constant NhlNmpRightWindowF = "mpRightWindowF";
%constant NhlNmpSatelliteAngle1F = "mpSatelliteAngle1F";
%constant NhlNmpSatelliteAngle2F = "mpSatelliteAngle2F";
%constant NhlNmpSatelliteDistF = "mpSatelliteDistF";
%constant NhlNmpTopAngleF = "mpTopAngleF";
%constant NhlNmpTopMapPosF = "mpTopMapPosF";
%constant NhlNmpTopNDCF = "mpTopNDCF";
%constant NhlNmpTopNPCF = "mpTopNPCF";
%constant NhlNmpTopPointLatF = "mpTopPointLatF";
%constant NhlNmpTopPointLonF = "mpTopPointLonF";
%constant NhlNmpTopWindowF = "mpTopWindowF";
%constant NhlNpmAnnoManagers = "pmAnnoManagers";
%constant NhlNpmAnnoViews = "pmAnnoViews";
%constant NhlNpmLabelBarDisplayMode = "pmLabelBarDisplayMode";
%constant NhlNpmLabelBarHeightF = "pmLabelBarHeightF";
%constant NhlNpmLabelBarKeepAspect = "pmLabelBarKeepAspect";
%constant NhlNpmLabelBarOrthogonalPosF = "pmLabelBarOrthogonalPosF";
%constant NhlNpmLabelBarParallelPosF = "pmLabelBarParallelPosF";
%constant NhlNpmLabelBarSide = "pmLabelBarSide";
%constant NhlNpmLabelBarWidthF = "pmLabelBarWidthF";
%constant NhlNpmLabelBarZone = "pmLabelBarZone";
%constant NhlNpmLegendDisplayMode = "pmLegendDisplayMode";
%constant NhlNpmLegendHeightF = "pmLegendHeightF";
%constant NhlNpmLegendKeepAspect = "pmLegendKeepAspect";
%constant NhlNpmLegendOrthogonalPosF = "pmLegendOrthogonalPosF";
%constant NhlNpmLegendParallelPosF = "pmLegendParallelPosF";
%constant NhlNpmLegendSide = "pmLegendSide";
%constant NhlNpmLegendWidthF = "pmLegendWidthF";
%constant NhlNpmLegendZone = "pmLegendZone";
%constant NhlNpmOverlaySequenceIds = "pmOverlaySequenceIds";
%constant NhlNpmTickMarkDisplayMode = "pmTickMarkDisplayMode";
%constant NhlNpmTickMarkZone = "pmTickMarkZone";
%constant NhlNpmTitleDisplayMode = "pmTitleDisplayMode";
%constant NhlNpmTitleZone = "pmTitleZone";
%constant NhlNprGraphicStyle = "prGraphicStyle";
%constant NhlNprPolyType = "prPolyType";
%constant NhlNprXArray = "prXArray";
%constant NhlNprYArray = "prYArray";
%constant NhlNsfCopyData = "sfCopyData";
%constant NhlNsfDataArray = "sfDataArray";
%constant NhlNsfDataMaxV = "sfDataMaxV";
%constant NhlNsfDataMinV = "sfDataMinV";
%constant NhlNsfExchangeDimensions = "sfExchangeDimensions";
%constant NhlNsfMissingValueV = "sfMissingValueV";
%constant NhlNsfXArray = "sfXArray";
%constant NhlNsfXCActualEndF = "sfXCActualEndF";
%constant NhlNsfXCActualStartF = "sfXCActualStartF";
%constant NhlNsfXCEndIndex = "sfXCEndIndex";
%constant NhlNsfXCEndSubsetV = "sfXCEndSubsetV";
%constant NhlNsfXCEndV = "sfXCEndV";
%constant NhlNsfXCStartIndex = "sfXCStartIndex";
%constant NhlNsfXCStartSubsetV = "sfXCStartSubsetV";
%constant NhlNsfXCStartV = "sfXCStartV";
%constant NhlNsfXCStride = "sfXCStride";
%constant NhlNsfYArray = "sfYArray";
%constant NhlNsfYCActualEndF = "sfYCActualEndF";
%constant NhlNsfYCActualStartF = "sfYCActualStartF";
%constant NhlNsfYCEndIndex = "sfYCEndIndex";
%constant NhlNsfYCEndSubsetV = "sfYCEndSubsetV";
%constant NhlNsfYCEndV = "sfYCEndV";
%constant NhlNsfYCStartIndex = "sfYCStartIndex";
%constant NhlNsfYCStartSubsetV = "sfYCStartSubsetV";
%constant NhlNsfYCStartV = "sfYCStartV";
%constant NhlNsfYCStride = "sfYCStride";
%constant NhlNstArrowLengthF = "stArrowLengthF";
%constant NhlNstArrowStride = "stArrowStride";
%constant NhlNstCrossoverCheckCount = "stCrossoverCheckCount";
%constant NhlNstExplicitLabelBarLabelsOn = "stExplicitLabelBarLabelsOn";
%constant NhlNstLabelBarEndLabelsOn = "stLabelBarEndLabelsOn";
%constant NhlNstLabelFormat = "stLabelFormat";
%constant NhlNstLengthCheckCount = "stLengthCheckCount";
%constant NhlNstLevelColors = "stLevelColors";
%constant NhlNstLevelCount = "stLevelCount";
%constant NhlNstLevelSelectionMode = "stLevelSelectionMode";
%constant NhlNstLevelSpacingF = "stLevelSpacingF";
%constant NhlNstLevels = "stLevels";
%constant NhlNstLineColor = "stLineColor";
%constant NhlNstLineStartStride = "stLineStartStride";
%constant NhlNstLineThicknessF = "stLineThicknessF";
%constant NhlNstMapDirection = "stMapDirection";
%constant NhlNstMaxLevelCount = "stMaxLevelCount";
%constant NhlNstMaxLevelValF = "stMaxLevelValF";
%constant NhlNstMinArrowSpacingF = "stMinArrowSpacingF";
%constant NhlNstMinDistanceF = "stMinDistanceF";
%constant NhlNstMinLevelValF = "stMinLevelValF";
%constant NhlNstMinLineSpacingF = "stMinLineSpacingF";
%constant NhlNstMinStepFactorF = "stMinStepFactorF";
%constant NhlNstMonoLineColor = "stMonoLineColor";
%constant NhlNstNoDataLabelOn = "stNoDataLabelOn";
%constant NhlNstNoDataLabelString = "stNoDataLabelString";
%constant NhlNstScalarFieldData = "stScalarFieldData";
%constant NhlNstScalarMissingValColor = "stScalarMissingValColor";
%constant NhlNstStepSizeF = "stStepSizeF";
%constant NhlNstStreamlineDrawOrder = "stStreamlineDrawOrder";
%constant NhlNstUseScalarArray = "stUseScalarArray";
%constant NhlNstVectorFieldData = "stVectorFieldData";

%constant NhlNvfDataArray = "vfDataArray";
%constant NhlNvfUDataArray = "vfUDataArray";
%constant NhlNvfVDataArray = "vfVDataArray";
%constant NhlNvfXArray = "vfXArray";
%constant NhlNvfYArray = "vfYArray";
%constant NhlNvfGridType = "vfGridType";
%constant NhlNvfPolarData = "vfPolarData";
%constant NhlNvfSubsetByIndex = "vfSubsetByIndex";
%constant NhlNvfCopyData = "vfCopyData";
%constant NhlNvfExchangeDimensions = "vfExchangeDimensions";
%constant NhlNvfExchangeUVData = "vfExchangeUVData";

%constant NhlNvfSingleMissingValue = "vfSingleMissingValue";
%constant NhlNvfMissingUValueV = "vfMissingUValueV";
%constant NhlNvfMissingVValueV = "vfMissingVValueV";
%constant NhlNvfMagMinV = "vfMagMinV";
%constant NhlNvfMagMaxV = "vfMagMaxV";
%constant NhlNvfUMinV = "vfUMinV";
%constant NhlNvfUMaxV = "vfUMaxV";
%constant NhlNvfVMinV = "vfVMinV";
%constant NhlNvfVMaxV = "vfVMaxV";
%constant NhlNvfXCStartV = "vfXCStartV";
%constant NhlNvfXCEndV = "vfXCEndV";
%constant NhlNvfYCStartV = "vfYCStartV";
%constant NhlNvfYCEndV = "vfYCEndV";
%constant NhlNvfXCStartSubsetV = "vfXCStartSubsetV";
%constant NhlNvfXCEndSubsetV = "vfXCEndSubsetV";
%constant NhlNvfYCStartSubset = "vfYCStartSubsetV";
%constant NhlNvfYCEndSubsetV = "vfYCEndSubsetV";
%constant NhlNvfXCStartIndex = "vfXCStartIndex";
%constant NhlNvfXCEndIndex = "vfXCEndIndex";
%constant NhlNvfYCStartIndex = "vfYCStartIndex";
%constant NhlNvfYCEndIndex = "vfYCEndIndex";
%constant NhlNvfXCStride = "vfXCStride";
%constant NhlNvfYCStride = "vfYCStride";
%constant NhlNvfXCActualStartF = "vfXCActualStartF";
%constant NhlNvfXCActualEndF = "vfXCActualEndF";
%constant NhlNvfXCElementCount = "vfXCElementCount";
%constant NhlNvfYCActualStartF = "vfYCActualStartF";
%constant NhlNvfYCActualEndF = "vfYCActualEndF";
%constant NhlNvfYCElementCount = "vfYCElementCount";


%constant NhlCvfDataArray = "VfDataArray";
%constant NhlCvfUDataArray = "VfUDataArray";
%constant NhlCvfVDataArray = "VfVDataArray";
%constant NhlCvfXArray = "VfXArray";
%constant NhlCvfYArray = "VfYArray";
%constant NhlCvfGridType = "VfGridType";
%constant NhlCvfPolarData = "VfPolarData";
%constant NhlCvfSubsetByIndex = "VfSubsetByIndex";
%constant NhlCvfCopyData = "VfCopyData";
%constant NhlCvfExchangeDimensions = "VfExchangeDimensions";
%constant NhlCvfExchangeUVData = "VfExchangeUVData";
 
%constant NhlCvfSingleMissingValue = "VfSingleMissingValue";
%constant NhlCvfMissingUValueV = "VfMissingUValueV";
%constant NhlCvfMissingVValueV = "VfMissingVValueV";
%constant NhlCvfMagMinV = "VfMagMinV";
%constant NhlCvfMagMaxV = "VfMagMaxV";
%constant NhlCvfUMinV = "VfUMinV";
%constant NhlCvfUMaxV = "VfUMaxV";
%constant NhlCvfVMinV = "VfVMinV";
%constant NhlCvfVMaxV = "VfVMaxV";
%constant NhlCvfXCStartV = "VfXCStartV";
%constant NhlCvfXCEndV = "VfXCEndV";
%constant NhlCvfYCStartV = "VfYCStartV";
%constant NhlCvfYCEndV = "VfYCEndV";
%constant NhlCvfXCStartSubsetV = "VfXCStartSubsetV";
%constant NhlCvfXCEndSubsetV = "VfXCEndSubsetV";
%constant NhlCvfYCStartSubsetV = "VfYCStartSubsetV";
%constant NhlCvfYCEndSubsetV = "VfYCEndSubsetV";
%constant NhlCvfXCStartIndex = "VfXCStartIndex";
%constant NhlCvfXCEndIndex = "VfXCEndIndex";
%constant NhlCvfYCStartIndex = "VfYCStartIndex";
%constant NhlCvfYCEndIndex = "VfYCEndIndex";
%constant NhlCvfXCStride = "VfXCStride";
%constant NhlCvfYCStride = "VfYCStride";
%constant NhlCvfXCActualStartF = "VfXCActualStartF";
%constant NhlCvfXCActualEndF = "VfXCActualEndF";
%constant NhlCvfXCElementCount = "VfXCElementCount";
%constant NhlCvfYCActualStartF = "VfYCActualStartF";
%constant NhlCvfYCActualEndF = "VfYCActualEndF";
%constant NhlCvfYCElementCount = "VfYCElementCount";

%constant NhlNstZeroFLabelAngleF = "stZeroFLabelAngleF";
%constant NhlNstZeroFLabelBackgroundColor = "stZeroFLabelBackgroundColor";
%constant NhlNstZeroFLabelConstantSpacingF = "stZeroFLabelConstantSpacingF";
%constant NhlNstZeroFLabelFont = "stZeroFLabelFont";
%constant NhlNstZeroFLabelFontAspectF = "stZeroFLabelFontAspectF";
%constant NhlNstZeroFLabelFontColor = "stZeroFLabelFontColor";
%constant NhlNstZeroFLabelFontHeightF = "stZeroFLabelFontHeightF";
%constant NhlNstZeroFLabelFontQuality = "stZeroFLabelFontQuality";
%constant NhlNstZeroFLabelFontThicknessF = "stZeroFLabelFontThicknessF";
%constant NhlNstZeroFLabelFuncCode = "stZeroFLabelFuncCode";
%constant NhlNstZeroFLabelJust = "stZeroFLabelJust";
%constant NhlNstZeroFLabelOn = "stZeroFLabelOn";
%constant NhlNstZeroFLabelOrthogonalPosF = "stZeroFLabelOrthogonalPosF";
%constant NhlNstZeroFLabelParallelPosF = "stZeroFLabelParallelPosF";
%constant NhlNstZeroFLabelPerimColor = "stZeroFLabelPerimColor";
%constant NhlNstZeroFLabelPerimOn = "stZeroFLabelPerimOn";
%constant NhlNstZeroFLabelPerimSpaceF = "stZeroFLabelPerimSpaceF";
%constant NhlNstZeroFLabelPerimThicknessF = "stZeroFLabelPerimThicknessF";
%constant NhlNstZeroFLabelSide = "stZeroFLabelSide";
%constant NhlNstZeroFLabelString = "stZeroFLabelString";
%constant NhlNstZeroFLabelTextDirection = "stZeroFLabelTextDirection";
%constant NhlNstZeroFLabelZone = "stZeroFLabelZone";
%constant NhlNtfDoNDCOverlay = "tfDoNDCOverlay";
%constant NhlNtfPlotManagerOn = "tfPlotManagerOn";
%constant NhlNtfPolyDrawList = "tfPolyDrawList";
%constant NhlNtfPolyDrawOrder = "tfPolyDrawOrder";
%constant NhlNtiDeltaF = "tiDeltaF";
%constant NhlNtiMainAngleF = "tiMainAngleF";
%constant NhlNtiMainConstantSpacingF = "tiMainConstantSpacingF";
%constant NhlNtiMainDirection = "tiMainDirection";
%constant NhlNtiMainFont = "tiMainFont";
%constant NhlNtiMainFontAspectF = "tiMainFontAspectF";
%constant NhlNtiMainFontColor = "tiMainFontColor";
%constant NhlNtiMainFontHeightF = "tiMainFontHeightF";
%constant NhlNtiMainFontQuality = "tiMainFontQuality";
%constant NhlNtiMainFontThicknessF = "tiMainFontThicknessF";
%constant NhlNtiMainFuncCode = "tiMainFuncCode";
%constant NhlNtiMainJust = "tiMainJust";
%constant NhlNtiMainOffsetXF = "tiMainOffsetXF";
%constant NhlNtiMainOffsetYF = "tiMainOffsetYF";
%constant NhlNtiMainOn = "tiMainOn";
%constant NhlNtiMainPosition = "tiMainPosition";
%constant NhlNtiMainSide = "tiMainSide";
%constant NhlNtiMainString = "tiMainString";
%constant NhlNtiUseMainAttributes = "tiUseMainAttributes";
%constant NhlNtiXAxisAngleF = "tiXAxisAngleF";
%constant NhlNtiXAxisConstantSpacingF = "tiXAxisConstantSpacingF";
%constant NhlNtiXAxisDirection = "tiXAxisDirection";
%constant NhlNtiXAxisFont = "tiXAxisFont";
%constant NhlNtiXAxisFontAspectF = "tiXAxisFontAspectF";
%constant NhlNtiXAxisFontColor = "tiXAxisFontColor";
%constant NhlNtiXAxisFontHeightF = "tiXAxisFontHeightF";
%constant NhlNtiXAxisFontQuality = "tiXAxisFontQuality";
%constant NhlNtiXAxisFontThicknessF = "tiXAxisFontThicknessF";
%constant NhlNtiXAxisFuncCode = "tiXAxisFuncCode";
%constant NhlNtiXAxisJust = "tiXAxisJust";
%constant NhlNtiXAxisOffsetXF = "tiXAxisOffsetXF";
%constant NhlNtiXAxisOffsetYF = "tiXAxisOffsetYF";
%constant NhlNtiXAxisOn = "tiXAxisOn";
%constant NhlNtiXAxisPosition = "tiXAxisPosition";
%constant NhlNtiXAxisSide = "tiXAxisSide";
%constant NhlNtiXAxisString = "tiXAxisString";
%constant NhlNtiYAxisAngleF = "tiYAxisAngleF";
%constant NhlNtiYAxisConstantSpacingF = "tiYAxisConstantSpacingF";
%constant NhlNtiYAxisDirection = "tiYAxisDirection";
%constant NhlNtiYAxisFont = "tiYAxisFont";
%constant NhlNtiYAxisFontAspectF = "tiYAxisFontAspectF";
%constant NhlNtiYAxisFontColor = "tiYAxisFontColor";
%constant NhlNtiYAxisFontHeightF = "tiYAxisFontHeightF";
%constant NhlNtiYAxisFontQuality = "tiYAxisFontQuality";
%constant NhlNtiYAxisFontThicknessF = "tiYAxisFontThicknessF";
%constant NhlNtiYAxisFuncCode = "tiYAxisFuncCode";
%constant NhlNtiYAxisJust = "tiYAxisJust";
%constant NhlNtiYAxisOffsetXF = "tiYAxisOffsetXF";
%constant NhlNtiYAxisOffsetYF = "tiYAxisOffsetYF";
%constant NhlNtiYAxisOn = "tiYAxisOn";
%constant NhlNtiYAxisPosition = "tiYAxisPosition";
%constant NhlNtiYAxisSide = "tiYAxisSide";
%constant NhlNtiYAxisString = "tiYAxisString";
%constant NhlNtmBorderLineColor = "tmBorderLineColor";
%constant NhlNtmBorderThicknessF = "tmBorderThicknessF";
%constant NhlNtmEqualizeXYSizes = "tmEqualizeXYSizes";
%constant NhlNtmLabelAutoStride = "tmLabelAutoStride";
%constant NhlNtmSciNoteCutoff = "tmSciNoteCutoff";
%constant NhlNtmXBAutoPrecision = "tmXBAutoPrecision";
%constant NhlNtmXBBorderOn = "tmXBBorderOn";
%constant NhlNtmXBDataLeftF = "tmXBDataLeftF";
%constant NhlNtmXBDataRightF = "tmXBDataRightF";
%constant NhlNtmXBFormat = "tmXBFormat";
%constant NhlNtmXBIrrTensionF = "tmXBIrrTensionF";
%constant NhlNtmXBIrregularPoints = "tmXBIrregularPoints";
%constant NhlNtmXBLabelAngleF = "tmXBLabelAngleF";
%constant NhlNtmXBLabelConstantSpacingF = "tmXBLabelConstantSpacingF";
%constant NhlNtmXBLabelDeltaF = "tmXBLabelDeltaF";
%constant NhlNtmXBLabelDirection = "tmXBLabelDirection";
%constant NhlNtmXBLabelFont = "tmXBLabelFont";
%constant NhlNtmXBLabelFontAspectF = "tmXBLabelFontAspectF";
%constant NhlNtmXBLabelFontColor = "tmXBLabelFontColor";
%constant NhlNtmXBLabelFontHeightF = "tmXBLabelFontHeightF";
%constant NhlNtmXBLabelFontQuality = "tmXBLabelFontQuality";
%constant NhlNtmXBLabelFontThicknessF = "tmXBLabelFontThicknessF";
%constant NhlNtmXBLabelFuncCode = "tmXBLabelFuncCode";
%constant NhlNtmXBLabelJust = "tmXBLabelJust";
%constant NhlNtmXBLabelStride = "tmXBLabelStride";
%constant NhlNtmXBLabels = "tmXBLabels";
%constant NhlNtmXBLabelsOn = "tmXBLabelsOn";
%constant NhlNtmXBMajorLengthF = "tmXBMajorLengthF";
%constant NhlNtmXBMajorLineColor = "tmXBMajorLineColor";
%constant NhlNtmXBMajorOutwardLengthF = "tmXBMajorOutwardLengthF";
%constant NhlNtmXBMajorThicknessF = "tmXBMajorThicknessF";
%constant NhlNtmXBMaxLabelLenF = "tmXBMaxLabelLenF";
%constant NhlNtmXBMaxTicks = "tmXBMaxTicks";
%constant NhlNtmXBMinLabelSpacingF = "tmXBMinLabelSpacingF";
%constant NhlNtmXBMinorLengthF = "tmXBMinorLengthF";
%constant NhlNtmXBMinorLineColor = "tmXBMinorLineColor";
%constant NhlNtmXBMinorOn = "tmXBMinorOn";
%constant NhlNtmXBMinorOutwardLengthF = "tmXBMinorOutwardLengthF";
%constant NhlNtmXBMinorPerMajor = "tmXBMinorPerMajor";
%constant NhlNtmXBMinorThicknessF = "tmXBMinorThicknessF";
%constant NhlNtmXBMinorValues = "tmXBMinorValues";
%constant NhlNtmXBMode = "tmXBMode";
%constant NhlNtmXBOn = "tmXBOn";
%constant NhlNtmXBPrecision = "tmXBPrecision";
%constant NhlNtmXBStyle = "tmXBStyle";
%constant NhlNtmXBTickEndF = "tmXBTickEndF";
%constant NhlNtmXBTickSpacingF = "tmXBTickSpacingF";
%constant NhlNtmXBTickStartF = "tmXBTickStartF";
%constant NhlNtmXBValues = "tmXBValues";
%constant NhlNtmXMajorGrid = "tmXMajorGrid";
%constant NhlNtmXMajorGridLineColor = "tmXMajorGridLineColor";
%constant NhlNtmXMajorGridLineDashPattern = "tmXMajorGridLineDashPattern";
%constant NhlNtmXMajorGridThicknessF = "tmXMajorGridThicknessF";
%constant NhlNtmXMinorGrid = "tmXMinorGrid";
%constant NhlNtmXMinorGridLineColor = "tmXMinorGridLineColor";
%constant NhlNtmXMinorGridLineDashPattern = "tmXMinorGridLineDashPattern";
%constant NhlNtmXMinorGridThicknessF = "tmXMinorGridThicknessF";
%constant NhlNtmXTAutoPrecision = "tmXTAutoPrecision";
%constant NhlNtmXTBorderOn = "tmXTBorderOn";
%constant NhlNtmXTDataLeftF = "tmXTDataLeftF";
%constant NhlNtmXTDataRightF = "tmXTDataRightF";
%constant NhlNtmXTFormat = "tmXTFormat";
%constant NhlNtmXTIrrTensionF = "tmXTIrrTensionF";
%constant NhlNtmXTIrregularPoints = "tmXTIrregularPoints";
%constant NhlNtmXTLabelAngleF = "tmXTLabelAngleF";
%constant NhlNtmXTLabelConstantSpacingF = "tmXTLabelConstantSpacingF";
%constant NhlNtmXTLabelDeltaF = "tmXTLabelDeltaF";
%constant NhlNtmXTLabelDirection = "tmXTLabelDirection";
%constant NhlNtmXTLabelFont = "tmXTLabelFont";
%constant NhlNtmXTLabelFontAspectF = "tmXTLabelFontAspectF";
%constant NhlNtmXTLabelFontColor = "tmXTLabelFontColor";
%constant NhlNtmXTLabelFontHeightF = "tmXTLabelFontHeightF";
%constant NhlNtmXTLabelFontQuality = "tmXTLabelFontQuality";
%constant NhlNtmXTLabelFontThicknessF = "tmXTLabelFontThicknessF";
%constant NhlNtmXTLabelFuncCode = "tmXTLabelFuncCode";
%constant NhlNtmXTLabelJust = "tmXTLabelJust";
%constant NhlNtmXTLabelStride = "tmXTLabelStride";
%constant NhlNtmXTLabels = "tmXTLabels";
%constant NhlNtmXTLabelsOn = "tmXTLabelsOn";
%constant NhlNtmXTMajorLengthF = "tmXTMajorLengthF";
%constant NhlNtmXTMajorLineColor = "tmXTMajorLineColor";
%constant NhlNtmXTMajorOutwardLengthF = "tmXTMajorOutwardLengthF";
%constant NhlNtmXTMajorThicknessF = "tmXTMajorThicknessF";
%constant NhlNtmXTMaxLabelLenF = "tmXTMaxLabelLenF";
%constant NhlNtmXTMaxTicks = "tmXTMaxTicks";
%constant NhlNtmXTMinLabelSpacingF = "tmXTMinLabelSpacingF";
%constant NhlNtmXTMinorLengthF = "tmXTMinorLengthF";
%constant NhlNtmXTMinorLineColor = "tmXTMinorLineColor";
%constant NhlNtmXTMinorOn = "tmXTMinorOn";
%constant NhlNtmXTMinorOutwardLengthF = "tmXTMinorOutwardLengthF";
%constant NhlNtmXTMinorPerMajor = "tmXTMinorPerMajor";
%constant NhlNtmXTMinorThicknessF = "tmXTMinorThicknessF";
%constant NhlNtmXTMinorValues = "tmXTMinorValues";
%constant NhlNtmXTMode = "tmXTMode";
%constant NhlNtmXTOn = "tmXTOn";
%constant NhlNtmXTPrecision = "tmXTPrecision";
%constant NhlNtmXTStyle = "tmXTStyle";
%constant NhlNtmXTTickEndF = "tmXTTickEndF";
%constant NhlNtmXTTickSpacingF = "tmXTTickSpacingF";
%constant NhlNtmXTTickStartF = "tmXTTickStartF";
%constant NhlNtmXTValues = "tmXTValues";
%constant NhlNtmXUseBottom = "tmXUseBottom";
%constant NhlNtmYLAutoPrecision = "tmYLAutoPrecision";
%constant NhlNtmYLBorderOn = "tmYLBorderOn";
%constant NhlNtmYLDataBottomF = "tmYLDataBottomF";
%constant NhlNtmYLDataTopF = "tmYLDataTopF";
%constant NhlNtmYLFormat = "tmYLFormat";
%constant NhlNtmYLIrrTensionF = "tmYLIrrTensionF";
%constant NhlNtmYLIrregularPoints = "tmYLIrregularPoints";
%constant NhlNtmYLLabelAngleF = "tmYLLabelAngleF";
%constant NhlNtmYLLabelConstantSpacingF = "tmYLLabelConstantSpacingF";
%constant NhlNtmYLLabelDeltaF = "tmYLLabelDeltaF";
%constant NhlNtmYLLabelDirection = "tmYLLabelDirection";
%constant NhlNtmYLLabelFont = "tmYLLabelFont";
%constant NhlNtmYLLabelFontAspectF = "tmYLLabelFontAspectF";
%constant NhlNtmYLLabelFontColor = "tmYLLabelFontColor";
%constant NhlNtmYLLabelFontHeightF = "tmYLLabelFontHeightF";
%constant NhlNtmYLLabelFontQuality = "tmYLLabelFontQuality";
%constant NhlNtmYLLabelFontThicknessF = "tmYLLabelFontThicknessF";
%constant NhlNtmYLLabelFuncCode = "tmYLLabelFuncCode";
%constant NhlNtmYLLabelJust = "tmYLLabelJust";
%constant NhlNtmYLLabelStride = "tmYLLabelStride";
%constant NhlNtmYLLabels = "tmYLLabels";
%constant NhlNtmYLLabelsOn = "tmYLLabelsOn";
%constant NhlNtmYLMajorLengthF = "tmYLMajorLengthF";
%constant NhlNtmYLMajorLineColor = "tmYLMajorLineColor";
%constant NhlNtmYLMajorOutwardLengthF = "tmYLMajorOutwardLengthF";
%constant NhlNtmYLMajorThicknessF = "tmYLMajorThicknessF";
%constant NhlNtmYLMaxLabelLenF = "tmYLMaxLabelLenF";
%constant NhlNtmYLMaxTicks = "tmYLMaxTicks";
%constant NhlNtmYLMinLabelSpacingF = "tmYLMinLabelSpacingF";
%constant NhlNtmYLMinorLengthF = "tmYLMinorLengthF";
%constant NhlNtmYLMinorLineColor = "tmYLMinorLineColor";
%constant NhlNtmYLMinorOn = "tmYLMinorOn";
%constant NhlNtmYLMinorOutwardLengthF = "tmYLMinorOutwardLengthF";
%constant NhlNtmYLMinorPerMajor = "tmYLMinorPerMajor";
%constant NhlNtmYLMinorThicknessF = "tmYLMinorThicknessF";
%constant NhlNtmYLMinorValues = "tmYLMinorValues";
%constant NhlNtmYLMode = "tmYLMode";
%constant NhlNtmYLOn = "tmYLOn";
%constant NhlNtmYLPrecision = "tmYLPrecision";
%constant NhlNtmYLStyle = "tmYLStyle";
%constant NhlNtmYLTickEndF = "tmYLTickEndF";
%constant NhlNtmYLTickSpacingF = "tmYLTickSpacingF";
%constant NhlNtmYLTickStartF = "tmYLTickStartF";
%constant NhlNtmYLValues = "tmYLValues";
%constant NhlNtmYMajorGrid = "tmYMajorGrid";
%constant NhlNtmYMajorGridLineColor = "tmYMajorGridLineColor";
%constant NhlNtmYMajorGridLineDashPattern = "tmYMajorGridLineDashPattern";
%constant NhlNtmYMajorGridThicknessF = "tmYMajorGridThicknessF";
%constant NhlNtmYMinorGrid = "tmYMinorGrid";
%constant NhlNtmYMinorGridLineColor = "tmYMinorGridLineColor";
%constant NhlNtmYMinorGridLineDashPattern = "tmYMinorGridLineDashPattern";
%constant NhlNtmYMinorGridThicknessF = "tmYMinorGridThicknessF";
%constant NhlNtmYRAutoPrecision = "tmYRAutoPrecision";
%constant NhlNtmYRBorderOn = "tmYRBorderOn";
%constant NhlNtmYRDataBottomF = "tmYRDataBottomF";
%constant NhlNtmYRDataTopF = "tmYRDataTopF";
%constant NhlNtmYRFormat = "tmYRFormat";
%constant NhlNtmYRIrrTensionF = "tmYRIrrTensionF";
%constant NhlNtmYRIrregularPoints = "tmYRIrregularPoints";
%constant NhlNtmYRLabelAngleF = "tmYRLabelAngleF";
%constant NhlNtmYRLabelConstantSpacingF = "tmYRLabelConstantSpacingF";
%constant NhlNtmYRLabelDeltaF = "tmYRLabelDeltaF";
%constant NhlNtmYRLabelDirection = "tmYRLabelDirection";
%constant NhlNtmYRLabelFont = "tmYRLabelFont";
%constant NhlNtmYRLabelFontAspectF = "tmYRLabelFontAspectF";
%constant NhlNtmYRLabelFontColor = "tmYRLabelFontColor";
%constant NhlNtmYRLabelFontHeightF = "tmYRLabelFontHeightF";
%constant NhlNtmYRLabelFontQuality = "tmYRLabelFontQuality";
%constant NhlNtmYRLabelFontThicknessF = "tmYRLabelFontThicknessF";
%constant NhlNtmYRLabelFuncCode = "tmYRLabelFuncCode";
%constant NhlNtmYRLabelJust = "tmYRLabelJust";
%constant NhlNtmYRLabelStride = "tmYRLabelStride";
%constant NhlNtmYRLabels = "tmYRLabels";
%constant NhlNtmYRLabelsOn = "tmYRLabelsOn";
%constant NhlNtmYRMajorLengthF = "tmYRMajorLengthF";
%constant NhlNtmYRMajorLineColor = "tmYRMajorLineColor";
%constant NhlNtmYRMajorOutwardLengthF = "tmYRMajorOutwardLengthF";
%constant NhlNtmYRMajorThicknessF = "tmYRMajorThicknessF";
%constant NhlNtmYRMaxLabelLenF = "tmYRMaxLabelLenF";
%constant NhlNtmYRMaxTicks = "tmYRMaxTicks";
%constant NhlNtmYRMinLabelSpacingF = "tmYRMinLabelSpacingF";
%constant NhlNtmYRMinorLengthF = "tmYRMinorLengthF";
%constant NhlNtmYRMinorLineColor = "tmYRMinorLineColor";
%constant NhlNtmYRMinorOn = "tmYRMinorOn";
%constant NhlNtmYRMinorOutwardLengthF = "tmYRMinorOutwardLengthF";
%constant NhlNtmYRMinorPerMajor = "tmYRMinorPerMajor";
%constant NhlNtmYRMinorThicknessF = "tmYRMinorThicknessF";
%constant NhlNtmYRMinorValues = "tmYRMinorValues";
%constant NhlNtmYRMode = "tmYRMode";
%constant NhlNtmYROn = "tmYROn";
%constant NhlNtmYRPrecision = "tmYRPrecision";
%constant NhlNtmYRStyle = "tmYRStyle";
%constant NhlNtmYRTickEndF = "tmYRTickEndF";
%constant NhlNtmYRTickSpacingF = "tmYRTickSpacingF";
%constant NhlNtmYRTickStartF = "tmYRTickStartF";
%constant NhlNtmYRValues = "tmYRValues";
%constant NhlNtmYUseLeft = "tmYUseLeft";
%constant NhlNtrXAxisType = "trXAxisType";
%constant NhlNtrXCoordPoints = "trXCoordPoints";
%constant NhlNtrXInterPoints = "trXInterPoints";
%constant NhlNtrXSamples = "trXSamples";
%constant NhlNtrXTensionF = "trXTensionF";
%constant NhlNtrYAxisType = "trYAxisType";
%constant NhlNtrYCoordPoints = "trYCoordPoints";
%constant NhlNtrYInterPoints = "trYInterPoints";
%constant NhlNtrYSamples = "trYSamples";
%constant NhlNtrYTensionF = "trYTensionF";
%constant NhlNtrXLog = "trXLog";
%constant NhlNtrYLog = "trYLog";
%constant NhlNtrLineInterpolationOn = "trLineInterpolationOn";
%constant NhlNtrXMaxF = "trXMaxF";
%constant NhlNtrXMinF = "trXMinF";
%constant NhlNtrXReverse = "trXReverse";
%constant NhlNtrYMaxF = "trYMaxF";
%constant NhlNtrYMinF = "trYMinF";
%constant NhlNtrYReverse = "trYReverse";
%constant NhlNtxAngleF = "txAngleF";
%constant NhlNtxBackgroundFillColor = "txBackgroundFillColor";
%constant NhlNtxConstantSpacingF = "txConstantSpacingF";
%constant NhlNtxDirection = "txDirection";
%constant NhlNtxFont = "txFont";
%constant NhlNtxFontAspectF = "txFontAspectF";
%constant NhlNtxFontColor = "txFontColor";
%constant NhlNtxFontHeightF = "txFontHeightF";
%constant NhlNtxFontQuality = "txFontQuality";
%constant NhlNtxFontThicknessF = "txFontThicknessF";
%constant NhlNtxFuncCode = "txFuncCode";
%constant NhlNtxJust = "txJust";
%constant NhlNtxPerimColor = "txPerimColor";
%constant NhlNtxPerimDashLengthF = "txPerimDashLengthF";
%constant NhlNtxPerimDashPattern = "txPerimDashPattern";
%constant NhlNtxPerimOn = "txPerimOn";
%constant NhlNtxPerimSpaceF = "txPerimSpaceF";
%constant NhlNtxPerimThicknessF = "txPerimThicknessF";
%constant NhlNtxPosXF = "txPosXF";
%constant NhlNtxPosYF = "txPosYF";
%constant NhlNtxString = "txString";
%constant NhlNvcExplicitLabelBarLabelsOn = "vcExplicitLabelBarLabelsOn";
%constant NhlNvcFillArrowEdgeColor = "vcFillArrowEdgeColor";
%constant NhlNvcFillArrowEdgeThicknessF = "vcFillArrowEdgeThicknessF";
%constant NhlNvcFillArrowFillColor = "vcFillArrowFillColor";
%constant NhlNvcFillArrowHeadInteriorXF = "vcFillArrowHeadInteriorXF";
%constant NhlNvcFillArrowHeadMinFracXF = "vcFillArrowHeadMinFracXF";
%constant NhlNvcFillArrowHeadMinFracYF = "vcFillArrowHeadMinFracYF";
%constant NhlNvcFillArrowHeadXF = "vcFillArrowHeadXF";
%constant NhlNvcFillArrowHeadYF = "vcFillArrowHeadYF";
%constant NhlNvcFillArrowMinFracWidthF = "vcFillArrowMinFracWidthF";
%constant NhlNvcFillArrowWidthF = "vcFillArrowWidthF";
%constant NhlNvcFillArrowsOn = "vcFillArrowsOn";
%constant NhlNvcFillOverEdge = "vcFillOverEdge";
%constant NhlNvcGlyphStyle = "vcGlyphStyle";
%constant NhlNvcLabelBarEndLabelsOn = "vcLabelBarEndLabelsOn";
%constant NhlNvcLabelFontColor = "vcLabelFontColor";
%constant NhlNvcLabelFontHeightF = "vcLabelFontHeightF";
%constant NhlNvcLabelsOn = "vcLabelsOn";
%constant NhlNvcLabelsUseVectorColor = "vcLabelsUseVectorColor";
%constant NhlNvcLevelColors = "vcLevelColors";
%constant NhlNvcLevelCount = "vcLevelCount";
%constant NhlNvcLevelSelectionMode = "vcLevelSelectionMode";
%constant NhlNvcLevelSpacingF = "vcLevelSpacingF";
%constant NhlNvcLevels = "vcLevels";
%constant NhlNvcLineArrowHeadMaxSizeF = "vcLineArrowHeadMaxSizeF";
%constant NhlNvcLineArrowHeadMinSizeF = "vcLineArrowHeadMinSizeF";
%constant NhlNvcLineArrowThicknessF = "vcLineArrowThicknessF";
%constant NhlNvcMagnitudeFormat = "vcMagnitudeFormat";
%constant NhlNvcMagnitudeScaleFactorF = "vcMagnitudeScaleFactorF";
%constant NhlNvcMagnitudeScaleValueF = "vcMagnitudeScaleValueF";
%constant NhlNvcMagnitudeScalingMode = "vcMagnitudeScalingMode";
%constant NhlNvcMapDirection = "vcMapDirection";
%constant NhlNvcMaxLevelCount = "vcMaxLevelCount";
%constant NhlNvcMaxLevelValF = "vcMaxLevelValF";
%constant NhlNvcMaxMagnitudeF = "vcMaxMagnitudeF";
%constant NhlNvcMinAnnoAngleF = "vcMinAnnoAngleF";
%constant NhlNvcMinAnnoArrowAngleF = "vcMinAnnoArrowAngleF";
%constant NhlNvcMinAnnoArrowEdgeColor = "vcMinAnnoArrowEdgeColor";
%constant NhlNvcMinAnnoArrowFillColor = "vcMinAnnoArrowFillColor";
%constant NhlNvcMinAnnoArrowLineColor = "vcMinAnnoArrowLineColor";
%constant NhlNvcMinAnnoArrowMinOffsetF = "vcMinAnnoArrowMinOffsetF";
%constant NhlNvcMinAnnoArrowSpaceF = "vcMinAnnoArrowSpaceF";
%constant NhlNvcMinAnnoArrowUseVecColor = "vcMinAnnoArrowUseVecColor";
%constant NhlNvcMinAnnoBackgroundColor = "vcMinAnnoBackgroundColor";
%constant NhlNvcMinAnnoConstantSpacingF = "vcMinAnnoConstantSpacingF";
%constant NhlNvcMinAnnoExplicitMagnitudeF = "vcMinAnnoExplicitMagnitudeF";
%constant NhlNvcMinAnnoFont = "vcMinAnnoFont";
%constant NhlNvcMinAnnoFontAspectF = "vcMinAnnoFontAspectF";
%constant NhlNvcMinAnnoFontColor = "vcMinAnnoFontColor";
%constant NhlNvcMinAnnoFontHeightF = "vcMinAnnoFontHeightF";
%constant NhlNvcMinAnnoFontQuality = "vcMinAnnoFontQuality";
%constant NhlNvcMinAnnoFontThicknessF = "vcMinAnnoFontThicknessF";
%constant NhlNvcMinAnnoFuncCode = "vcMinAnnoFuncCode";
%constant NhlNvcMinAnnoJust = "vcMinAnnoJust";
%constant NhlNvcMinAnnoOn = "vcMinAnnoOn";
%constant NhlNvcMinAnnoOrientation = "vcMinAnnoOrientation";
%constant NhlNvcMinAnnoOrthogonalPosF = "vcMinAnnoOrthogonalPosF";
%constant NhlNvcMinAnnoParallelPosF = "vcMinAnnoParallelPosF";
%constant NhlNvcMinAnnoPerimColor = "vcMinAnnoPerimColor";
%constant NhlNvcMinAnnoPerimOn = "vcMinAnnoPerimOn";
%constant NhlNvcMinAnnoPerimSpaceF = "vcMinAnnoPerimSpaceF";
%constant NhlNvcMinAnnoPerimThicknessF = "vcMinAnnoPerimThicknessF";
%constant NhlNvcMinAnnoSide = "vcMinAnnoSide";
%constant NhlNvcMinAnnoString1 = "vcMinAnnoString1";
%constant NhlNvcMinAnnoString1On = "vcMinAnnoString1On";
%constant NhlNvcMinAnnoString2 = "vcMinAnnoString2";
%constant NhlNvcMinAnnoString2On = "vcMinAnnoString2On";
%constant NhlNvcMinAnnoTextDirection = "vcMinAnnoTextDirection";
%constant NhlNvcMinAnnoZone = "vcMinAnnoZone";
%constant NhlNvcMinDistanceF = "vcMinDistanceF";
%constant NhlNvcMinFracLengthF = "vcMinFracLengthF";
%constant NhlNvcMinLevelValF = "vcMinLevelValF";
%constant NhlNvcMinMagnitudeF = "vcMinMagnitudeF";
%constant NhlNvcMonoFillArrowEdgeColor = "vcMonoFillArrowEdgeColor";
%constant NhlNvcMonoFillArrowFillColor = "vcMonoFillArrowFillColor";
%constant NhlNvcMonoLineArrowColor = "vcMonoLineArrowColor";
%constant NhlNvcMonoWindBarbColor = "vcMonoWindBarbColor";
%constant NhlNvcNoDataLabelOn = "vcNoDataLabelOn";
%constant NhlNvcNoDataLabelString = "vcNoDataLabelString";
%constant NhlNvcPositionMode = "vcPositionMode";
%constant NhlNvcRefAnnoAngleF = "vcRefAnnoAngleF";
%constant NhlNvcRefAnnoArrowAngleF = "vcRefAnnoArrowAngleF";
%constant NhlNvcRefAnnoArrowEdgeColor = "vcRefAnnoArrowEdgeColor";
%constant NhlNvcRefAnnoArrowFillColor = "vcRefAnnoArrowFillColor";
%constant NhlNvcRefAnnoArrowLineColor = "vcRefAnnoArrowLineColor";
%constant NhlNvcRefAnnoArrowMinOffsetF = "vcRefAnnoArrowMinOffsetF";
%constant NhlNvcRefAnnoArrowSpaceF = "vcRefAnnoArrowSpaceF";
%constant NhlNvcRefAnnoArrowUseVecColor = "vcRefAnnoArrowUseVecColor";
%constant NhlNvcRefAnnoBackgroundColor = "vcRefAnnoBackgroundColor";
%constant NhlNvcRefAnnoConstantSpacingF = "vcRefAnnoConstantSpacingF";
%constant NhlNvcRefAnnoExplicitMagnitudeF = "vcRefAnnoExplicitMagnitudeF";
%constant NhlNvcRefAnnoFont = "vcRefAnnoFont";
%constant NhlNvcRefAnnoFontAspectF = "vcRefAnnoFontAspectF";
%constant NhlNvcRefAnnoFontColor = "vcRefAnnoFontColor";
%constant NhlNvcRefAnnoFontHeightF = "vcRefAnnoFontHeightF";
%constant NhlNvcRefAnnoFontQuality = "vcRefAnnoFontQuality";
%constant NhlNvcRefAnnoFontThicknessF = "vcRefAnnoFontThicknessF";
%constant NhlNvcRefAnnoFuncCode = "vcRefAnnoFuncCode";
%constant NhlNvcRefAnnoJust = "vcRefAnnoJust";
%constant NhlNvcRefAnnoOn = "vcRefAnnoOn";
%constant NhlNvcRefAnnoOrientation = "vcRefAnnoOrientation";
%constant NhlNvcRefAnnoOrthogonalPosF = "vcRefAnnoOrthogonalPosF";
%constant NhlNvcRefAnnoParallelPosF = "vcRefAnnoParallelPosF";
%constant NhlNvcRefAnnoPerimColor = "vcRefAnnoPerimColor";
%constant NhlNvcRefAnnoPerimOn = "vcRefAnnoPerimOn";
%constant NhlNvcRefAnnoPerimSpaceF = "vcRefAnnoPerimSpaceF";
%constant NhlNvcRefAnnoPerimThicknessF = "vcRefAnnoPerimThicknessF";
%constant NhlNvcRefAnnoSide = "vcRefAnnoSide";
%constant NhlNvcRefAnnoString1 = "vcRefAnnoString1";
%constant NhlNvcRefAnnoString1On = "vcRefAnnoString1On";
%constant NhlNvcRefAnnoString2 = "vcRefAnnoString2";
%constant NhlNvcRefAnnoString2On = "vcRefAnnoString2On";
%constant NhlNvcRefAnnoTextDirection = "vcRefAnnoTextDirection";
%constant NhlNvcRefAnnoZone = "vcRefAnnoZone";
%constant NhlNvcRefLengthF = "vcRefLengthF";
%constant NhlNvcRefMagnitudeF = "vcRefMagnitudeF";
%constant NhlNvcScalarFieldData = "vcScalarFieldData";
%constant NhlNvcScalarMissingValColor = "vcScalarMissingValColor";
%constant NhlNvcScalarValueFormat = "vcScalarValueFormat";
%constant NhlNvcScalarValueScaleFactorF = "vcScalarValueScaleFactorF";
%constant NhlNvcScalarValueScaleValueF = "vcScalarValueScaleValueF";
%constant NhlNvcScalarValueScalingMode = "vcScalarValueScalingMode";
%constant NhlNvcUseRefAnnoRes = "vcUseRefAnnoRes";
%constant NhlNvcUseScalarArray = "vcUseScalarArray";
%constant NhlNvcVectorDrawOrder = "vcVectorDrawOrder";
%constant NhlNvcVectorFieldData = "vcVectorFieldData";
%constant NhlNvcWindBarbCalmCircleSizeF = "vcWindBarbCalmCircleSizeF";
%constant NhlNvcWindBarbColor = "vcWindBarbColor";
%constant NhlNvcWindBarbLineThicknessF = "vcWindBarbLineThicknessF";
%constant NhlNvcWindBarbScaleFactorF = "vcWindBarbScaleFactorF";
%constant NhlNvcWindBarbTickAngleF = "vcWindBarbTickAngleF";
%constant NhlNvcWindBarbTickLengthF = "vcWindBarbTickLengthF";
%constant NhlNvcWindBarbTickSpacingF = "vcWindBarbTickSpacingF";
%constant NhlNvcZeroFLabelAngleF = "vcZeroFLabelAngleF";
%constant NhlNvcZeroFLabelBackgroundColor = "vcZeroFLabelBackgroundColor";
%constant NhlNvcZeroFLabelConstantSpacingF = "vcZeroFLabelConstantSpacingF";
%constant NhlNvcZeroFLabelFont = "vcZeroFLabelFont";
%constant NhlNvcZeroFLabelFontAspectF = "vcZeroFLabelFontAspectF";
%constant NhlNvcZeroFLabelFontColor = "vcZeroFLabelFontColor";
%constant NhlNvcZeroFLabelFontHeightF = "vcZeroFLabelFontHeightF";
%constant NhlNvcZeroFLabelFontQuality = "vcZeroFLabelFontQuality";
%constant NhlNvcZeroFLabelFontThicknessF = "vcZeroFLabelFontThicknessF";
%constant NhlNvcZeroFLabelFuncCode = "vcZeroFLabelFuncCode";
%constant NhlNvcZeroFLabelJust = "vcZeroFLabelJust";
%constant NhlNvcZeroFLabelOn = "vcZeroFLabelOn";
%constant NhlNvcZeroFLabelOrthogonalPosF = "vcZeroFLabelOrthogonalPosF";
%constant NhlNvcZeroFLabelParallelPosF = "vcZeroFLabelParallelPosF";
%constant NhlNvcZeroFLabelPerimColor = "vcZeroFLabelPerimColor";
%constant NhlNvcZeroFLabelPerimOn = "vcZeroFLabelPerimOn";
%constant NhlNvcZeroFLabelPerimSpaceF = "vcZeroFLabelPerimSpaceF";
%constant NhlNvcZeroFLabelPerimThicknessF = "vcZeroFLabelPerimThicknessF";
%constant NhlNvcZeroFLabelSide = "vcZeroFLabelSide";
%constant NhlNvcZeroFLabelString = "vcZeroFLabelString";
%constant NhlNvcZeroFLabelTextDirection = "vcZeroFLabelTextDirection";
%constant NhlNvcZeroFLabelZone = "vcZeroFLabelZone";
%constant NhlNvfYCStartSubsetV = "vfYCStartSubsetV";
%constant NhlNvpAnnoManagerId = "vpAnnoManagerId";
%constant NhlNvpHeightF = "vpHeightF";
%constant NhlNvpKeepAspect = "vpKeepAspect";
%constant NhlNvpOn = "vpOn";
%constant NhlNvpUseSegments = "vpUseSegments";
%constant NhlNvpWidthF = "vpWidthF";
%constant NhlNvpXF = "vpXF";
%constant NhlNvpYF = "vpYF";
%constant NhlNwkMetaName = "wkMetaName";
%constant NhlNwkDeviceLowerX = "wkDeviceLowerX";
%constant NhlNwkDeviceLowerY = "wkDeviceLowerY";
%constant NhlNwkDeviceUpperX = "wkDeviceUpperX";
%constant NhlNwkDeviceUpperY = "wkDeviceUpperY";
%constant NhlNwkPSFileName = "wkPSFileName";
%constant NhlNwkPSFormat = "wkPSFormat";
%constant NhlNwkPSResolution = "wkPSResolution";
%constant NhlNwkPDFFileName = "wkPDFFileName";
%constant NhlNwkPDFFormat = "wkPDFFormat";
%constant NhlNwkPDFResolution = "wkPDFResolution";
%constant NhlNwkVisualType = "wkVisualType";
%constant NhlNwkColorModel = "wkColorModel";
%constant NhlNwkBackgroundColor = "wkBackgroundColor";
%constant NhlNwkColorMap = "wkColorMap";
%constant NhlNwkColorMapLen = "wkColorMapLen";
%constant NhlNwkDashTableLength = "wkDashTableLength";
%constant NhlNwkDefGraphicStyleId = "wkDefGraphicStyleId";
%constant NhlNwkFillTableLength = "wkFillTableLength";
%constant NhlNwkForegroundColor = "wkForegroundColor";
%constant NhlNwkGksWorkId = "wkGksWorkId";
%constant NhlNwkMarkerTableLength = "wkMarkerTableLength";
%constant NhlNwkTopLevelViews = "wkTopLevelViews";
%constant NhlNwkViews = "wkViews";
%constant NhlNwkPause = "wkPause";
%constant NhlNwkWindowId = "wkWindowId";
%constant NhlNwkXColorMode = "wkXColorMode";
%constant NhlNwsCurrentSize = "wsCurrentSize";
%constant NhlNwsMaximumSize = "wsMaximumSize";
%constant NhlNwsThresholdSize = "wsThresholdSize";
%constant NhlNxyComputeXMax = "xyComputeXMax";
%constant NhlNxyComputeXMin = "xyComputeXMin";
%constant NhlNxyComputeYMax = "xyComputeYMax";
%constant NhlNxyComputeYMin = "xyComputeYMin";
%constant NhlNxyCoordData = "xyCoordData";
%constant NhlNxyCoordDataSpec = "xyCoordDataSpec";
%constant NhlNxyCurveDrawOrder = "xyCurveDrawOrder";
%constant NhlNxyDashPattern = "xyDashPattern";
%constant NhlNxyDashPatterns = "xyDashPatterns";
%constant NhlNxyExplicitLabels = "xyExplicitLabels";
%constant NhlNxyExplicitLegendLabels = "xyExplicitLegendLabels";
%constant NhlNxyLabelMode = "xyLabelMode";
%constant NhlNxyLineColor = "xyLineColor";
%constant NhlNxyLineColors = "xyLineColors";
%constant NhlNxyLineDashSegLenF = "xyLineDashSegLenF";
%constant NhlNxyLineLabelConstantSpacingF = "xyLineLabelConstantSpacingF";
%constant NhlNxyLineLabelFont = "xyLineLabelFont";
%constant NhlNxyLineLabelFontAspectF = "xyLineLabelFontAspectF";
%constant NhlNxyLineLabelFontColor = "xyLineLabelFontColor";
%constant NhlNxyLineLabelFontColors = "xyLineLabelFontColors";
%constant NhlNxyLineLabelFontHeightF = "xyLineLabelFontHeightF";
%constant NhlNxyLineLabelFontQuality = "xyLineLabelFontQuality";
%constant NhlNxyLineLabelFontThicknessF = "xyLineLabelFontThicknessF";
%constant NhlNxyLineLabelFuncCode = "xyLineLabelFuncCode";
%constant NhlNxyLineThicknessF = "xyLineThicknessF";
%constant NhlNxyLineThicknesses = "xyLineThicknesses";
%constant NhlNxyMarkLineMode = "xyMarkLineMode";
%constant NhlNxyMarkLineModes = "xyMarkLineModes";
%constant NhlNxyMarker = "xyMarker";
%constant NhlNxyMarkerColor = "xyMarkerColor";
%constant NhlNxyMarkerColors = "xyMarkerColors";
%constant NhlNxyMarkerSizeF = "xyMarkerSizeF";
%constant NhlNxyMarkerSizes = "xyMarkerSizes";
%constant NhlNxyMarkerThicknessF = "xyMarkerThicknessF";
%constant NhlNxyMarkerThicknesses = "xyMarkerThicknesses";
%constant NhlNxyMarkers = "xyMarkers";
%constant NhlNxyMonoDashPattern = "xyMonoDashPattern";
%constant NhlNxyMonoLineColor = "xyMonoLineColor";
%constant NhlNxyMonoLineLabelFontColor = "xyMonoLineLabelFontColor";
%constant NhlNxyMonoLineThickness = "xyMonoLineThickness";
%constant NhlNxyMonoMarkLineMode = "xyMonoMarkLineMode";
%constant NhlNxyMonoMarker = "xyMonoMarker";
%constant NhlNxyMonoMarkerColor = "xyMonoMarkerColor";
%constant NhlNxyMonoMarkerSize = "xyMonoMarkerSize";
%constant NhlNxyMonoMarkerThickness = "xyMonoMarkerThickness";
%constant NhlNxyXIrrTensionF = "xyXIrrTensionF";
%constant NhlNxyXIrregularPoints = "xyXIrregularPoints";
%constant NhlNxyXStyle = "xyXStyle";
%constant NhlNxyYIrrTensionF = "xyYIrrTensionF";
%constant NhlNxyYIrregularPoints = "xyYIrregularPoints";
%constant NhlNxyYStyle = "xyYStyle";

%constant NhlTFillIndexFullEnum = "FillIndexFullEnum";
%constant NhlTFillIndexFullEnumGenArray = "FillIndexFullEnumGenArray";
%constant NhlUNSPECIFIEDFILL = -2;

%constant NhlTFillIndex = "FillIndex";
%constant NhlTFillIndexGenArray = "FillIndexGenArray";
%constant NhlHOLLOWFILL = -1;
%constant NhlNULLFILL = -1;
%constant NhlSOLIDFILL = 0;
%constant NhlWK_INITIAL_FILL_BUFSIZE = 128;

%include cpointer.i
%pointer_functions(int, intp);

%include carrays.i
%array_functions(float,floatArray)

//
// Include the required NumPy header.
//
%header %{
#include <Numeric/arrayobject.h>
%}

//
//  Required function call for NumPy extension modules.
//
%init %{
import_array();
%}

%wrapper %{
%}

%include "typemaps.i"


//
// Typemaps for converting NumPy Python arrays to C arrays.
//

//
// numinputs=0 indicates to SWIG that an argument which is an
// input argument in the C code, but is an output argument
// in Python should not be counted as one of the Python input
// arguments.
//

%typemap (in) float *sequence_as_float {
  int i,ndims,tdims=1;
  PyArrayObject *arr;
  arr =
   (PyArrayObject *) PyArray_ContiguousFromObject($input,PyArray_DOUBLE,0,0);
  ndims = arr->nd;
  for (i = 0; i < ndims; i++) {
    tdims *= arr->dimensions[i];
  }
  $1 = (float *) d2f(tdims, (double *) arr->data);
}

%typemap (in) nglRes *rlist {
  $1 = (void *) &nglRlist;
}

%typemap (in) double *sequence_as_double {
  PyArrayObject *arr;
  arr =
   (PyArrayObject *) PyArray_ContiguousFromObject($input,PyArray_DOUBLE,0,0);
  $1 = (double *) arr->data;
}

%typemap (in) void *sequence_as_void {
  PyArrayObject *arr;
  arr =
   (PyArrayObject *) PyArray_ContiguousFromObject($input,PyArray_DOUBLE,0,0);
  $1 = (void *) arr->data;
}

%typemap (in) int *sequence_as_int {
  PyArrayObject *arr;
  arr =
   (PyArrayObject *) PyArray_ContiguousFromObject($input,PyArray_INT,0,0);
  $1 = (int *) arr->data;
}

%typemap (argout) (int *numberf) {
  int dims[1];
  dims[0] = *($1);
  $result = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,(char *) result);
}
%typemap(in,numinputs=0) int *numberf (int tempx) {
  $1 = &tempx;
}

%typemap (argout) (char *string_out) {
  $result = $1;  
}
%typemap(in,numinputs=0) char *string_out (char **tempc) {
  $1 = tempc;
}

%typemap (argout) (int *numberd) {
  int dims[1];
  dims[0] = *($1);
  $result = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_DOUBLE,(char *) result);
}
%typemap(in,numinputs=0) int *numberd (int tempx) {
  $1 = &tempx;
}

%typemap (argout) (int nxir, int nyir, float *p_array_float_out[]) {
  int dims[2];
  PyObject *o;
  dims[0] = arg10;
  dims[1] = arg11;
  o = (PyObject *)PyArray_FromDimsAndData(2,dims,PyArray_FLOAT,
                  (char *) arg12[0]);
  resultobj = t_output_helper(resultobj,o);
}
%typemap(in,numinputs=0) float *p_array_float_out[] (float *tempx) {
  $1 = &tempx;
}

%typemap (argout) (int nxir, int nyir, double *p_array_double_out[]) {
  int dims[2];
  PyObject *o;
  dims[0] = $1;
  dims[1] = $2;
  o = (PyObject *)PyArray_FromDimsAndData(2,dims,PyArray_DOUBLE,
                  (char *) $3[0]);
  resultobj = t_output_helper(resultobj,o);
}
%typemap(in,numinputs=0) double *p_array_double_out[] (double *tempx) {
  $1 = &tempx;
}

%typemap (argout)(int nxir, int nyir, int nzir, double *p_array3_double_out[]) {
  int dims[3];
  PyObject *o;
  dims[0] = $1;
  dims[1] = $2;
  dims[2] = $3;
  o = (PyObject *)PyArray_FromDimsAndData(3,dims,PyArray_DOUBLE,
                  (char *) $4[0]);
  resultobj = t_output_helper(resultobj,o);
}
%typemap(in,numinputs=0) double *p_array3_double_out[] (double *tempx) {
  $1 = &tempx;
}

%typemap (argout) (int nx, float *t_array_float_out) {
  int dims[1];
  PyObject *o;
  dims[0] = nx;
  o = (PyObject *)PyArray_FromDimsAndData(2,dims,PyArray_FLOAT,
                  (char *) arg7);
  resultobj = t_output_helper(resultobj,o);
}
%typemap(in,numinputs=0) float *t_array_float_out[] (float *tempx) {
  $1 = &tempx;
}

%typemap (argout) (int *int_out) {
  $result = (PyObject *) PyInt_FromLong((long) result);
}
%typemap(in,numinputs=0) int *int_out (int tempx) {
  $1 = &tempx;
}

%typemap (argout) (int *numberi) {
  int dims[1];
  dims[0] = *($1);
  $result = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_INT,(char *) result);
}
%typemap(in,numinputs=0) int *numberi (int tempx) {
  $1 = &tempx;
}

%typemap(in,numinputs=0) int *numbers (int tempx) %{
  $1 = &tempx;
%}

%typemap (argout) (float **data_addr, int *num_elements) {
  int dims[1];
  PyObject *o;
  dims[0] = *$2;
  o = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,(char *) *$1);
  $result = t_output_helper($result,o);
}
%typemap(in,numinputs=0) float **data_addr (float *temp) {
  $1 = &temp;
}
%typemap(in,numinputs=0) int *num_elements (int temp) {
  $1 = &temp;
}

%typemap (out) nglPlotId {
 PyObject *return_list;
 PyObject *l_base,      *l_contour, *l_vector,  *l_streamline,
          *l_map,       *l_xy     , *l_xydspec, *l_text,
          *l_primitive, *l_cafield, *l_sffield, *l_vffield;
 nglPlotId pid;
 int i;

 pid = $1;

 return_list = PyList_New(12);

 if (pid.nbase == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,0,Py_None);
 }
 else {
   l_base = PyList_New(pid.nbase);
   for (i = 0; i < pid.nbase; i++) {
     PyList_SetItem(l_base,i,PyInt_FromLong((long) *(pid.base+i)));
   }
   PyList_SetItem(return_list,0,l_base);
 }

 if (pid.ncontour == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,1,Py_None);
 }
 else {
   l_contour = PyList_New(pid.ncontour);
   for (i = 0; i < pid.ncontour; i++) {
     PyList_SetItem(l_contour,i,PyInt_FromLong((long) *(pid.contour+i)));
   }
   PyList_SetItem(return_list,1,l_contour);
 }

 if (pid.nvector == 0) {
   PyList_SetItem(return_list,2,Py_None);
 }
 else {
   l_vector = PyList_New(pid.nvector);
   for (i = 0; i < pid.nvector; i++) {
     PyList_SetItem(l_vector,i,PyInt_FromLong((long) *(pid.vector+i)));
   }
   PyList_SetItem(return_list,2,l_vector);
 }

 if (pid.nstreamline == 0) {
   PyList_SetItem(return_list,3,Py_None);
 }
 else {
   l_streamline = PyList_New(pid.nstreamline);
   for (i = 0; i < pid.nstreamline; i++) {
     PyList_SetItem(l_streamline,i,PyInt_FromLong((long) *(pid.streamline+i)));
   }
   PyList_SetItem(return_list,3,l_streamline);
 }

 if (pid.nmap == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,4,Py_None);
 }
 else {
   l_map = PyList_New(pid.nmap);
   for (i = 0; i < pid.nmap; i++) {
     PyList_SetItem(l_map,i,PyInt_FromLong((long) *(pid.map+i)));
   }
   PyList_SetItem(return_list,4,l_map);
 }

 if (pid.nxy == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,5,Py_None);
 }
 else {
   l_xy = PyList_New(pid.nxy);
   for (i = 0; i < pid.nxy; i++) {
     PyList_SetItem(l_xy,i,PyInt_FromLong((long) *(pid.xy+i)));
   }
   PyList_SetItem(return_list,5,l_xy);
 }

 if (pid.nxydspec == 0) {
   PyList_SetItem(return_list,6,Py_None);
 }
 else {
   l_xydspec = PyList_New(pid.nxydspec);
   for (i = 0; i < pid.nxydspec; i++) {
     PyList_SetItem(l_xydspec,i,PyInt_FromLong((long) *(pid.xydspec+i)));
   }
   PyList_SetItem(return_list,6,l_xydspec);
 }

 if (pid.ntext == 0) {
   PyList_SetItem(return_list,7,Py_None);
 }
 else {
   l_text = PyList_New(pid.ntext);
   for (i = 0; i < pid.ntext; i++) {
     PyList_SetItem(l_text,i,PyInt_FromLong((long) *(pid.text+i)));
   }
   PyList_SetItem(return_list,7,l_text);
 }

 if (pid.nprimitive == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,8,Py_None);
 }
 else {
   l_primitive = PyList_New(pid.nprimitive);
   for (i = 0; i < pid.nprimitive; i++) {
     PyList_SetItem(l_primitive,i,PyInt_FromLong((long) *(pid.primitive+i)));
   }
   PyList_SetItem(return_list,8,l_primitive);
 }

 if (pid.ncafield == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,9,Py_None);
 }
 else {
   l_cafield = PyList_New(pid.ncafield);
   for (i = 0; i < pid.ncafield; i++) {
     PyList_SetItem(l_cafield,i,PyInt_FromLong((long) *(pid.cafield+i)));
   }
   PyList_SetItem(return_list,9,l_cafield);
 }

 if (pid.nsffield == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,10,Py_None);
 }
 else {
   l_sffield = PyList_New(pid.nsffield);
   for (i = 0; i < pid.nsffield; i++) {
     PyList_SetItem(l_sffield,i,PyInt_FromLong((long) *(pid.sffield+i)));
   }
   PyList_SetItem(return_list,10,l_sffield);
 }

 if (pid.nvffield == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,11,Py_None);
 }
 else {
   l_vffield = PyList_New(pid.nvffield);
   for (i = 0; i < pid.nvffield; i++) {
     PyList_SetItem(l_vffield,i,PyInt_FromLong((long) *(pid.vffield+i)));
   }
   PyList_SetItem(return_list,11,l_vffield);
 }

 Py_INCREF(return_list); 
 $result = return_list;
}

%typemap(in) (nglPlotId *plot_seq) {
  nglPlotId *inlist;
  PyObject *ptmp;
  PyObject *l_base,      *l_contour, *l_vector,  *l_streamline,
           *l_map,       *l_xy     , *l_xydspec, *l_text,
           *l_primitive, *l_cafield, *l_sffield, *l_vffield;
  int      *ibase,       *icontour,  *ivector,   *istreamline,
           *imap,        *ixy     ,  *ixydspec,  *itext,
           *iprimitive,  *icafield,  *isffield,  *ivffield;
  int      i,j;

  if (!PyList_Check($input)) {
    printf("PlotIds must be a Python list\n");
    Py_INCREF(Py_None);
    return Py_None;
  }

  inlist = (nglPlotId *) malloc(PyList_Size($input)*sizeof(nglPlotId));

  for (j = 0; j < PyList_Size($input); j++) {
    ptmp = PyList_GetItem($input,j);  /* ptmp is an Ngl Python plot id */

    l_base = PyList_GetItem(ptmp,0);
    if (l_base == Py_None) {
      inlist[j].nbase = 0;
      inlist[j].base = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_base)) {
        printf("PlotId base element must be None or a Python list\n");
      }
      else {
        inlist[j].nbase = PyList_Size(l_base);
        ibase = (int *) malloc(inlist[j].nbase*sizeof(int));
        for (i = 0; i < inlist[j].nbase; i++) {
          *(ibase+i) = (int) PyInt_AsLong(PyList_GetItem(l_base,i));
        }
        inlist[j].base = ibase;
      }
    }

    l_contour = PyList_GetItem(ptmp,1);
    if (l_contour == Py_None) {
      inlist[j].ncontour = 0;
      inlist[j].contour = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_contour)) {
        printf("PlotId contour element must be None or a Python list\n");
      }
      else {
        inlist[j].ncontour = PyList_Size(l_contour);
        icontour = (int *) malloc(inlist[j].ncontour*sizeof(int));
        for (i = 0; i < inlist[j].ncontour; i++) {
          *(icontour+i) = (int) PyInt_AsLong(PyList_GetItem(l_contour,i));
        }
        inlist[j].contour = icontour;
      }
    }

    l_vector = PyList_GetItem(ptmp,2);
    if (l_vector == Py_None) {
      inlist[j].nvector = 0;
      inlist[j].vector = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_vector)) {
        printf("PlotId vector element must be None or a Python list\n");
      }
      else {
        inlist[j].nvector = PyList_Size(l_vector);
        ivector = (int *) malloc(inlist[j].nvector*sizeof(int));
        for (i = 0; i < inlist[j].nvector; i++) {
          *(ivector+i) = (int) PyInt_AsLong(PyList_GetItem(l_vector,i));
        }
        inlist[j].vector = ivector;
      }
    }

    l_streamline = PyList_GetItem(ptmp,3);
    if (l_streamline == Py_None) {
      inlist[j].nstreamline = 0;
      inlist[j].streamline = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_streamline)) {
        printf("PlotId streamline element must be None or a Python list\n");
      }
      else {
        inlist[j].nstreamline = PyList_Size(l_streamline);
        istreamline = (int *) malloc(inlist[j].nstreamline*sizeof(int));
        for (i = 0; i < inlist[j].nstreamline; i++) {
          *(istreamline+i) = (int) PyInt_AsLong(PyList_GetItem(l_streamline,i));
        }
        inlist[j].streamline = istreamline;
      }
    }

    l_map = PyList_GetItem(ptmp,4);
    if (l_map == Py_None) {
      inlist[j].nmap = 0;
      inlist[j].map = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_map)) {
        printf("PlotId map element must be None or a Python list\n");
      }
      else {
        inlist[j].nmap = PyList_Size(l_map);
        imap = (int *) malloc(inlist[j].nmap*sizeof(int));
        for (i = 0; i < inlist[j].nmap; i++) {
          *(imap+i) = (int) PyInt_AsLong(PyList_GetItem(l_map,i));
        }
        inlist[j].map = imap;
      }
    }

    l_xy = PyList_GetItem(ptmp,5);
    if (l_xy == Py_None) {
      inlist[j].nxy = 0;
      inlist[j].xy = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_xy)) {
        printf("PlotId xy element must be None or a Python list\n");
      }
      else {
        inlist[j].nxy = PyList_Size(l_xy);
        ixy = (int *) malloc(inlist[j].nxy*sizeof(int));
        for (i = 0; i < inlist[j].nxy; i++) {
          *(ixy+i) = (int) PyInt_AsLong(PyList_GetItem(l_xy,i));
        }
        inlist[j].xy = ixy;
      }
    }

    l_xydspec = PyList_GetItem(ptmp,6);
    if (l_xydspec == Py_None) {
      inlist[j].nxydspec = 0;
      inlist[j].xydspec = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_xydspec)) {
        printf("PlotId xydspec element must be None or a Python list\n");
      }
      else {
        inlist[j].nxydspec = PyList_Size(l_xydspec);
        ixydspec = (int *) malloc(inlist[j].nxydspec*sizeof(int));
        for (i = 0; i < inlist[j].nxydspec; i++) {
          *(ixydspec+i) = (int) PyInt_AsLong(PyList_GetItem(l_xydspec,i));
        }
        inlist[j].xydspec = ixydspec;
      }
    }

    l_text = PyList_GetItem(ptmp,7);
    if (l_text == Py_None) {
      inlist[j].ntext = 0;
      inlist[j].text = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_text)) {
        printf("PlotId text element must be None or a Python list\n");
      }
      else {
        inlist[j].ntext = PyList_Size(l_text);
        itext = (int *) malloc(inlist[j].ntext*sizeof(int));
        for (i = 0; i < inlist[j].ntext; i++) {
          *(itext+i) = (int) PyInt_AsLong(PyList_GetItem(l_text,i));
        }
        inlist[j].text = itext;
      }
    }

    l_primitive = PyList_GetItem(ptmp,8);
    if (l_primitive == Py_None) {
      inlist[j].nprimitive = 0;
      inlist[j].primitive = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_primitive)) {
        printf("PlotId primitive element must be None or a Python list\n");
      }
      else {
        inlist[j].nprimitive = PyList_Size(l_primitive);
        iprimitive = (int *) malloc(inlist[j].nprimitive*sizeof(int));
        for (i = 0; i < inlist[j].nprimitive; i++) {
          *(iprimitive+i) = (int) PyInt_AsLong(PyList_GetItem(l_primitive,i));
        }
        inlist[j].primitive = iprimitive;
      }
    }

    l_cafield = PyList_GetItem(ptmp,9);
    if (l_cafield == Py_None) {
      inlist[j].ncafield = 0;
      inlist[j].cafield = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_cafield)) {
        printf("PlotId cafield element must be None or a Python list\n");
      }
      else {
        inlist[j].ncafield = PyList_Size(l_cafield);
        icafield = (int *) malloc(inlist[j].ncafield*sizeof(int));
        for (i = 0; i < inlist[j].ncafield; i++) {
          *(icafield+i) = (int) PyInt_AsLong(PyList_GetItem(l_cafield,i));
        }
        inlist[j].cafield = icafield;
      }
    }

    l_sffield = PyList_GetItem(ptmp,10);
    if (l_sffield == Py_None) {
      inlist[j].nsffield = 0;
      inlist[j].sffield = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_sffield)) {
        printf("PlotId sffield element must be None or a Python list\n");
      }
      else {
        inlist[j].nsffield = PyList_Size(l_sffield);
        isffield = (int *) malloc(inlist[j].nsffield*sizeof(int));
        for (i = 0; i < inlist[j].nsffield; i++) {
          *(isffield+i) = (int) PyInt_AsLong(PyList_GetItem(l_sffield,i));
        }
        inlist[j].sffield = isffield;
      }
    }

    l_vffield = PyList_GetItem(ptmp,11);
    if (l_vffield == Py_None) {
      inlist[j].nvffield = 0;
      inlist[j].vffield = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_vffield)) {
        printf("PlotId vffield element must be None or a Python list\n");
      }
      else {
        inlist[j].nvffield = PyList_Size(l_vffield);
        ivffield = (int *) malloc(inlist[j].nvffield*sizeof(int));
        for (i = 0; i < inlist[j].nvffield; i++) {
          *(ivffield+i) = (int) PyInt_AsLong(PyList_GetItem(l_vffield,i));
        }
        inlist[j].vffield = ivffield;
      }
    }
  }

  $1 = inlist;
}

%typemap (in) nglPlotId *plot {
  int i;
  nglPlotId inlist;

  PyObject *l_base,      *l_contour, *l_vector,  *l_streamline,
           *l_map,       *l_xy     , *l_xydspec, *l_text,
           *l_primitive, *l_cafield, *l_sffield, *l_vffield;
  int      *ibase,       *icontour,  *ivector,   *istreamline,
           *imap,        *ixy     ,  *ixydspec,  *itext,
           *iprimitive,  *icafield,  *isffield,  *ivffield;

  if (PyList_Check($input) == 0) {
    printf("PlotIds must be Python lists\n"); 
  }

  l_base = PyList_GetItem($input,0);
  if (l_base == Py_None) {
    inlist.nbase = 0;
    inlist.base = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_base)) {
      printf("PlotId base element must be None or a Python list\n");
    }
    else {
      inlist.nbase = PyList_Size(l_base);
      ibase = (int *) malloc(inlist.nbase*sizeof(int));
      for (i = 0; i < inlist.nbase; i++) {
        *(ibase+i) = (int) PyInt_AsLong(PyList_GetItem(l_base,i));
      }
      inlist.base = ibase;
    }
  }

  l_contour = PyList_GetItem($input,1);
  if (l_contour == Py_None) {
    inlist.ncontour = 0;
    inlist.contour = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_contour)) {
      printf("PlotId contour element must be None or a Python list\n");
    }
    else {
      inlist.ncontour = PyList_Size(l_contour);
      icontour = (int *) malloc(inlist.ncontour*sizeof(int));
      for (i = 0; i < inlist.ncontour; i++) {
        *(icontour+i) = (int) PyInt_AsLong(PyList_GetItem(l_contour,i));
      }
      inlist.contour = icontour;
    }
  }

  l_vector = PyList_GetItem($input,2);
  if (l_vector == Py_None) {
    inlist.nvector = 0;
    inlist.vector = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_vector)) {
      printf("PlotId vector element must be None or a Python list\n");
    }
    else {
      inlist.nvector = PyList_Size(l_vector);
      ivector = (int *) malloc(inlist.nvector*sizeof(int));
      for (i = 0; i < inlist.nvector; i++) {
        *(ivector+i) = (int) PyInt_AsLong(PyList_GetItem(l_vector,i));
      }
      inlist.vector = ivector;
    }
  }

  l_streamline = PyList_GetItem($input,3);
  if (l_streamline == Py_None) {
    inlist.nstreamline = 0;
    inlist.streamline = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_streamline)) {
      printf("PlotId streamline element must be None or a Python list\n");
    }
    else {
      inlist.nstreamline = PyList_Size(l_streamline);
      istreamline = (int *) malloc(inlist.nstreamline*sizeof(int));
      for (i = 0; i < inlist.nstreamline; i++) {
        *(istreamline+i) = (int) PyInt_AsLong(PyList_GetItem(l_streamline,i));
      }
      inlist.streamline = istreamline;
    }
  }

  l_map = PyList_GetItem($input,4);
  if (l_map == Py_None) {
    inlist.nmap = 0;
    inlist.map = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_map)) {
      printf("PlotId map element must be None or a Python list\n");
    }
    else {
      inlist.nmap = PyList_Size(l_map);
      imap = (int *) malloc(inlist.nmap*sizeof(int));
      for (i = 0; i < inlist.nmap; i++) {
        *(imap+i) = (int) PyInt_AsLong(PyList_GetItem(l_map,i));
      }
      inlist.map = imap;
    }
  }

  l_xy = PyList_GetItem($input,5);
  if (l_xy == Py_None) {
    inlist.nxy = 0;
    inlist.xy = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_xy)) {
      printf("PlotId xy element must be None or a Python list\n");
    }
    else {
      inlist.nxy = PyList_Size(l_xy);
      ixy = (int *) malloc(inlist.nxy*sizeof(int));
      for (i = 0; i < inlist.nxy; i++) {
        *(ixy+i) = (int) PyInt_AsLong(PyList_GetItem(l_xy,i));
      }
      inlist.xy = ixy;
    }
  }

  l_xydspec = PyList_GetItem($input,6);
  if (l_xydspec == Py_None) {
    inlist.nxydspec = 0;
    inlist.xydspec = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_xydspec)) {
      printf("PlotId xydspec element must be None or a Python list\n");
    }
    else {
      inlist.nxydspec = PyList_Size(l_xydspec);
      ixydspec = (int *) malloc(inlist.nxydspec*sizeof(int));
      for (i = 0; i < inlist.nxydspec; i++) {
        *(ixydspec+i) = (int) PyInt_AsLong(PyList_GetItem(l_xydspec,i));
      }
      inlist.xydspec = ixydspec;
    }
  }

  l_text = PyList_GetItem($input,7);
  if (l_text == Py_None) {
    inlist.ntext = 0;
    inlist.text = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_text)) {
      printf("PlotId text element must be None or a Python list\n");
    }
    else {
      inlist.ntext = PyList_Size(l_text);
      itext = (int *) malloc(inlist.ntext*sizeof(int));
      for (i = 0; i < inlist.ntext; i++) {
        *(itext+i) = (int) PyInt_AsLong(PyList_GetItem(l_text,i));
      }
      inlist.text = itext;
    }
  }

  l_primitive = PyList_GetItem($input,8);
  if (l_primitive == Py_None) {
    inlist.nprimitive = 0;
    inlist.primitive = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_primitive)) {
      printf("PlotId primitive element must be None or a Python list\n");
    }
    else {
      inlist.nprimitive = PyList_Size(l_primitive);
      iprimitive = (int *) malloc(inlist.nprimitive*sizeof(int));
      for (i = 0; i < inlist.nprimitive; i++) {
        *(iprimitive+i) = (int) PyInt_AsLong(PyList_GetItem(l_primitive,i));
      }
      inlist.primitive = iprimitive;
    }
  }

  l_cafield = PyList_GetItem($input,9);
  if (l_cafield == Py_None) {
    inlist.ncafield = 0;
    inlist.cafield = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_cafield)) {
      printf("PlotId cafield element must be None or a Python list\n");
    }
    else {
      inlist.ncafield = PyList_Size(l_cafield);
      icafield = (int *) malloc(inlist.ncafield*sizeof(int));
      for (i = 0; i < inlist.ncafield; i++) {
        *(icafield+i) = (int) PyInt_AsLong(PyList_GetItem(l_cafield,i));
      }
      inlist.cafield = icafield;
    }
  }

  l_sffield = PyList_GetItem($input,10);
  if (l_sffield == Py_None) {
    inlist.nsffield = 0;
    inlist.sffield = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_sffield)) {
      printf("PlotId sffield element must be None or a Python list\n");
    }
    else {
      inlist.nsffield = PyList_Size(l_sffield);
      isffield = (int *) malloc(inlist.nsffield*sizeof(int));
      for (i = 0; i < inlist.nsffield; i++) {
        *(isffield+i) = (int) PyInt_AsLong(PyList_GetItem(l_sffield,i));
      }
      inlist.sffield = isffield;
    }
  }

  l_vffield = PyList_GetItem($input,11);
  if (l_vffield == Py_None) {
    inlist.nvffield = 0;
    inlist.vffield = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_vffield)) {
      printf("PlotId vffield element must be None or a Python list\n");
    }
    else {
      inlist.nvffield = PyList_Size(l_vffield);
      ivffield = (int *) malloc(inlist.nvffield*sizeof(int));
      for (i = 0; i < inlist.nvffield; i++) {
        *(ivffield+i) = (int) PyInt_AsLong(PyList_GetItem(l_vffield,i));
      }
      inlist.vffield = ivffield;
    }
  }

  $1 = &inlist;
}

%typemap(out) NhlString *NhlGetStringArray {
  int i;
  PyObject *rlist;
  rlist = PyList_New(*arg3);
  for (i = 0; i < *arg3; i++) {
    PyList_SetItem(rlist,i,PyString_FromString((char *)result[i]));
  }
  $result = rlist;
}

%typemap(in) NhlString * {
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (NhlString *) malloc((size+1)*sizeof(char *));
    if (size == 0) {
      $1[0] = 0;
    }
    else {
      for (i = 0; i < size; i++) {
        PyObject *o = PyList_GetItem($input,i);
        if (PyString_Check(o)) {
          $1[i] = PyString_AsString(PyList_GetItem($input,i));
        }
        else {
          PyErr_SetString(PyExc_TypeError,"List must contain strings");
          free ($1);
          return NULL;
        }
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError,"Not a list");
    return NULL;
  }
}

//
// typemap to free memory that may have been allocated in processing
// Python sequences as C floats.
//
// %typemap (freearg) float *sequence_as_float {
//   free ( (float *) $1);
// }
//

//
// typemap to convert an NhlErrorTypes on output to a Python int.
//
// typedef int NhlErrorTypes;
%typemap(out) NhlErrorTypes {
  $result = PyInt_FromLong ((long) $1);
}

%typemap (argout) float *farray_float_out {
  int dims[1];
  PyObject *o;
  dims[0] = arg4;
  o = (PyObject *)PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,
                  (char *) arg6);
  resultobj = t_output_helper(resultobj,o);
}
%typemap(in,numinputs=0) float *farray_float_out (float tempx) %{
  $1 = &tempx;
%}

//
// Typemaps for gendat.
//
%typemap (argout) float *array_float_out {
  int dims[1];
  dims[0] = arg1;
  $result = (PyObject *) PyArray_FromDimsAndData(1,dims,PyArray_FLOAT,(char *) arg8);
}
%typemap(in,numinputs=0) float *array_float_out (float tempx) %{
  $1 = &tempx;
%}

%typemap (argout) int *err_out {
  PyObject *o;
  o = (PyObject *) PyInt_FromLong((long) *$1);
  $result = o;
}
%typemap(in,numinputs=0) int *err_out (int tempx) %{
  $1 = &tempx;
%}

%typemap (in) res_names *rcs_names {
  PyObject *key,*value;
  int pos=0,rnumber,count;
  res_names trname;
  char **trnames;
  
  
  if (PyDict_Check($input)) {
    count = 0;
    trname.number = PyDict_Size($input);
    trnames = (char **) malloc(trname.number*sizeof(char *));
    while (PyDict_Next($input, &pos, &key, &value)) {
      trnames[count] = PyString_AsString(key);
      count++;
    }
    trname.strings = trnames;
  }
  else {
    printf("Internal error: Resource name lists must be dictionaries\n");
  }
  $1 = (res_names *) &trname;
}

//
//  I need to set a value for arg8 for gendat based on the value of arg1.
//  The following is the only way I could think to do this. I
//  need to insert a value for arg8 after the ParseTuple, but
//  before the call to gendat.
//
%typemap(in) int argi %{
  $1 = (int) PyInt_AsLong($input);
  arg8 = (float *) malloc(arg1*sizeof(float));
%}

%typemap(in) ResInfo *rlist {
  int i,pos=0,list_type,list_len,count;
  PyObject *key,*value;
  PyArrayObject *arr;
  char **strings;
  double *dvals;
  int *ivals,array_type,rlist,ndims,*len_dims;
  long *lvals;
  static ResInfo trname;
  char **trnames;

/*
 *  Clear the resource list.
 */
  rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(rlist);

/*
 *  Check on the type of the argument - it must be a dictionary.
 */
  if (PyDict_Check($input)) {
    count = 0;
    trname.nstrings = PyDict_Size($input);
    trnames = (char **) malloc(trname.nstrings*sizeof(char *));
    pos = 0;
/*
 *  Loop over the keyword/value pairs in the dictionary.
 *  The values must be one of: tuple, int, float, long,
 *  list, string, or array.
 */
    while (PyDict_Next($input, &pos, &key, &value)) {
      trnames[count] = PyString_AsString(key);
      count++;

/*
 *  value is a tuple.
 */
      if (PyTuple_Check(value)) {
/*
 *  Lists and tuples are not allowed as items in a tuple value.
 */
        if (PyList_Check(PyTuple_GetItem(value,0)) ||
           PyTuple_Check(PyTuple_GetItem(value,0))) {
          printf("Tuple vlaues are not allowed to have list or tuple items.\n");
          return NULL;
        }
        list_len = PyTuple_Size(value);
/*
 *  Determine if the tuple is a tuple of strings, ints, or floats.
 *  
 *    list_type = 2 (int)
 *              = 0 (string)
 *              = 1 (float)
 */
        list_type = 2;
        if (PyString_Check(PyTuple_GetItem(value,0))) {
/*
 *  Check that all items in the tuple are strings.
 */
          for (i = 0; i < list_len ; i++) {
            if (!PyString_Check(PyTuple_GetItem(value,i))) {
              printf("All items in the tuple value for resource %s must be strings\n",PyString_AsString(key));
            return NULL;
            }
          }
          list_type = 0;
        }
        else {
/*
 *  If the items in the tuple value are not strings, then
 *  they must all be ints or floats.
 */
          for (i = 0; i < list_len ; i++) {
            if ( (!PyFloat_Check(PyTuple_GetItem(value,i))) &&
                 (!PyInt_Check(PyTuple_GetItem(value,i))) ) {
              printf("All items in the tuple value for resource %s must be ints or floats.\n",PyString_AsString(key));
              return NULL;
              break;
            }
          }
/*
 *  Check to see if the tuple has all ints and, if not, type it as
 *  a tuple of floats.
 */
          for (i = 0; i < list_len ; i++) {
            if (PyFloat_Check(PyTuple_GetItem(value,i))) {
              list_type = 1;
              break;
            }
          }
        }

/*
 *  Make the appropriate NhlRLSet calls based on the type of
 *  tuple elements.
 */
        switch (list_type) {
          case 0:
            strings = (char **) malloc(list_len*sizeof(char *));
            for (i = 0; i < list_len ; i++) {
              strings[i] = PyString_AsString(PyTuple_GetItem(value,i));
            }
            NhlRLSetStringArray(rlist,PyString_AsString(key),strings,list_len);
            break;
          case 1:
            dvals = (double *) malloc(list_len*sizeof(double));
            for (i = 0; i < list_len ; i++) {
              dvals[i] = PyFloat_AsDouble(PyTuple_GetItem(value,i));
            }
            NhlRLSetDoubleArray(rlist,PyString_AsString(key),dvals,list_len);
            break;
          case 2:
            ivals = (int *) malloc(list_len*sizeof(int));
            for (i = 0; i < list_len ; i++) {
              ivals[i] = (int) PyInt_AsLong(PyTuple_GetItem(value,i));
            }
            NhlRLSetIntegerArray(rlist,PyString_AsString(key),ivals,list_len);
            break;
        }
      }
/*
 *  value is a list.
 */
      else if (PyList_Check(value)) {
/*
 *  Lists and tuples are not allowed as items in a list value.
 */
        if (PyList_Check(PyList_GetItem(value,0)) ||
           PyList_Check(PyList_GetItem(value,0))) {
          printf("Use Numeric arrays for multiple dimension arrays.\n");
          return NULL;
        }
        list_len = PyList_Size(value);
/*
 *  Determine if the list is a list of strings, ints, or floats.
 *  
 *    list_type = 2 (int)
 *              = 0 (string)
 *              = 1 (float)
 */
        list_type = 2;
        if (PyString_Check(PyList_GetItem(value,0))) {
/*
 *  Check that all items in the list are strings.
 */
          for (i = 0; i < list_len ; i++) {
            if (!PyString_Check(PyList_GetItem(value,i))) {
              printf("All items in the list value for resource %s must be strings\n",PyString_AsString(key));
              return NULL;
              break;
            }
          }
          list_type = 0;
        }
        else {
/*
 *  If the items in the list value are not strings, then
 *  they must all be ints or floats.
 */
          for (i = 0; i < list_len ; i++) {
            if ( (!PyFloat_Check(PyList_GetItem(value,i))) &&
                 (!PyInt_Check(PyList_GetItem(value,i))) ) {
              printf("All items in the list value for resource %s must be ints or floats.\n",PyString_AsString(key));
              return NULL;
            }
          }
/*
 *  Check to see if the list has all ints and, if not, type it as
 *  a list of floats.
 */
          for (i = 0; i < list_len ; i++) {
            if (PyFloat_Check(PyList_GetItem(value,i))) {
              list_type = 1;
            }
          }
        }
        switch (list_type) {
          case 0:
            strings = (char **) malloc(list_len*sizeof(char *));
            for (i = 0; i < list_len ; i++) {
              strings[i] = PyString_AsString(PyList_GetItem(value,i));
            }
            NhlRLSetStringArray(rlist,PyString_AsString(key),strings,list_len);
            break;
          case 1:
            dvals = (double *) malloc(list_len*sizeof(double));
            for (i = 0; i < list_len ; i++) {
              dvals[i] = PyFloat_AsDouble(PyList_GetItem(value,i));
            }
            NhlRLSetDoubleArray(rlist,PyString_AsString(key),dvals,list_len);
            break;
          case 2:
            ivals = (int *) malloc(list_len*sizeof(int));
            for (i = 0; i < list_len ; i++) {
              ivals[i] = (int) PyInt_AsLong(PyList_GetItem(value,i));
            }
            NhlRLSetIntegerArray(rlist,PyString_AsString(key),ivals,list_len);
            break;
        }
      }
/*
 *  value is an int.
 */
      else if (PyInt_Check(value)) {
        NhlRLSetInteger(rlist,PyString_AsString(key),(int) PyInt_AsLong(value));
      }
/*
 *  value is a float.
 */
      else if (PyFloat_Check(value)) {
        NhlRLSetDouble(rlist,PyString_AsString(key),PyFloat_AsDouble(value));
      }
/*
 *  value is a long.
 */
      else if (PyLong_Check(value)) {
        NhlRLSetInteger(rlist,PyString_AsString(key),(int) PyInt_AsLong(value));
      }
/*
 *  value is a string
 */
      else if (PyString_Check(value)) {
        NhlRLSetString(rlist,PyString_AsString(key),PyString_AsString(value));
      }
/*
 *  value is an array.
 */
      else if (PyArray_Check(value)) {
        
        array_type = (int) ((PyArrayObject *)value)->descr->type_num;
/*
 *  Process the legal array types.
 */
        if (array_type == PyArray_LONG || array_type == PyArray_INT) {
          arr = (PyArrayObject *) PyArray_ContiguousFromObject \
                     ((PyObject *) value,PyArray_LONG,0,0);
          lvals = (long *)arr->data;
          ndims = arr->nd;
          len_dims = arr->dimensions;
          NhlRLSetMDLongArray(rlist,PyString_AsString(key),lvals,ndims,len_dims);
        }
        else if (array_type == PyArray_FLOAT || array_type == PyArray_DOUBLE) {
          arr = (PyArrayObject *) PyArray_ContiguousFromObject \
                 ((PyObject *) value,PyArray_DOUBLE,0,0);
          dvals = (double *)arr->data;
          ndims = arr->nd;
          len_dims = arr->dimensions;
          NhlRLSetMDDoubleArray(rlist,PyString_AsString(key),dvals,ndims,len_dims);
        }
        else {
          printf(
            "Numeric arrays must be of type Int, Int32, Float, Float0, Float32, or Float64.\n");
            return NULL;
        }
      }
      else {
        printf("  value for keyword %s is invalid.\n",PyString_AsString(key));
        return NULL;
      }
    }
    trname.strings = trnames;
  }
  else {
    printf("Resource lists must be dictionaries\n");
  }
  trname.id = rlist;
  $1 = (ResInfo *) &trname;
}

%typemap(in) int res_id {
  int i,pos=0,list_type,list_len,count;
  PyObject *key,*value;
  PyArrayObject *arr;
  char **strings;
  double *dvals;
  int *ivals,array_type,rlist,ndims,*len_dims;
  long *lvals;
  ResInfo trname;
  char **trnames;

/*
 *  Clear the resource list.
 */
  rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(rlist);

/*
 *  Check on the type of the argument - it must be a dictionary.
 */
  if (PyDict_Check($input)) {
    count = 0;
    trname.nstrings = PyDict_Size($input);
    trnames = (char **) malloc(trname.nstrings*sizeof(char *));
    pos = 0;
/*
 *  Loop over the keyword/value pairs in the dictionary.
 *  The values must be one of: tuple, int, float, long,
 *  list, string, or array.
 */
    while (PyDict_Next($input, &pos, &key, &value)) {
      trnames[count] = PyString_AsString(key);
      count++;

/*
 *  value is a tuple.
 */
      if (PyTuple_Check(value)) {
/*
 *  Lists and tuples are not allowed as items in a tuple value.
 */
        if (PyList_Check(PyTuple_GetItem(value,0)) ||
           PyTuple_Check(PyTuple_GetItem(value,0))) {
          printf("Tuple vlaues are not allowed to have list or tuple items.\n");
          return NULL;
        }
        list_len = PyTuple_Size(value);
/*
 *  Determine if the tuple is a tuple of strings, ints, or floats.
 *  
 *    list_type = 2 (int)
 *              = 0 (string)
 *              = 1 (float)
 */
        list_type = 2;
        if (PyString_Check(PyTuple_GetItem(value,0))) {
/*
 *  Check that all items in the tuple are strings.
 */
          for (i = 0; i < list_len ; i++) {
            if (!PyString_Check(PyTuple_GetItem(value,i))) {
              printf("All items in the tuple value for resource %s must be strings\n",PyString_AsString(key));
            return NULL;
            }
          }
          list_type = 0;
        }
        else {
/*
 *  If the items in the tuple value are not strings, then
 *  they must all be ints or floats.
 */
          for (i = 0; i < list_len ; i++) {
            if ( (!PyFloat_Check(PyTuple_GetItem(value,i))) &&
                 (!PyInt_Check(PyTuple_GetItem(value,i))) ) {
              printf("All items in the tuple value for resource %s must be ints or floats.\n",PyString_AsString(key));
              return NULL;
              break;
            }
          }
/*
 *  Check to see if the tuple has all ints and, if not, type it as
 *  a tuple of floats.
 */
          for (i = 0; i < list_len ; i++) {
            if (PyFloat_Check(PyTuple_GetItem(value,i))) {
              list_type = 1;
              break;
            }
          }
        }

/*
 *  Make the appropriate NhlRLSet calls based on the type of
 *  tuple elements.
 */
        switch (list_type) {
          case 0:
            strings = (char **) malloc(list_len*sizeof(char *));
            for (i = 0; i < list_len ; i++) {
              strings[i] = PyString_AsString(PyTuple_GetItem(value,i));
            }
            NhlRLSetStringArray(rlist,PyString_AsString(key),strings,list_len);
            break;
          case 1:
            dvals = (double *) malloc(list_len*sizeof(double));
            for (i = 0; i < list_len ; i++) {
              dvals[i] = PyFloat_AsDouble(PyTuple_GetItem(value,i));
            }
            NhlRLSetDoubleArray(rlist,PyString_AsString(key),dvals,list_len);
            break;
          case 2:
            ivals = (int *) malloc(list_len*sizeof(int));
            for (i = 0; i < list_len ; i++) {
              ivals[i] = (int) PyInt_AsLong(PyTuple_GetItem(value,i));
            }
            NhlRLSetIntegerArray(rlist,PyString_AsString(key),ivals,list_len);
            break;
        }
      }
/*
 *  value is a list.
 */
      else if (PyList_Check(value)) {
/*
 *  Lists and tuples are not allowed as items in a list value.
 */
        if (PyList_Check(PyList_GetItem(value,0)) ||
           PyList_Check(PyList_GetItem(value,0))) {
          printf("Use Numeric arrays for multiple dimension arrays.\n");
          return NULL;
        }
        list_len = PyList_Size(value);
/*
 *  Determine if the list is a list of strings, ints, or floats.
 *  
 *    list_type = 2 (int)
 *              = 0 (string)
 *              = 1 (float)
 */
        list_type = 2;
        if (PyString_Check(PyList_GetItem(value,0))) {
/*
 *  Check that all items in the list are strings.
 */
          for (i = 0; i < list_len ; i++) {
            if (!PyString_Check(PyList_GetItem(value,i))) {
              printf("All items in the list value for resource %s must be strings\n",PyString_AsString(key));
              return NULL;
              break;
            }
          }
          list_type = 0;
        }
        else {
/*
 *  If the items in the list value are not strings, then
 *  they must all be ints or floats.
 */
          for (i = 0; i < list_len ; i++) {
            if ( (!PyFloat_Check(PyList_GetItem(value,i))) &&
                 (!PyInt_Check(PyList_GetItem(value,i))) ) {
              printf("All items in the list value for resource %s must be ints or floats.\n",PyString_AsString(key));
              return NULL;
            }
          }
/*
 *  Check to see if the list has all ints and, if not, type it as
 *  a list of floats.
 */
          for (i = 0; i < list_len ; i++) {
            if (PyFloat_Check(PyList_GetItem(value,i))) {
              list_type = 1;
            }
          }
        }
        switch (list_type) {
          case 0:
            strings = (char **) malloc(list_len*sizeof(char *));
            for (i = 0; i < list_len ; i++) {
              strings[i] = PyString_AsString(PyList_GetItem(value,i));
            }
            NhlRLSetStringArray(rlist,PyString_AsString(key),strings,list_len);
            break;
          case 1:
            dvals = (double *) malloc(list_len*sizeof(double));
            for (i = 0; i < list_len ; i++) {
              dvals[i] = PyFloat_AsDouble(PyList_GetItem(value,i));
            }
            NhlRLSetDoubleArray(rlist,PyString_AsString(key),dvals,list_len);
            break;
          case 2:
            ivals = (int *) malloc(list_len*sizeof(int));
            for (i = 0; i < list_len ; i++) {
              ivals[i] = (int) PyInt_AsLong(PyList_GetItem(value,i));
            }
            NhlRLSetIntegerArray(rlist,PyString_AsString(key),ivals,list_len);
            break;
        }
      }
/*
 *  value is an int.
 */
      else if (PyInt_Check(value)) {
        NhlRLSetInteger(rlist,PyString_AsString(key),(int) PyInt_AsLong(value));
      }
/*
 *  value is a float.
 */
      else if (PyFloat_Check(value)) {
        NhlRLSetDouble(rlist,PyString_AsString(key),PyFloat_AsDouble(value));
      }
/*
 *  value is a long.
 */
      else if (PyLong_Check(value)) {
        NhlRLSetInteger(rlist,PyString_AsString(key),(int) PyInt_AsLong(value));
      }
/*
 *  value is a string
 */
      else if (PyString_Check(value)) {
        NhlRLSetString(rlist,PyString_AsString(key),PyString_AsString(value));
      }
/*
 *  value is an array.
 */
      else if (PyArray_Check(value)) {
        
        array_type = (int) ((PyArrayObject *)value)->descr->type_num;
/*
 *  Process the legal array types.
 */
        if (array_type == PyArray_LONG || array_type == PyArray_INT) {
          arr = (PyArrayObject *) PyArray_ContiguousFromObject \
                ((PyObject *) value,PyArray_LONG,0,0);
          lvals = (long *)arr->data;
          ndims = arr->nd;
          len_dims = arr->dimensions;
          NhlRLSetMDLongArray(rlist,PyString_AsString(key),lvals,ndims,len_dims);
        }
        else if (array_type == PyArray_FLOAT || array_type == PyArray_DOUBLE) {
          arr = (PyArrayObject *) PyArray_ContiguousFromObject \
                 ((PyObject *) value,PyArray_DOUBLE,0,0);
          dvals = (double *)arr->data;
          ndims = arr->nd;
          len_dims = arr->dimensions;
          NhlRLSetMDDoubleArray(rlist,PyString_AsString(key),dvals,ndims,len_dims);
        }
        else {
          printf(
            "Numeric arrays must be of type int, int32, float, float0, float32, or float64.\n");
            return NULL;
        }
      }
      else {
        printf("  value for keyword %s is invalid.\n",PyString_AsString(key));
        return NULL;
      }
    }
    trname.strings = trnames;
  }
  else {
    printf("Resource lists must be dictionaries\n");
  }
  trname.id = rlist;
  $1 = rlist;
}

typedef char *NhlString;
typedef int   NhlBoolean;
typedef int NhlFillIndexFullEnum;
typedef int NhlFillIndex;


enum NhlRLType {NhlSETRL, NhlGETRL};
enum NhlErrorTypes {NhlFATAL, NhlWARNING, NhlINFO, NhlNOERROR};
enum NhlcnLevelUseMode {NhlNOLINE, NhlLINEONLY, NhlLABELONLY, NhlLINEANDLABEL};
enum NhlPolyType {NhlPOLYLINE, NhlPOLYMARKER, NhlPOLYGON};

%constant NhlDEFAULT_APP = 0;
%constant False = 0;
%constant True = 1;
%constant NhlAUTOMATIC = 0;
%constant NhlMANUAL = 1;
%constant NhlEXPLICIT = 2;
%constant NhlLOG = 0;
%constant NhlLINEAR = 1;
%constant NhlIRREGULAR = 2;
%constant NhlGEOGRAPHIC = 3;
%constant NhlTIME = 4;

%constant NhlEUNKNOWN    = 1000;
%constant NhlENODATA     = 1101;
%constant NhlECONSTFIELD = 1102;
%constant NhlEZEROFIELD  = 1103;
%constant NhlEZEROSPAN   = 1104;

extern const char *_NGGetNCARGEnv(const char *);
extern void NhlInitialize();
extern void NhlClose();
extern void NhlRLClear(int);
extern NhlErrorTypes NhlSetValues (int, int res_id);
extern NhlErrorTypes NhlRLSetString (int, NhlString, NhlString);
extern NhlErrorTypes NhlRLSetFloat (int, NhlString, float);
extern NhlErrorTypes NhlRLSetDouble (int, NhlString, double);
extern NhlErrorTypes NhlRLSetInteger (int, NhlString, int);
extern NhlErrorTypes NhlNDCPolyline (int, int, float *sequence_as_float, float *sequence_as_float, int);
extern NhlErrorTypes NhlNDCPolymarker (int, int, float *sequence_as_float, float *sequence_as_float, int);
extern NhlErrorTypes NhlNDCPolygon (int, int, float *sequence_as_float, float *sequence_as_float, int);
extern NhlErrorTypes NhlDataPolyline (int, int, float *sequence_as_float,
  float *sequence_as_float, int);
extern NhlErrorTypes NhlDataPolymarker (int, int, float *sequence_as_float,
  float *sequence_as_float, int);
extern NhlErrorTypes NhlDataPolygon (int, int, float *sequence_as_float,
  float *sequence_as_float, int);
extern NhlErrorTypes NhlDraw (int);
extern NhlErrorTypes NhlFreeColor (int, int);
extern int NhlGetGksCi (int, int);
extern int NhlGetWorkspaceObjectId();
extern NhlBoolean NhlIsAllocatedColor(int, int);
extern NhlBoolean NhlIsApp(int);
extern NhlBoolean NhlIsDataComm(int);
extern NhlBoolean NhlIsDataItem(int);
extern NhlBoolean NhlIsDataSpec(int);
extern NhlBoolean NhlRLIsSet(int,NhlString);
extern void NhlRLUnSet(int,NhlString);
extern NhlBoolean NhlIsTransform(int);
extern NhlBoolean NhlIsView(int);
extern NhlBoolean NhlIsWorkstation(int);
extern const char *NhlName(int);
extern int NhlNewColor(int, float, float, float);
extern int NhlNewDashPattern(int, NhlString);
extern int NhlNewMarker(int, NhlString, int, float, float, float, float, float);
extern NhlErrorTypes NhlSetColor(int, int, float, float, float);
extern NhlErrorTypes NhlUpdateData(int);
extern NhlErrorTypes NhlUpdateWorkstation(int);
extern void NhlOpen();

extern NhlErrorTypes NhlCreate (int *OUTPUT, const char *, NhlClass, int, int);
extern int NhlRLCreate(NhlRLType);
extern NhlErrorTypes NhlFrame (int);
extern NhlErrorTypes NhlDestroy (int);
extern NhlErrorTypes NhlRLSetMDIntegerArray(int, char *, int *sequence_as_int, int, int *sequence_as_int);
extern NhlErrorTypes NhlRLSetMDDoubleArray(int, char *, double *sequence_as_double, int, int *sequence_as_int);
extern NhlErrorTypes NhlRLSetMDFloatArray(int, char *, float *sequence_as_float, int, int *sequence_as_int);
extern NhlErrorTypes NhlRLSetFloatArray(int, char *, float *sequence_as_float, int);
extern NhlErrorTypes NhlRLSetIntegerArray(int, char *, int *sequence_as_int, int);
extern NhlErrorTypes NhlRLSetStringArray(int, NhlString, NhlString *, int);
extern NhlErrorTypes NhlGetValues(int, int res_id);
extern float NhlGetFloat(int oid, char *name);
extern float *NhlGetFloatArray(int oid, char *name, int *numberf);
extern int NhlGetInteger(int oid, char *name);
extern int *NhlGetIntegerArray(int oid, char *name, int *numberi);
extern float NhlGetDouble(int oid, char *name);
extern double *NhlGetDoubleArray(int oid, char *name, int *numberd);
extern NhlErrorTypes NhlAddOverlay(int,int,int);
extern NhlErrorTypes NhlClearWorkstation(int);
extern NhlErrorTypes NhlRemoveAnnotation(int, int);
extern int NhlAddAnnotation(int,int);
extern int NhlAppGetDefaultParentId();
extern int NhlGetParentWorkstation(int);
extern const char *NhlClassName(int);
extern NhlString NhlGetString(int, NhlString);
extern int NhlAddData(int, NhlString, int);
extern NhlErrorTypes NhlRemoveData(int, NhlString, int);
extern NhlErrorTypes NhlRemoveOverlay(int, int, NhlBoolean);
extern NhlString *NhlGetStringArray(int oid, char *name, int *numbers);
extern void NhlRLDestroy(int);
extern int NhlGetNamedColorIndex(int, const char *);
extern NhlErrorTypes NhlGetBB(int, NhlBoundingBox *);
extern NhlErrorTypes NhlChangeWorkstation(int, int);
extern NhlErrorTypes NhlPGetBB(int, float *OUTPUT, float *OUTPUT, float *OUTPUT,
                                float *OUTPUT);
extern PyObject *NhlPNDCToData(int, float *sequence_as_float, 
                               float *sequence_as_float, int, float,
                               float, int, int);
extern PyObject *NhlPDataToNDC(int, float *sequence_as_float, 
                               float *sequence_as_float, int, float,
                               float, int, int);
extern PyObject *NhlGetMDFloatArray(int, char *);
extern PyObject *NhlGetMDDoubleArray(int, char *);
extern PyObject *NhlGetMDIntegerArray(int, char *);
extern NhlClass NhlPAppClass ();
extern NhlClass NhlPNcgmWorkstationClass ();
extern NhlClass NhlPXWorkstationClass ();
extern NhlClass NhlPPSWorkstationClass ();
extern NhlClass NhlPPDFWorkstationClass ();
extern NhlClass NhlPLogLinPlotClass();
extern NhlClass NhlPGraphicStyleClass();
extern NhlClass NhlPScalarFieldClass ();
extern NhlClass NhlPContourPlotClass();
extern NhlClass NhlPtextItemClass();
extern NhlClass NhlPscalarFieldClass();
extern NhlClass NhlPmapPlotClass();
extern NhlClass NhlPcoordArraysClass();
extern NhlClass NhlPxyPlotClass();
extern NhlClass NhlPtickMarkClass();
extern NhlClass NhlPtitleClass();
extern NhlClass NhlPlabelBarClass();
extern NhlClass NhlPlegendClass();
extern NhlClass NhlPvectorFieldClass();
extern NhlClass NhlPvectorPlotClass();
extern NhlClass NhlPstreamlinePlotClass();

extern const char *NGGetNCARGEnv(const char *);

extern void set_PCMP04(int, float);
extern void gendat (int argi, int, int, int, int, float, float, float *array_float_out);
extern void gactivate_ws(int);
extern void gdeactivate_ws(int);
extern void bndary();
extern void c_plotif(float, float, int);
extern void c_cpseti(NhlString, int);
extern void c_cpsetr(NhlString, float);
extern void c_pcseti(NhlString, int);
extern void c_pcsetr(NhlString, float);
extern void c_set(float, float, float, float, float, float, float, float, int);
extern void c_cprect(float *sequence_as_float, int, int, int, float *sequence_as_float, int, int *sequence_as_int, int);
extern void c_cpcldr(float *sequence_as_float, float *sequence_as_float, int *sequence_as_int);
extern void c_plchhq(float, float, NhlString, float, float, float);

extern int open_wks_wrap(const char *, const char *, ResInfo *rlist,
                         ResInfo *rlist, nglRes *rlist);

extern nglPlotId labelbar_ndc_wrap(int, int, NhlString *, int, void
                         *sequence_as_void, void *sequence_as_void,
                         const char *, const char *, 
                         ResInfo *rlist, nglRes *rlist);

extern nglPlotId legend_ndc_wrap(int, int, NhlString *, int, void
                                 *sequence_as_void, void *sequence_as_void,
                                 const char *, const char *, 
                                 ResInfo *rlist, nglRes *rlist);

extern nglPlotId contour_wrap(int, void *sequence_as_void, 
                            const char *, int, int,
                            int, void *, const char *, int, void *, 
                            const char *,
                            int, void *, ResInfo *rlist, ResInfo *rlist, 
                            ResInfo *rlist, nglRes *rlist);
extern nglPlotId map_wrap(int, ResInfo *rlist, nglRes *rlist);
extern nglPlotId contour_map_wrap(int, void *sequence_as_void, 
                            const char *, int, int,
                            int, void *, const char *, int, void *, 
                            const char *, int,
                            void *, ResInfo *rlist, ResInfo *rlist, ResInfo *rlist,
                            nglRes *rlist);
extern nglPlotId xy_wrap(int, void *sequence_as_void, void *sequence_as_void, 
                const char *, const char *, int, int *sequence_as_int,
                int, int *sequence_as_int, int, int, void *, void *,
                ResInfo *rlist, ResInfo *rlist, ResInfo *rlist, nglRes *rlist);
extern nglPlotId y_wrap(int, void *sequence_as_void, 
                const char *, int, int *sequence_as_int, int, void *,
                ResInfo *rlist, ResInfo *rlist, ResInfo *rlist, nglRes *rlist);
extern nglPlotId vector_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, const char *,
                           const char *, int, int, int, void *, 
                           const char *, int, void *, const char *, int, int,
                           void *, void *, ResInfo *rlist, ResInfo *rlist,
                           ResInfo *rlist, nglRes *rlist);
extern nglPlotId vector_map_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, const char *, 
                           const char *, int, int, int, void *, 
                           const char *, int, void *, const char *, int, int, 
                           void *, void *, ResInfo *rlist, ResInfo *rlist, 
                           ResInfo *rlist, nglRes *rlist);
extern nglPlotId vector_scalar_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void,
                           void *sequence_as_void, const char *, 
                           const char *, const char *, int, int, int, void *, 
                           const char *, int, void *, const char *, int, int, 
                           int, void *, void *, void *, 
                           ResInfo *rlist, ResInfo *rlist, ResInfo *rlist,
                           ResInfo *rlist, nglRes *rlist);
extern nglPlotId vector_scalar_map_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, 
                           void *sequence_as_void, const char *, 
                           const char *, const char *, int, int, int, void *,
                           const char *, int, void *, const char *, int, int,
                           int, void *, void *, void *,
                           ResInfo *rlist, ResInfo *rlist, 
                           ResInfo *rlist, ResInfo *rlist, nglRes *rlist);
extern nglPlotId streamline_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, const char *,
                           const char *, int, int, int, void *,
                           const char *, int, void *, const char *, 
                           int, int,
                           void *, void *, ResInfo *rlist, ResInfo *rlist, 
                           ResInfo *rlist, nglRes *rlist);
extern nglPlotId streamline_map_wrap(int, void *sequence_as_void, 
                            void *sequence_as_void, const char *,
                            const char *, int, int, int, void *,
                            const char *, int, void *, const char *, 
                            int, int,
                            void *, void *, ResInfo *rlist, ResInfo *rlist, 
                            ResInfo *rlist, nglRes *rlist);
extern nglPlotId text_ndc_wrap(int, NhlString, void *sequence_as_void,
                             void *sequence_as_void, const char *,
                             const char *, ResInfo *rlist, nglRes *rlist);
extern nglPlotId text_wrap(int, nglPlotId *plot, NhlString, void *sequence_as_void,
                             void *sequence_as_void, const char *,
                             const char *, ResInfo *rlist, nglRes *rlist);
extern nglPlotId add_text_wrap(int, nglPlotId *plot, NhlString, void *sequence_as_void,
                             void *sequence_as_void, const char *,
                             const char *, ResInfo *rlist, ResInfo *rlist,
                             nglRes *rlist);

extern void maximize_plots(int, nglPlotId *plot, int, int, nglRes *rlist);

extern void poly_wrap(int, nglPlotId *plot, void *sequence_as_void, 
                       void *sequence_as_void, const char *type_x,
                       const char *type_y, int, int, int, void *, void*,
                       NhlPolyType, ResInfo *rlist, nglRes *rlist);
extern nglPlotId add_poly_wrap(int, nglPlotId *plot, void *sequence_as_void,                       void *sequence_as_void, const char *type_x,
                       const char *type_y, int, int, int, void *, void*,
                       NhlPolyType, ResInfo *rlist, nglRes *rlist);
void panel_wrap(int, nglPlotId *plot_seq, int, int *sequence_as_int, int, 
                 ResInfo *rlist, ResInfo *rlist, nglRes *rlist);

extern PyObject *mapgci(float, float, float, float, int);
extern PyObject *dcapethermo(double *sequence_as_double, double *sequence_as_double, int, double, int, double);

extern void draw_colormap_wrap(int);
extern void natgridc(int, float *sequence_as_float, float *sequence_as_float,
                       float *sequence_as_float, int, int, 
                       float *sequence_as_float, float *sequence_as_float,
                       int *OUTPUT, 
                       int nxir, int nyir, float *p_array_float_out[]);
extern PyObject *ftcurvc(int, float *sequence_as_float, 
                         float *sequence_as_float,
                         int m, float *sequence_as_float);
extern PyObject *ftcurvpc(int, float *sequence_as_float, 
                         float *sequence_as_float, float,
                         int m, float *sequence_as_float);
extern PyObject *ftcurvpic(float, float, float, int, 
                           float *sequence_as_float, 
                           float *sequence_as_float);
extern void c_rgbhls(float, float, float, float *OUTPUT, float *OUTPUT, float *OUTPUT);
extern void c_hlsrgb(float, float, float, float *OUTPUT, float *OUTPUT, float *OUTPUT);
extern void c_rgbhsv(float, float, float, float *OUTPUT, float *OUTPUT, float *OUTPUT);
extern void c_hsvrgb(float, float, float, float *OUTPUT, float *OUTPUT, float *OUTPUT);
extern void c_rgbyiq(float, float, float, float *OUTPUT, float *OUTPUT, float *OUTPUT);
extern void c_yiqrgb(float, float, float, float *OUTPUT, float *OUTPUT, float *OUTPUT);

extern void c_wmbarbp(int, float, float, float, float);
extern void c_wmsetip(NhlString,int);
extern void c_wmsetrp(NhlString,float);
extern void c_wmsetcp(NhlString,NhlString);
extern int  c_wmgetip(NhlString);
extern float c_wmgetrp(NhlString);
extern NhlString c_wmgetcp(NhlString);

extern void c_nnseti(NhlString, int);
extern void c_nnsetrd(NhlString, double);
extern void c_nnsetc(NhlString, NhlString);
extern void c_nngeti(NhlString, int *OUTPUT);
extern void c_nngetrd(NhlString, double *OUTPUT);
extern NhlString c_nngetcp(NhlString);

extern c_mapgci(float, float, float, float, int, float *, float*);
extern double c_dgcdist(double, double, double, double, int);
extern double c_dcapethermo(double *, double *, int, double, int, 
                            double **, double, int *, int *, int *);
extern void c_dptlclskewt(double, double, double, 
                                 double *OUTPUT, double *OUTPUT);
extern double c_dtmrskewt(double, double);
extern double c_dtdaskewt(double, double);
extern double c_dsatlftskewt(double, double);
extern double c_dshowalskewt(double *sequence_as_double, 
                             double *sequence_as_double, 
                             double *sequence_as_double, int);
extern double c_dpwskewt(double *sequence_as_double, 
                             double *sequence_as_double, int);

extern void *pvoid();
extern void set_nglRes_i(int, int);
extern int get_nglRes_i(int);
extern void set_nglRes_f(int, float);
extern float get_nglRes_f(int);
extern void set_nglRes_c(int, NhlString *);
extern NhlString *get_nglRes_c(int);
extern void set_nglRes_s(int, NhlString);
extern NhlString get_nglRes_s(int);

extern NhlErrorTypes NglGaus_p(int num, int nxir, int nyir, double *p_array_double_out[]);

extern void NglVinth2p (double *sequence_as_double, 
                 int nxir, int nyir, int nzir, double *p_array3_double_out[], 
                 double *sequence_as_double,
                 double *sequence_as_double, double, 
                 double *sequence_as_double, double *sequence_as_double, 
                 int, int, double *sequence_as_double, double,
                 int, int, int);

%newobject _NGGetNCARGEnv(const char *);
%newobject  NhlSetValues (int, int res_id);
%newobject  NhlRLSetString (int, NhlString, NhlString);
%newobject  NhlRLSetFloat (int, NhlString, float);
%newobject  NhlRLSetDouble (int, NhlString, double);
%newobject  NhlRLSetInteger (int, NhlString, int);
%newobject  NhlNDCPolyline (int, int, float *sequence_as_float, float *sequence_as_float, int);
%newobject  NhlNDCPolymarker (int, int, float *sequence_as_float, float *sequence_as_float, int);
%newobject  NhlNDCPolygon (int, int, float *sequence_as_float, float *sequence_as_float, int);
%newobject  NhlDataPolyline (int, int, float *sequence_as_float,
  float *sequence_as_float, int);
%newobject  NhlDataPolymarker (int, int, float *sequence_as_float,
  float *sequence_as_float, int);
%newobject  NhlDataPolygon (int, int, float *sequence_as_float,
  float *sequence_as_float, int);
%newobject  NhlDraw (int);
%newobject  NhlFreeColor (int, int);
%newobject  NhlGetGksCi (int, int);
%newobject  NhlGetWorkspaceObjectId();
%newobject  NhlIsAllocatedColor(int, int);
%newobject  NhlIsApp(int);
%newobject  NhlIsDataComm(int);
%newobject  NhlIsDataItem(int);
%newobject  NhlIsDataSpec(int);
%newobject  NhlRLIsSet(int,NhlString);
%newobject  NhlRLUnSet(int,NhlString);
%newobject  NhlIsTransform(int);
%newobject  NhlIsView(int);
%newobject  NhlIsWorkstation(int);
%newobject  NhlName(int);

%newobject  NhlNewColor(int, float, float, float);
%newobject  NhlNewDashPattern(int, NhlString);
%newobject  NhlNewMarker(int, NhlString, int, float, float, float,
                         float, float);
%newobject  NhlSetColor(int, int, float, float, float);
%newobject  NhlUpdateData(int);
%newobject  NhlUpdateWorkstation(int);

%newobject  NhlCreate (int *OUTPUT, const char *, NhlClass, int, int);
%newobject  NhlRLCreate(NhlRLType);
%newobject  NhlFrame (int);
%newobject  NhlDestroy (int);
%newobject  NhlRLSetMDIntegerArray(int, char *, int *sequence_as_int, int, int *sequence_as_int);
%newobject  NhlRLSetMDDoubleArray(int, char *, double *sequence_as_double, int, int *sequence_as_int);
%newobject  NhlRLSetMDFloatArray(int, char *, float *sequence_as_float, int, int *sequence_as_int);
%newobject  NhlRLSetFloatArray(int, char *, float *sequence_as_float, int);
%newobject  NhlRLSetIntegerArray(int, char *, int *sequence_as_int, int);
%newobject  NhlRLSetStringArray(int, NhlString, NhlString *, int);
%newobject  NhlGetValues(int, int res_id);
%newobject  NhlGetFloat(int oid, char *name);
%newobject  NhlGetFloatArray(int oid, char *name, int *numberf);
%newobject  NhlGetInteger(int oid, char *name);
%newobject  NhlGetIntegerArray(int oid, char *name, int *numberi);
%newobject  NhlGetDouble(int oid, char *name);
%newobject  NhlGetDoubleArray(int oid, char *name, int *numberd);
%newobject  NhlAddOverlay(int,int,int);
%newobject  NhlClearWorkstation(int);
%newobject  NhlRemoveAnnotation(int, int);
%newobject  NhlAddAnnotation(int,int);
%newobject  NhlAppGetDefaultParentId();
%newobject  NhlGetParentWorkstation(int);
%newobject  NhlClassName(int);
%newobject  NhlGetString(int, NhlString);
%newobject  NhlAddData(int, NhlString, int);
%newobject  NhlRemoveData(int, NhlString, int);
%newobject  NhlRemoveOverlay(int, int, NhlBoolean);
%newobject  NhlGetStringArray(int oid, char *name, int *numbers);
%newobject  NhlGetNamedColorIndex(int, const char *);
%newobject  NhlGetBB(int, NhlBoundingBox *);
%newobject  NhlChangeWorkstation(int, int);
%newobject  NhlPGetBB(int, float *OUTPUT, float *OUTPUT, float *OUTPUT,
                                float *OUTPUT);
%newobject  NhlPNDCToData(int, float *sequence_as_float, 
                               float *sequence_as_float, int, float,
                               float, int, int);
%newobject NhlPDataToNDC(int, float *sequence_as_float, 
                               float *sequence_as_float, int, float,
                               float, int, int);
%newobject NhlGetMDFloatArray(int, char *);
%newobject NhlGetMDDoubleArray(int, char *);
%newobject NhlGetMDIntegerArray(int, char *);
%newobject  NhlPAppClass ();
%newobject  NhlPNcgmWorkstationClass ();
%newobject  NhlPXWorkstationClass ();
%newobject  NhlPPSWorkstationClass ();
%newobject  NhlPPDFWorkstationClass ();
%newobject  NhlPLogLinPlotClass();
%newobject  NhlPGraphicStyleClass();
%newobject  NhlPScalarFieldClass ();
%newobject  NhlPContourPlotClass();
%newobject  NhlPtextItemClass();
%newobject  NhlPscalarFieldClass();
%newobject  NhlPmapPlotClass();
%newobject  NhlPcoordArraysClass();
%newobject  NhlPxyPlotClass();
%newobject  NhlPtickMarkClass();
%newobject  NhlPtitleClass();
%newobject  NhlPlabelBarClass();
%newobject  NhlPlegendClass();
%newobject  NhlPvectorFieldClass();
%newobject  NhlPvectorPlotClass();
%newobject  NhlPstreamlinePlotClass();

%newobject NGGetNCARGEnv(const char *);

%newobject  open_wks_wrap(const char *, const char *, ResInfo *rlist,
                          ResInfo *rlist, nglRes *rlist);

%newobject labelbar_ndc_wrap(int, int, NhlString *, int,
                             void *sequence_as_void,
                             void *sequence_as_void, const char *,
                             const char *, ResInfo *rlist, nglRes *rlist);

%newobject legend_ndc_wrap(int, int, NhlString *, int,
                           void *sequence_as_void,
                           void *sequence_as_void, const char *,
                           const char *, ResInfo *rlist, nglRes *rlist);

%newobject  contour_wrap(int, void *sequence_as_void, 
                            const char *, int, int,
                            int, void *, const char *, int, void *, 
                            const char *,
                            int, void *, ResInfo *rlist, ResInfo *rlist, 
                            ResInfo *rlist, nglRes *rlist);
%newobject  map_wrap(int, ResInfo *rlist, nglRes *rlist);
%newobject  contour_map_wrap(int, void *sequence_as_void, 
                            const char *, int, int,
                            int, void *, const char *, int, void *, 
                            const char *, int,
                            void *, ResInfo *rlist, ResInfo *rlist, ResInfo *rlist,
                            nglRes *rlist);
%newobject  xy_wrap(int, void *sequence_as_void, void *sequence_as_void, 
                const char *, const char *, int, int *sequence_as_int,
                int, int *sequence_as_int, int, int, void *, void *,
                ResInfo *rlist, ResInfo *rlist, ResInfo *rlist, nglRes *rlist);
%newobject  y_wrap(int, void *sequence_as_void, 
                const char *, int, int *sequence_as_int, int, void *,
                ResInfo *rlist, ResInfo *rlist, ResInfo *rlist, nglRes *rlist);
%newobject  vector_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, const char *,
                           const char *, int, int, int, void *, 
                           const char *, int, void *, const char *, int, int,
                           void *, void *, ResInfo *rlist, ResInfo *rlist,
                           ResInfo *rlist, nglRes *rlist);
%newobject  vector_map_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, const char *, 
                           const char *, int, int, int, void *, 
                           const char *, int, void *, const char *, int, int, 
                           void *, void *, ResInfo *rlist, ResInfo *rlist, 
                           ResInfo *rlist, nglRes *rlist);
%newobject  vector_scalar_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void,
                           void *sequence_as_void, const char *, 
                           const char *, const char *, int, int, int, void *, 
                           const char *, int, void *, const char *, int, int, 
                           int, void *, void *, void *, 
                           ResInfo *rlist, ResInfo *rlist, ResInfo *rlist,
                           ResInfo *rlist, nglRes *rlist);
%newobject  vector_scalar_map_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, 
                           void *sequence_as_void, const char *, 
                           const char *, const char *, int, int, int, void *,
                           const char *, int, void *, const char *, int, int,
                           int, void *, void *, void *,
                           ResInfo *rlist, ResInfo *rlist, 
                           ResInfo *rlist, ResInfo *rlist, nglRes *rlist);
%newobject  streamline_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, const char *,
                           const char *, int, int, int, void *,
                           const char *, int, void *, const char *, 
                           int, int,
                           void *, void *, ResInfo *rlist, ResInfo *rlist, 
                           ResInfo *, nglRes *rlist);
%newobject  streamline_map_wrap(int, void *sequence_as_void, 
                            void *sequence_as_void, const char *,
                            const char *, int, int, int, void *,
                            const char *, int, void *, const char *, 
                            int, int,
                            void *, void *, ResInfo *rlist, ResInfo *rlist, 
                            ResInfo *rlist, nglRes *rlist);
%newobject  text_ndc_wrap(int, NhlString, void *sequence_as_void,
                             void *sequence_as_void, const char *,
                             const char *, ResInfo *rlist, nglRes *rlist);
%newobject  text_wrap(int, nglPlotId *plot, NhlString, void *sequence_as_void,
                             void *sequence_as_void, const char *,
                             const char *, ResInfo *rlist, nglRes *rlist);
%newobject  add_text_wrap(int, nglPlotId *plot, NhlString, void *sequence_as_void,
                             void *sequence_as_void, const char *,
                             const char *, ResInfo *rlist, ResInfo *rlist,
                             nglRes *rlist);

%newobject maximize_plots(int, nglPlotId *plot, int, int, nglRes *rlist);

%newobject  add_poly_wrap(int, nglPlotId *plot, void *sequence_as_void,                       void *sequence_as_void, const char *type_x,
                       const char *type_y, int, int, int, void *, void*,
                       NhlPolyType, ResInfo *rlist, nglRes *rlist);
%newobject ftcurvc(int, float *sequence_as_float, 
                         float *sequence_as_float,
                         int m, float *sequence_as_float);
%newobject ftcurvpc(int, float *sequence_as_float, 
                         float *sequence_as_float, float,
                         int m, float *sequence_as_float);
%newobject ftcurvpic(float, float, float, int, 
                           float *sequence_as_float, 
                           float *sequence_as_float);
%newobject  get_nglRes_i(int);
%newobject  get_nglRes_f(int);
%newobject  get_nglRes_c(int);
