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
#include <ncarg/hlu/View.h>

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

#include <numpy/arrayobject.h>

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
extern void c_nnseti(NhlString,int);
extern void c_nnsetrd(NhlString,double);
extern void c_nnsetc(NhlString,NhlString);

extern NhlClass NhlPAppClass ();
extern NhlClass NhlPNcgmWorkstationClass ();
extern NhlClass NhlPXWorkstationClass ();
extern NhlClass NhlPPSWorkstationClass ();
extern NhlClass NhlPPDFWorkstationClass ();
extern NhlClass NhlPLogLinPlotClass ();
extern NhlClass NhlPGraphicStyleClass ();
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
extern const char *NGGetNCARGEnv(const char *name);

extern void c_wmbarbp(int, float, float, float, float);
extern void c_wmsetip(NhlString,int);
extern void c_wmsetrp(NhlString,float);
extern void c_wmsetcp(NhlString,NhlString);
extern void c_wmstnmp(int, float, float, NhlString);
extern int  c_wmgetip(NhlString);
extern float c_wmgetrp(NhlString);
extern NhlString c_wmgetcp(NhlString);

extern void c_nnseti(NhlString, int);
extern void c_nnsetrd(NhlString, double);
extern void c_nnsetc(NhlString, NhlString);
extern void c_nngeti(NhlString, int *OUTPUT);
extern void c_nngetrd(NhlString, double *OUTPUT);
extern NhlString c_nngetcp(NhlString);

extern void *pvoid();

extern double c_dgcdist(double, double, double, double, int);
extern double c_dshowalskewt(double *, double *, double *, int);
extern double c_dpwskewt(double *, double *, int);
extern double c_dsatlftskewt(double, double);
extern double c_dtmrskewt(double, double);
extern double c_dtdaskewt(double, double);
extern void c_dptlclskewt(double, double, double, double *, double *);

static PyObject* t_output_helper(PyObject* target, PyObject* o) {
    PyObject*   o2;
    PyObject*   o3;

    if (!target) {                   
        target = o;
    } else if (target == Py_None) {  
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
  else if (pos == 49) {
    nglRlist.nglPointTickmarksOutward = ival;
  }
  else if (pos == 54) {
    nglRlist.nglXRefLineColor = ival;
  }
  else if (pos == 55) {
    nglRlist.nglYRefLineColor = ival;
  }
  else if (pos == 56) {
    nglRlist.nglMaskLambertConformal = ival;
  }
  else if (pos == 57) {
    nglRlist.nglMaskLambertConformalOutlineOn = ival;
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
  else if (pos == 49) {
    return(nglRlist.nglPointTickmarksOutward);
  }
  else if (pos == 54) {
    return(nglRlist.nglXRefLineColor);
  }
  else if (pos == 55) {
    return(nglRlist.nglYRefLineColor);
  }
  else if (pos == 56) {
    return(nglRlist.nglMaskLambertConformal);
  }
  else if (pos == 57) {
    return(nglRlist.nglMaskLambertConformalOutlineOn);
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
  else if (pos == 50) {
    nglRlist.nglXRefLine = ival;
  }
  else if (pos == 51) {
    nglRlist.nglYRefLine = ival;
  }
  else if (pos == 52) {
    nglRlist.nglXRefLineThicknessF = ival;
  }
  else if (pos == 53) {
    nglRlist.nglYRefLineThicknessF = ival;
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
  else if (pos == 50) {
    return(nglRlist.nglXRefLine);
  }
  else if (pos == 51) {
    return(nglRlist.nglYRefLine);
  }
  else if (pos == 52) {
    return(nglRlist.nglXRefLineThicknessF);
  }
  else if (pos == 53) {
    return(nglRlist.nglYRefLineThicknessF);
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
  PyObject *obj1,*obj2,*resultobj;

  npy_intp dims[1];

  rlati = (float *) malloc(npts*sizeof(float));
  rloni = (float *) malloc(npts*sizeof(float));
  c_mapgci(alat, alon, blat, blon, npts, rlati, rloni);
  dims[0] = (npy_intp)npts;
  obj1 = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,
                                                (char *) rlati);
  obj2 = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,
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
  PyObject *obj1,*obj2,*obj3,*obj4,*obj5,*resultobj;
  int jlcl, jlfc, jcross;
  npy_intp dims[1];
  double cape, *tparcel;

  cape = c_dcapethermo(penv, tenv, nlvl, lclmb, iprnt, &tparcel, tmsg,
                       &jlcl, &jlfc, &jcross);                       

  dims[0] = (npy_intp)nlvl;
  obj1 = (PyObject *) PyFloat_FromDouble(cape);
  obj2 = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_DOUBLE,
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

  int ier;
  npy_intp dims[1];

  yo = (float *) malloc(m*sizeof(float));
  ier = c_ftcurv(n,x,y,m,xo,yo);
  status = (PyObject *) PyInt_FromLong((long) ier);
  dims[0] = (npy_intp)m;
  obj1 = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,
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

  int ier;
  npy_intp dims[1];

  yo = (float *) malloc(m*sizeof(float));
  ier = c_ftcurvp(n,x,y,p,m,xo,yo);
  status = (PyObject *) PyInt_FromLong((long) ier);
  dims[0] = (npy_intp)m;
  obj1 = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,
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
  PyObject *status,*resultobj;

  int ier;

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
    ng_size_t nnumber;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetFloatArray(grlist,name,&fscales,&nnumber);
    NhlGetValues(oid,grlist);
    *number = (int)nnumber;

    return (fscales);
}

double *NhlGetDoubleArray(int oid, char *name, int *number)
{
    int grlist;
    double *dar;
    ng_size_t nnumber;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetDoubleArray(grlist,name,&dar,&nnumber);
    NhlGetValues(oid,grlist);

    *number = (int)nnumber;
    return (dar);
}

int *NhlGetIntegerArray(int oid, char *name, int *number)
{
    int grlist;
    int *iar;
    ng_size_t nnumber;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetIntegerArray(grlist,name,&iar,&nnumber);
    NhlGetValues(oid,grlist);

    *number = (int)nnumber;
    return (iar);
}

NhlString *NhlGetStringArray(int oid, char *name, int *number)
{
    int grlist;
    NhlString *slist;
    ng_size_t nnumber;

    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetStringArray(grlist,name,&slist,&nnumber);
    NhlGetValues(oid,grlist);

    *number = (int)nnumber;
    return (slist);
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
  int status,rval;
  npy_intp dims[1];
  float out_of_range;

  xout = (float *) malloc(n*sizeof(float));
  yout = (float *) malloc(n*sizeof(float));
  rval = (int) NhlQNDCToData(pid, x, y, n, xout, yout, xmissing,
                             ymissing, ixmissing, iymissing,
                             &status, &out_of_range);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 
  dims[0] = (npy_intp)n;
  obj1 = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,
                                                (char *) xout);
  obj2 = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,
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
  int status,rval;
  npy_intp dims[1];
  float out_of_range;

  xout = (float *) malloc(n*sizeof(float));
  yout = (float *) malloc(n*sizeof(float));
  rval = (int) NhlQDataToNDC(pid, x, y, n, xout, yout, xmissing,
                             ymissing, ixmissing, iymissing,
                             &status, &out_of_range);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 
  dims[0] = (npy_intp)n;
  obj1 = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,
                                                (char *) xout);
  obj2 = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,
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
  int i, num_dims, grlist;
  ng_size_t *len_dims;
  float *bptr;
  NhlErrorTypes rval;
  npy_intp *len_dims_npy;

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  rval = NhlRLGetMDFloatArray(grlist,name,&bptr,&num_dims,&len_dims);
  NhlGetValues(pid,grlist);

  len_dims_npy = (npy_intp *)malloc(num_dims*sizeof(npy_intp));
  for(i=0;i<num_dims;i++) len_dims_npy[i] = (npy_intp)len_dims[i];

  obj1 = (PyObject *) PyArray_SimpleNewFromData(num_dims,len_dims_npy,
                                                PyArray_FLOAT,(char *) bptr);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 

  free(len_dims_npy);

  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,nhlerr);
  resultobj = t_output_helper(resultobj,obj1);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

PyObject *NhlGetMDDoubleArray(int pid, char *name) {
  PyObject *obj1, *nhlerr, *resultobj;
  int i, num_dims, grlist;
  ng_size_t *len_dims;
  double *bptr;
  NhlErrorTypes rval;
  npy_intp *len_dims_npy;

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  rval = NhlRLGetMDDoubleArray(grlist,name,&bptr,&num_dims,&len_dims);
  NhlGetValues(pid,grlist);

  len_dims_npy = (npy_intp *)malloc(num_dims*sizeof(npy_intp));
  for(i=0;i<num_dims;i++) len_dims_npy[i] = (npy_intp)len_dims[i];

  obj1 = (PyObject *) PyArray_SimpleNewFromData(num_dims,len_dims_npy,
                                                PyArray_DOUBLE,(char *) bptr);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 

  free(len_dims_npy);

  resultobj = Py_None;
  resultobj = t_output_helper(resultobj,nhlerr);
  resultobj = t_output_helper(resultobj,obj1);
  if (resultobj == Py_None) Py_INCREF(Py_None);
  return resultobj;
}

PyObject *NhlGetMDIntegerArray(int pid, char *name) {
  PyObject *obj1, *nhlerr, *resultobj;
  int i, num_dims, grlist;
  ng_size_t *len_dims;
  int *bptr;
  NhlErrorTypes rval;
  npy_intp *len_dims_npy;

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  rval = NhlRLGetMDIntegerArray(grlist,name,&bptr,&num_dims,&len_dims);
  NhlGetValues(pid,grlist);

  len_dims_npy = (npy_intp *)malloc(num_dims*sizeof(npy_intp));
  for(i=0;i<num_dims;i++) len_dims_npy[i] = (npy_intp)len_dims[i];

  obj1 = (PyObject *) PyArray_SimpleNewFromData(num_dims,len_dims_npy,
                                                PyArray_INT,(char *) bptr);
  nhlerr = (PyObject *) PyInt_FromLong((long) rval); 

  free(len_dims_npy);

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

%include cpointer.i
%pointer_functions(int, intp);

%include carrays.i
%array_functions(float,floatArray)

//
// Include the required NumPy header.
//
%header %{
#include <numpy/arrayobject.h>
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
// The first three typemaps are here since the recent update of
// SWIG parses all input arguments as objects and does not recognize
// numpy scalars as the appropriate numbers.
//
%typemap (in) float {
  $1 = (float) PyFloat_AsDouble ($input);
}
%typemap (in) double {
  $1 = PyFloat_AsDouble ($input);
}
%typemap (in) int {
  $1 = (int) PyInt_AsLong ($input);
}



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
   (PyArrayObject *) PyArray_ContiguousFromAny($input,PyArray_DOUBLE,0,0);
  ndims = arr->nd;
  for (i = 0; i < ndims; i++) {
    tdims *= (int) arr->dimensions[i];
  }
  $1 = (float *) d2f(tdims, (double *) arr->data);
}

%typemap (in) nglRes *rlist {
  $1 = (void *) &nglRlist;
}

%typemap (in) double *sequence_as_double {
  PyArrayObject *arr;
  arr =
   (PyArrayObject *) PyArray_ContiguousFromAny($input,PyArray_DOUBLE,0,0);
  $1 = (double *) arr->data;
}

%typemap (in) void *sequence_as_void {
  PyArrayObject *arr;
  arr =
   (PyArrayObject *) PyArray_ContiguousFromAny($input,PyArray_DOUBLE,0,0);
  $1 = (void *) arr->data;
}

%typemap (in) int *sequence_as_int {
  PyArrayObject *arr;
  arr =
   (PyArrayObject *) PyArray_ContiguousFromAny($input,PyArray_INT,0,0);
  $1 = (int *) arr->data;
}

%typemap (in) ng_size_t *sequence_as_ngsizet {
  PyArrayObject *arr;
  arr =
   (PyArrayObject *) PyArray_ContiguousFromAny($input,PyArray_LONG,0,0);
  $1 = (long *) arr->data;
}

%typemap (argout) (int *numberf) {
  npy_intp dims[1];
  dims[0] = (npy_intp)*($1);
  $result = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,(char *) result);
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
  npy_intp dims[1];
  dims[0] = (int)*($1);
  $result = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_DOUBLE,(char *) result);
}
%typemap(in,numinputs=0) int *numberd (int tempx) {
  $1 = &tempx;
}

%typemap (argout) (int nxir, int nyir, float *p_array_float_out[]) {
  npy_intp dims[2];
  PyObject *o;
  dims[0] = (npy_intp)arg10;
  dims[1] = (npy_intp)arg11;
  o = (PyObject *)PyArray_SimpleNewFromData(2,dims,PyArray_FLOAT,
                  (char *) arg12[0]);
  resultobj = t_output_helper(resultobj,o);
}
%typemap(in,numinputs=0) float *p_array_float_out[] (float *tempx) {
  $1 = &tempx;
}

%typemap (argout) (int nxir, int nyir, double *p_array_double_out[]) {
  npy_intp dims[2];
  PyObject *o;
  dims[0] = (npy_intp)$1;
  dims[1] = (npy_intp)$2;
  o = (PyObject *)PyArray_SimpleNewFromData(2,dims,PyArray_DOUBLE,
                  (char *) $3[0]);
  resultobj = t_output_helper(resultobj,o);
}
%typemap(in,numinputs=0) double *p_array_double_out[] (double *tempx) {
  $1 = &tempx;
}

%typemap (argout)(int nxir, int nyir, int nzir, double *p_array3_double_out[]) {
  npy_intp dims[3];
  PyObject *o;
  dims[0] = (npy_intp)$1;
  dims[1] = (npy_intp)$2;
  dims[2] = (npy_intp)$3;
  o = (PyObject *)PyArray_SimpleNewFromData(3,dims,PyArray_DOUBLE,
                  (char *) $4[0]);
  resultobj = t_output_helper(resultobj,o);
}
%typemap(in,numinputs=0) double *p_array3_double_out[] (double *tempx) {
  $1 = &tempx;
}

%typemap (argout) (int nx, float *t_array_float_out) {
  npy_intp dims[1];
  PyObject *o;
  dims[0] = (npy_intp)nx;
  o = (PyObject *)PyArray_SimpleNewFromData(2,dims,PyArray_FLOAT,
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
  npy_intp dims[1];
  dims[0] = (npy_intp)*($1);
  $result = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_INT,(char *) result);
}
%typemap(in,numinputs=0) int *numberi (int tempx) {
  $1 = &tempx;
}

%typemap(in,numinputs=0) int *numbers (int tempx) %{
  $1 = &tempx;
%}

%typemap (argout) (float **data_addr, int *num_elements) {
  npy_intp dims[1];
  PyObject *o;
  dims[0] = (npy_intp)*$2;
  o = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,(char *) *$1);
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
 PyObject *l_base,      *l_contour , *l_vector,  *l_streamline,
          *l_map,       *l_xy      , *l_xydspec, *l_text,
          *l_primitive, *l_labelbar, *l_legend,  *l_cafield, 
          *l_sffield, *l_vffield;
 nglPlotId pid;
 int i;

 pid = $1;

 return_list = PyList_New(14);

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

 if (pid.nlabelbar == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,9,Py_None);
 }
 else {
   l_labelbar = PyList_New(pid.nlabelbar);
   for (i = 0; i < pid.nlabelbar; i++) {
     PyList_SetItem(l_labelbar,i,PyInt_FromLong((long) *(pid.labelbar+i)));
   }
   PyList_SetItem(return_list,9,l_labelbar);
 }

 if (pid.nlegend == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,10,Py_None);
 }
 else {
   l_legend = PyList_New(pid.nlegend);
   for (i = 0; i < pid.nlegend; i++) {
     PyList_SetItem(l_legend,i,PyInt_FromLong((long) *(pid.legend+i)));
   }
   PyList_SetItem(return_list,10,l_legend);
 }

 if (pid.ncafield == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,11,Py_None);
 }
 else {
   l_cafield = PyList_New(pid.ncafield);
   for (i = 0; i < pid.ncafield; i++) {
     PyList_SetItem(l_cafield,i,PyInt_FromLong((long) *(pid.cafield+i)));
   }
   PyList_SetItem(return_list,11,l_cafield);
 }

 if (pid.nsffield == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,12,Py_None);
 }
 else {
   l_sffield = PyList_New(pid.nsffield);
   for (i = 0; i < pid.nsffield; i++) {
     PyList_SetItem(l_sffield,i,PyInt_FromLong((long) *(pid.sffield+i)));
   }
   PyList_SetItem(return_list,12,l_sffield);
 }

 if (pid.nvffield == 0) {
   Py_INCREF(Py_None); 
   PyList_SetItem(return_list,13,Py_None);
 }
 else {
   l_vffield = PyList_New(pid.nvffield);
   for (i = 0; i < pid.nvffield; i++) {
     PyList_SetItem(l_vffield,i,PyInt_FromLong((long) *(pid.vffield+i)));
   }
   PyList_SetItem(return_list,13,l_vffield);
 }

 Py_INCREF(return_list); 
 $result = return_list;
}

%typemap(in) (nglPlotId *plot_seq) {
  nglPlotId *inlist;
  PyObject *ptmp;
  PyObject *l_base,      *l_contour , *l_vector,  *l_streamline,
           *l_map,       *l_xy      , *l_xydspec, *l_text,
           *l_primitive, *l_labelbar, *l_legend,  *l_cafield, 
           *l_sffield,   *l_vffield;
  int      *ibase,       *icontour ,  *ivector,   *istreamline,
           *imap,        *ixy      ,  *ixydspec,  *itext,
           *iprimitive,  *ilabelbar,  *ilegend,   *icafield,
           *isffield,    *ivffield;
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

    l_labelbar = PyList_GetItem(ptmp,9);
    if (l_labelbar == Py_None) {
      inlist[j].nlabelbar = 0;
      inlist[j].labelbar = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_labelbar)) {
        printf("PlotId labelbar element must be None or a Python list\n");
      }
      else {
        inlist[j].nlabelbar = PyList_Size(l_labelbar);
        ilabelbar = (int *) malloc(inlist[j].nlabelbar*sizeof(int));
        for (i = 0; i < inlist[j].nlabelbar; i++) {
          *(ilabelbar+i) = (int) PyInt_AsLong(PyList_GetItem(l_labelbar,i));
        }
        inlist[j].labelbar = ilabelbar;
      }
    }

    l_legend = PyList_GetItem(ptmp,10);
    if (l_legend == Py_None) {
      inlist[j].nlegend = 0;
      inlist[j].legend = (int *) NULL;
    }
    else {
      if (!PyList_Check(l_legend)) {
        printf("PlotId legend element must be None or a Python list\n");
      }
      else {
        inlist[j].nlegend = PyList_Size(l_legend);
        ilegend = (int *) malloc(inlist[j].nlegend*sizeof(int));
        for (i = 0; i < inlist[j].nlegend; i++) {
          *(ilegend+i) = (int) PyInt_AsLong(PyList_GetItem(l_legend,i));
        }
        inlist[j].legend = ilegend;
      }
    }

    l_cafield = PyList_GetItem(ptmp,11);
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

    l_sffield = PyList_GetItem(ptmp,12);
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

    l_vffield = PyList_GetItem(ptmp,13);
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

  PyObject *l_base,      *l_contour , *l_vector,  *l_streamline,
           *l_map,       *l_xy      , *l_xydspec, *l_text,
           *l_primitive, *l_labelbar, *l_legend,  *l_cafield, 
           *l_sffield,   *l_vffield;
  int      *ibase,       *icontour ,  *ivector,   *istreamline,
           *imap,        *ixy      ,  *ixydspec,  *itext,
           *iprimitive,  *ilabelbar,  *ilegend,   *icafield,  
           *isffield,    *ivffield;

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

  l_labelbar = PyList_GetItem($input,9);
  if (l_labelbar == Py_None) {
    inlist.nlabelbar = 0;
    inlist.labelbar = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_labelbar)) {
      printf("PlotId labelbar element must be None or a Python list\n");
    }
    else {
      inlist.nlabelbar = PyList_Size(l_labelbar);
      ilabelbar = (int *) malloc(inlist.nlabelbar*sizeof(int));
      for (i = 0; i < inlist.nlabelbar; i++) {
        *(ilabelbar+i) = (int) PyInt_AsLong(PyList_GetItem(l_labelbar,i));
      }
      inlist.labelbar = ilabelbar;
    }
  }

  l_legend = PyList_GetItem($input,10);
  if (l_legend == Py_None) {
    inlist.nlegend = 0;
    inlist.legend = (int *) NULL;
  }
  else {
    if (!PyList_Check(l_legend)) {
      printf("PlotId legend element must be None or a Python list\n");
    }
    else {
      inlist.nlegend = PyList_Size(l_legend);
      ilegend = (int *) malloc(inlist.nlegend*sizeof(int));
      for (i = 0; i < inlist.nlegend; i++) {
        *(ilegend+i) = (int) PyInt_AsLong(PyList_GetItem(l_legend,i));
      }
      inlist.legend = ilegend;
    }
  }

  l_cafield = PyList_GetItem($input,11);
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

  l_sffield = PyList_GetItem($input,12);
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

  l_vffield = PyList_GetItem($input,13);
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
  npy_intp dims[1];
  PyObject *o;
  dims[0] = (npy_intp)arg4;
  o = (PyObject *)PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,
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
  npy_intp dims[1];
  dims[0] = (npy_intp)arg1;
  $result = (PyObject *) PyArray_SimpleNewFromData(1,dims,PyArray_FLOAT,(char *) arg8);
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
  int rnumber,count;
  Py_ssize_t pos=0;
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
  int i,list_type,count;
  ng_size_t list_len;
  Py_ssize_t pos=0;
  PyObject *key,*value;
  PyArrayObject *arr;
  char **strings;
  double *dvals;
  int *ivals,array_type,rlist,ndims;
  ng_size_t *len_dims;
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
          printf("Tuple values are not allowed to have list or tuple items.\n");
          return NULL;
        }
        list_len = (ng_size_t)PyTuple_Size(value);
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
          printf("Use NumPy arrays for multiple dimension arrays.\n");
          return NULL;
        }
        list_len = (ng_size_t)PyList_Size(value);
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
 *  Check for scalars.
 */
      else if (PyArray_IsAnyScalar(value)) {
/*
 *  Check for Python Scalars.
 */
        if (PyArray_IsPythonScalar(value)) {
/*
 *  value is a Python int.
 */
          if (PyInt_Check(value)) {
            NhlRLSetInteger(rlist,PyString_AsString(key),
                               (int) PyInt_AsLong(value));
          }
/*
 *  value is a Python float.
 */
          else if (PyFloat_Check(value)) {
            NhlRLSetDouble(rlist,PyString_AsString(key),
                               PyFloat_AsDouble(value));
          }
/*
 *  value is a Python long.
 */
          else if (PyLong_Check(value)) {
            NhlRLSetInteger(rlist,PyString_AsString(key),
                               (int) PyInt_AsLong(value));
          }
/*
 *  value is a Python string
 */
          else if (PyString_Check(value)) {
            NhlRLSetString(rlist,PyString_AsString(key),
                               PyString_AsString(value));
          }
        }
/*
 *  otherwise we have numpy scalars
 */
        else {
/*
 *  value is a numpy int.
 */
          if (PyArray_IsScalar(value,Int)) {
            NhlRLSetInteger(rlist,PyString_AsString(key),
                               (int) PyInt_AsLong(value));
          }
/*
 *  value is a numpy float.
 */
          else if (PyArray_IsScalar(value,Float)) {
            NhlRLSetDouble(rlist,PyString_AsString(key),
                               PyFloat_AsDouble(value));
          }
/*
 *  value is a numpy long.
 */
          else if (PyArray_IsScalar(value,Long)) {
            NhlRLSetInteger(rlist,PyString_AsString(key),
                               (int) PyInt_AsLong(value));
          }
/*
 *  value is a numpy string
 */
          else if (PyArray_IsScalar(value,String)) {
            NhlRLSetString(rlist,PyString_AsString(key),
                               PyString_AsString(value));
          }
        }
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
          arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                     ((PyObject *) value,PyArray_LONG,0,0);
          lvals = (long *)arr->data;
          ndims = arr->nd;
          len_dims = (ng_size_t *)malloc(ndims*sizeof(ng_size_t));
          for(i = 0; i < ndims; i++ ) {
            len_dims[i] = (ng_size_t)arr->dimensions[i];
          }
          NhlRLSetMDLongArray(rlist,PyString_AsString(key),lvals,ndims,len_dims);
        }
        else if (array_type == PyArray_FLOAT || array_type == PyArray_DOUBLE) {
          arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                 ((PyObject *) value,PyArray_DOUBLE,0,0);
          dvals = (double *)arr->data;
          ndims = arr->nd;
          len_dims = (ng_size_t *)malloc(ndims*sizeof(ng_size_t));
          for(i = 0; i < ndims; i++ ) {
            len_dims[i] = (ng_size_t)arr->dimensions[i];
          }
          NhlRLSetMDDoubleArray(rlist,PyString_AsString(key),dvals,ndims,len_dims);
        }
        else {
          printf(
            "NumPy arrays must be of type int, int32, float, float0, float32, or float64.\n");
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
  int i,list_type,count;
  ng_size_t list_len;
  Py_ssize_t pos=0;
  PyObject *key,*value;
  PyArrayObject *arr;
  char **strings;
  double *dvals;
  int *ivals,array_type,rlist,ndims;
  ng_size_t *len_dims;
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
          printf("Tuple values are not allowed to have list or tuple items.\n");
          return NULL;
        }
        list_len = (ng_size_t)PyTuple_Size(value);
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
          printf("Use NumPy arrays for multiple dimension arrays.\n");
          return NULL;
        }
        list_len = (ng_size_t)PyList_Size(value);
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
          arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                ((PyObject *) value,PyArray_LONG,0,0);
          lvals = (long *)arr->data;
          ndims = arr->nd;
          len_dims = (ng_size_t *)malloc(ndims*sizeof(ng_size_t));
          for(i = 0; i < ndims; i++ ) {
            len_dims[i] = (ng_size_t)arr->dimensions[i];
          }
          NhlRLSetMDLongArray(rlist,PyString_AsString(key),lvals,ndims,len_dims);
        }
        else if (array_type == PyArray_FLOAT || array_type == PyArray_DOUBLE) {
          arr = (PyArrayObject *) PyArray_ContiguousFromAny \
                 ((PyObject *) value,PyArray_DOUBLE,0,0);
          dvals = (double *)arr->data;
          ndims = arr->nd;
          len_dims = (ng_size_t *)malloc(ndims*sizeof(ng_size_t));
          for(i = 0; i < ndims; i++ ) {
            len_dims[i] = (ng_size_t)arr->dimensions[i];
          }
          NhlRLSetMDDoubleArray(rlist,PyString_AsString(key),dvals,ndims,len_dims);
        }
        else {
          printf(
            "NumPy arrays must be of type int, int32, float, float0, float32, or float64.\n");
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

extern float *c_cssgrid(int, float *sequence_as_float, float *sequence_as_float, float *sequence_as_float, int, int, float *sequence_as_float, float *sequence_as_float, int *);

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
extern NhlErrorTypes NhlRLSetMDIntegerArray(int, char *, int *sequence_as_int, int, ng_size_t *sequence_as_ngsizet);
extern NhlErrorTypes NhlRLSetMDDoubleArray(int, char *, double *sequence_as_double, int, ng_size_t *sequence_as_ngsizet);
extern NhlErrorTypes NhlRLSetMDFloatArray(int, char *, float *sequence_as_float, int, ng_size_t *sequence_as_ngsizet);
extern NhlErrorTypes NhlRLSetFloatArray(int, char *, float *sequence_as_float, ng_size_t);
extern NhlErrorTypes NhlRLSetIntegerArray(int, char *, int *sequence_as_int, ng_size_t);
extern NhlErrorTypes NhlRLSetStringArray(int, NhlString, NhlString *, ng_size_t);
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
extern NhlErrorTypes NhlChangeWorkstation(int, int);
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

extern nglPlotId blank_plot_wrap(int, ResInfo *rlist, nglRes *rlist);
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
extern nglPlotId streamline_scalar_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void,
                           void *sequence_as_void, const char *, 
                           const char *, const char *, int, int, int, void *, 
                           const char *, int, void *, const char *, int, int, 
                           int, void *, void *, void *, 
                           ResInfo *rlist, ResInfo *rlist, ResInfo *rlist,
                           ResInfo *rlist, nglRes *rlist);
extern nglPlotId streamline_scalar_map_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, 
                           void *sequence_as_void, const char *, 
                           const char *, const char *, int, int, int, void *,
                           const char *, int, void *, const char *, int, int,
                           int, void *, void *, void *,
                           ResInfo *rlist, ResInfo *rlist, 
                           ResInfo *rlist, ResInfo *rlist, nglRes *rlist);
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
                       const char *type_y, int, int, int,
                       void *sequence_as_void, void *sequence_as_void,
                       NhlPolyType, ResInfo *rlist, nglRes *rlist);
extern nglPlotId add_poly_wrap(int, nglPlotId *plot, void *sequence_as_void,
                       void *sequence_as_void, const char *type_x,
                       const char *type_y, int, int, int, int,
                       void *sequence_as_void, void *sequence_as_void,
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

extern void getbb(int, float *OUTPUT, float *OUTPUT, float *OUTPUT, float *OUTPUT);


extern void c_wmbarbp(int, float, float, float, float);
extern void c_wmsetip(NhlString,int);
extern void c_wmsetrp(NhlString,float);
extern void c_wmsetcp(NhlString,NhlString);
extern void c_wmstnmp(int, float, float, NhlString);
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
%newobject  NhlRLSetMDIntegerArray(int, char *, int *sequence_as_int, int, ng_size_t *sequence_as_ngsizet);
%newobject  NhlRLSetMDDoubleArray(int, char *, double *sequence_as_double, int, ng_size_t *sequence_as_ngsizet);
%newobject  NhlRLSetMDFloatArray(int, char *, float *sequence_as_float, int, ng_size_t *sequence_as_ngsizet);
%newobject  NhlRLSetFloatArray(int, char *, float *sequence_as_float, ng_size_t);
%newobject  NhlRLSetIntegerArray(int, char *, int *sequence_as_int, ng_size_t);
%newobject  NhlRLSetStringArray(int, NhlString, NhlString *, ng_size_t);
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
%newobject  NhlChangeWorkstation(int, int);
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

%newobject  blank_plot_wrap(int, ResInfo *rlist, nglRes *rlist);
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
%newobject  streamline_scalar_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void,
                           void *sequence_as_void, const char *, 
                           const char *, const char *, int, int, int, void *, 
                           const char *, int, void *, const char *, int, int, 
                           int, void *, void *, void *, 
                           ResInfo *rlist, ResInfo *rlist, ResInfo *rlist,
                           ResInfo *rlist, nglRes *rlist);
%newobject  streamline_scalar_map_wrap(int, void *sequence_as_void, 
                           void *sequence_as_void, 
                           void *sequence_as_void, const char *, 
                           const char *, const char *, int, int, int, void *,
                           const char *, int, void *, const char *, int, int,
                           int, void *, void *, void *,
                           ResInfo *rlist, ResInfo *rlist, 
                           ResInfo *rlist, ResInfo *rlist, nglRes *rlist);
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

%newobject poly_wrap(int, nglPlotId *plot, void *sequence_as_void, 
                     void *sequence_as_void, const char *type_x,
                     const char *type_y, int, int, int, 
                     void *sequence_as_void, void *sequence_as_void,
                     NhlPolyType, ResInfo *rlist, nglRes *rlist);
%newobject  add_poly_wrap(int, nglPlotId *plot, void *sequence_as_void,
                          void *sequence_as_void, const char *type_x,
                          const char *type_y, int, int, int, int,
                          void *sequence_as_void, void *sequence_as_void,
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
