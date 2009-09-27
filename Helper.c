#include <ncarg/hlu/GraphicStyleP.h>
#include <ncarg/hlu/LogLinPlotP.h>
#include <ncarg/hlu/PSWorkstationP.h>
#include <ncarg/hlu/PDFWorkstationP.h>
#include <ncarg/hlu/XWorkstationP.h>
#include <ncarg/hlu/NcgmWorkstationP.h>
#include <ncarg/hlu/AppP.h>
#include <ncarg/hlu/hluP.h>
#include <ncarg/hlu/ResListP.h>
#include <ncarg/hlu/ConvertP.h>
#include <ncarg/hlu/VarArg.h>
#include <ncarg/hlu/ScalarField.h>
#include <ncarg/hlu/ContourPlot.h>
#include <ncarg/hlu/MapPlot.h>
#include <ncarg/hlu/XyPlot.h>
#include <ncarg/hlu/CoordArrays.h>
#include <ncarg/hlu/StreamlinePlot.h>
#include <ncarg/hlu/VectorPlot.h>
#include <stdlib.h>
#include <ncarg/c.h>
#include <ncarg/hlu/hlu.h>
#include <ncarg/hlu/NresDB.h>

/*
 *  Externals for processing functions.
 */
extern double NGCALLF(dgcdist,DGCDIST)(double *,double *,double *,double *,
                                       int *);
extern double NGCALLF(dcapethermo,DCAPETHERMO)(double *, double *,
  int *, double *, int *, double *, double *, int *, int *, int *);
extern void NGCALLF(dptlclskewt,DPTLCLSKEWT)(double *, double *, double *,
                                             double *, double *);
extern double NGCALLF(dtmrskewt,DTMRSKEWT)(double *,double *);
extern double NGCALLF(dtdaskewt,DTDASKEWT)(double *,double *);
extern double NGCALLF(dsatlftskewt,DSATLFTSKEWT)(double *,double *);
extern double NGCALLF(dshowalskewt,DSHOWALSKEWT)(double *,double *,
                                                 double *,int *);
extern double NGCALLF(dpwskewt,DPWSKEWT)(double *,double *,int*);
extern void NGCALLF(gaqdncl,GAQDNCL)(int *, double *, double *, double *, int *, int *);
extern void c_nngetc(char *, char *);

char *c_nngetcp(char *);

/*
 *  Various Nhl functions.
 */
NhlClass NhlPAppClass ()
{
  return((NhlClass) NhlappClass);
}
NhlClass NhlPNcgmWorkstationClass ()
{
  return((NhlClass) NhlncgmWorkstationClass);
}
NhlClass NhlPXWorkstationClass ()
{
  return((NhlClass) NhlxWorkstationClass);
}
NhlClass NhlPPSWorkstationClass ()
{
  return((NhlClass) NhlpsWorkstationClass);
}
NhlClass NhlPPDFWorkstationClass ()
{
  return((NhlClass) NhlpdfWorkstationClass);
}
NhlClass NhlPLogLinPlotClass ()
{
  return(NhllogLinPlotClass);
}
NhlClass NhlPGraphicStyleClass ()
{
  return(NhlgraphicStyleClass);
}
NhlClass NhlPScalarFieldClass ()
{
  return(NhlscalarFieldClass);
}
NhlClass NhlPContourPlotClass()
{
  return(NhlcontourPlotClass);
}
NhlClass NhlPtextItemClass()
{
  return(NhltextItemClass);
}
NhlClass NhlPscalarFieldClass()
{
  return(NhlscalarFieldClass);
}
NhlClass NhlPmapPlotClass()
{
  return(NhlmapPlotClass);
}
NhlClass NhlPcoordArraysClass()
{
  return(NhlcoordArraysClass);
}
NhlClass NhlPxyPlotClass()
{
  return(NhlxyPlotClass);
}
NhlClass NhlPtickMarkClass()
{
  return(NhltickMarkClass);
}
NhlClass NhlPtitleClass()
{
  return(NhltitleClass);
}
NhlClass NhlPlabelBarClass()
{
  return(NhllabelBarClass);
}
NhlClass NhlPlegendClass()
{
  return(NhllegendClass);
}
NhlClass NhlPvectorFieldClass()
{
  return(NhlvectorFieldClass);
}
NhlClass NhlPvectorPlotClass()
{
  return(NhlvectorPlotClass);
}
NhlClass NhlPstreamlinePlotClass()
{
  return(NhlstreamlinePlotClass);
}

/*
 *  Misc.
 */
const char *NGGetNCARGEnv(const char *name)
{
  return (_NGGetNCARGEnv(name));
}

void *pvoid()
{
  void *p;
  return (p);
}

/*
 *  Processing and processing support functions.
 */
double c_dgcdist(double lat1, double lon1, double lat2, double lon2, int iu) {
  return   (double) NGCALLF(dgcdist,DGCDIST)(&lat1, &lon1, &lat2, &lon2, &iu);
}
double c_dcapethermo(double *penv, double *tenv, int nlvl, double lclmb, 
  int iprnt, double **tparcel, double tmsg, int *jlcl, int *jlfc, int *jcross) {

  *tparcel = (double *) calloc(nlvl, sizeof(double));
  return (double) NGCALLF(dcapethermo,DCAPETHERMO)(penv, tenv, &nlvl,
                  &lclmb, &iprnt, *tparcel, &tmsg, jlcl, jlfc, jcross);
}
void c_dptlclskewt(double p, double t, double td, double *pc, double *tc) {
  NGCALLF(dptlclskewt,DPTLCLSKEWT)(&p, &t, &td, pc, tc);
}
double c_dtmrskewt(double w, double p) {
  return  (double) NGCALLF(dtmrskewt,DTMRSKEWT)(&w, &p);
}
double c_dtdaskewt(double w, double p) {
  return  (double) NGCALLF(dtdaskewt,DTDASKEWT)(&w, &p);
}
double c_dsatlftskewt(double thw, double p) {
  return  (double) NGCALLF(dsatlftskewt,DSATLFTSKEWT)(&thw, &p);
}
double c_dshowalskewt(double *p, double *t, double *td, int nlvls) {
  return  (double) NGCALLF(dshowalskewt,DSHOWALSKEWT)(p, t, td, &nlvls);
}
double c_dpwskewt(double *td, double *p, int n) {
  return  (double) NGCALLF(dpwskewt,DPWSKEWT)(td, p, &n);
}

NhlErrorTypes NglGaus (int nlat, double **output)
{
  int nl,lwork = 0,i,ierror;
  double *theta,*wts,*work = NULL;
  double rtod = (double)180.0/(double)3.14159265358979323846;

  nl    = 2 * nlat;
  theta = (double*)malloc(sizeof(double)*nl);
  wts   = (double*)malloc(sizeof(double)*nl);
  lwork = 4 * nl*(nl+1)+2;
  work  = (double*)malloc(sizeof(double)*lwork);
  NGCALLF(gaqdncl,GAQDNCL)(&nl,theta,wts,work,&lwork,&ierror);
  free(work);
  *output = (double*)malloc(sizeof(double)*nl*2);
  for(i = 0; i < nl; i++) {
    (*output)[2*i]    = rtod*theta[i] - 90.0;
    (*output)[2*i+1]  = wts[i];
  }
  free(wts);
  free(theta);

  return NhlNOERROR;
}

char *c_nngetcp(char *pnam) {
  static char xc[100];

  c_nngetc(pnam,xc);
  return &xc[0];
}
