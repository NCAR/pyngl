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

extern res_names trname;

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

const char *NGGetNCARGEnv(const char *name)
{
  return (_NGGetNCARGEnv(name));
}

void *pvoid()
{
  void *p;
  return (p);
}

void test_res_names(void *name_list) {
/*  res_names rn; */
  printf ("Got to test_res_names\n");
}
