#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ncarg/hlu/hlu.h>
#include <ncarg/hlu/ResList.h>
#include <ncarg/hlu/App.h>
#include <ncarg/hlu/XWorkstation.h>
#include <ncarg/hlu/NcgmWorkstation.h>
#include <ncarg/hlu/PSWorkstation.h>
#include <ncarg/hlu/PDFWorkstation.h>
#include <ncarg/hlu/ContourPlot.h>
#include <ncarg/hlu/ScalarField.h>
#include <netcdf.h>

main()
{
/*
 * Declare variables for the HLU routine calls.
 */

  int     app, wks, contour, sf_rlist, cn_rlist, wk_rlist;

/*
 * Prototype the external functions. 
 */
  extern int gsn_create_app(char *);
  extern int gsn_open_wks(char *, char *, int);
  extern int gsn_contour(int, float *, int, int, int, int);

/*
 * Declare variables for getting information from netCDF file.
 */

  int   ncid, t_id, lon_id, lat_id;
  float *T;
  int   *lon, *lat;
  long  start[3], count[3], nlon, nlat;
  char  filename[256];
  const char *dir = _NGGetNCARGEnv("data");

/*
 * Open the netCDF file.
 */

  sprintf( filename, "%s/cdf/meccatemp.cdf", dir );
  ncid = ncopen(filename,NC_NOWRITE);

/*
 * Get the lat/lon dimension ids so we can retrieve their lengths.
 */

  lat_id = ncdimid(ncid,"lat");
  lon_id = ncdimid(ncid,"lon");
  ncdiminq(ncid,lat_id,(char *)0,&nlat);
  ncdiminq(ncid,lon_id,(char *)0,&nlon);

/*
 * Get temperature and lat/lon ids.
 */

  t_id   = ncvarid(ncid,"t");
  lat_id = ncvarid(ncid,"lat");
  lon_id = ncvarid(ncid,"lon");

/*
 * Read in first nlat x nlon subsection of "t"
 */

  T = (float *)calloc(nlat*nlon,sizeof(float));

  start[0] = start[1] = start[2] = 0;
  count[0] = 1;
  count[1] = nlat;
  count[2] = nlon;
  ncvarget(ncid,t_id,(long const *)start,(long const *)count,T);

/*
 * Read in lat and lon coordinate arrays.
 */

  lat = (int *)calloc(nlat,sizeof(int));
  lon = (int *)calloc(nlon,sizeof(int));

  count[0] = nlat;
  ncvarget(ncid,lat_id,(long const *)start,(long const *)count,lat);
  count[0] = nlon;
  ncvarget(ncid,lon_id,(long const *)start,(long const *)count,lon);

/*
 * Close the netCDF file.
 */

  ncclose(ncid);

/*----------------------------------------------------------------------*
 *
 * Start graphics portion of code.
 *
/*----------------------------------------------------------------------*/

/*
 * Initialize HLU library and create application object.
 */
  
  app = gsn_create_app("test");

/*
 * Set up workstation resource list.
 */

  wk_rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(wk_rlist);

/* 
 * Set color map resource and open workstation.
 */

  NhlRLSetString(wk_rlist,"wkColorMap","gsdtol");
  wks = gsn_open_wks("ncgm","test", wk_rlist);

/*
 * Set up contour and scalar field resource lists.
 */

  cn_rlist = NhlRLCreate(NhlSETRL);
  sf_rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(cn_rlist);
  NhlRLClear(sf_rlist);

  printf("cn_rlist, sf_rlist %d %d\n",cn_rlist, sf_rlist);

/*
 * Set some scalar field resources.
 */

  NhlRLSetIntegerArray(sf_rlist,"sfXArray",lon,nlon);
  NhlRLSetIntegerArray(sf_rlist,"sfYArray",lat,nlat);

/*
 * Set some contour resources.
 */

  NhlRLSetString(cn_rlist, "cnFillOn",             "True");
  NhlRLSetString(cn_rlist, "cnLineLabelsOn",       "False");
  NhlRLSetString(cn_rlist, "lbPerimOn",            "False");
  NhlRLSetString(cn_rlist, "pmLabelBarDisplayMode","ALWAYS");

/*
 * Create and draw contour plot, and advance frame.
 */

  contour = gsn_contour(wks, T, nlat, nlon, sf_rlist, cn_rlist);

/*
 * NhlDestroy destroys the given id and all of its children.
 */

  NhlDestroy(wks);

/*
 * Restores state.
 */

  NhlClose();
  exit(0);
}


int gsn_create_app(char *name)
{
  int app, srlist;

/*
 * Initialize HLU library.
 */
  NhlInitialize();

/*
 * Initialize variable for holding resources.
 */

  srlist = NhlRLCreate(NhlSETRL);

/*
 * Create Application object.
 */

  NhlRLClear(srlist);
  NhlRLSetString(srlist,"appDefaultParent","True");
  NhlRLSetString(srlist,"appUsrDir","./");
  NhlCreate(&app,name,NhlappClass,NhlDEFAULT_APP,srlist);

/*
 * Clean up and return.
 */

  NhlRLDestroy(srlist);
  return(app);
}


int gsn_open_wks(char *type, char *name, int wk_rlist)
{
  int wks, len;
  char *filename;

  if(!strcmp(type,"x11") || !strcmp(type,"X11")) {
/*
 * Create an XWorkstation object.
 */

    NhlRLSetInteger(wk_rlist,"wkPause",True);
    NhlCreate(&wks,"x11",NhlxWorkstationClass,
			  NhlDEFAULT_APP,wk_rlist);
  }
  else if(!strcmp(type,"ncgm") || !strcmp(type,"NCGM")) {
/*
 * Generate NCGM file name.
 */

    len      = strlen(name);
    filename = (char *)calloc(len+6,sizeof(char));

    strncpy(filename,name,len);
    strcat(filename,".ncgm");
/*
 * Create a meta file object.
 */

    NhlRLSetString(wk_rlist,"wkMetaName",filename);
    NhlCreate(&wks,"ncgm",NhlncgmWorkstationClass,NhlDEFAULT_APP,wk_rlist);
  }
  else if(!strcmp(type,"ps") || !strcmp(type,"PS")) {
/*
 * Generate PS file name.
 */

    len      = strlen(name);
    filename = (char *)calloc(len+4,sizeof(char));

    strncpy(filename,name,len);
    strcat(filename,".ps");

/*
 * Create a PS workstation.
 */

    NhlRLSetString(wk_rlist,"wkPSFileName",filename);
    NhlCreate(&wks,"ps",NhlpsWorkstationClass,NhlDEFAULT_APP,wk_rlist);
  }
  else if(!strcmp(type,"pdf") || !strcmp(type,"PDF")) {
/*
 * Generate PDF file name.
 */

    len      = strlen(name);
    filename = (char *)calloc(len+4,sizeof(char));

    strncpy(filename,name,len);
    strcat(filename,".pdf");

/*
 * Create a PDF workstation.
 */

    NhlRLSetString(wk_rlist,"wkPDFFileName",filename);
    NhlCreate(&wks,"pdf",NhlpdfWorkstationClass,NhlDEFAULT_APP,wk_rlist);
  }
  else {
    printf("Invalid workstation type, must be 'x11', 'ncgm', 'ps', or 'pdf'\n");
  }

/*
 * Clean up and return.
 */

  free(filename);
  return(wks);
}

int gsn_contour(int wks, float *data, int ylen, int xlen, int sf_rlist,
				int cn_rlist)
{
  int field, contour, length[2], app, srlist;

/*
 * Retrieve application id.
 */

  app = NhlAppGetDefaultParentId();

/*
 * Create a scalar field object that will be used as the
 * dataset for the contour object.
 */

  length[0] = ylen;
  length[1] = xlen;

  NhlRLSetMDFloatArray(sf_rlist,"sfDataArray",data,2,(int *)length);
  NhlCreate(&field,"field",NhlscalarFieldClass,app,sf_rlist);

/*
 * Assign the data object that was created earlier.
 */

  NhlRLSetInteger(cn_rlist,"cnScalarFieldData",field);

/*
 * Create plot.
 */

  NhlCreate(&contour,"contour",NhlcontourPlotClass,wks,cn_rlist);

/*
 * Draw plot and advance frame.
 */

  NhlDraw(contour);
  NhlFrame(wks);

/*
 * Return.
 */

  return(contour);
}


