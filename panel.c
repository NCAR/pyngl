#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gsun.h"

#define NCOLORS 17
#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))
#define NG  5

void create_title(char *title, nglPlotId *parray, nglPlotId *parray_copy,
                  int nplots, int *panel_dims,int ndims,
                  const char *extra_string, int row_spec)
{
  int i, first_time;
  char title2[10];

  if(row_spec) {
    sprintf(title,":F25:RowSpec = %d", panel_dims[0]);
    for(i = 1; i < ndims; i++ ) {
      sprintf(title2,", %d", panel_dims[i]);
      strcat(title,title2);
    }
  }
  else {
    sprintf(title,":F25:%d rows/%d columns", panel_dims[0], panel_dims[1]);
  }

  first_time = 1;
  for(i = 0; i < nplots; i++) {
    if(parray[i].base == 0) {
      if(first_time) {
        sprintf(title2,":C:plots %d", i);
        strcat(title,title2);
        first_time = 0;
      }
      else {
        sprintf(title2,", %d", i);
        strcat(title,title2);
      }
      parray[i].base = parray_copy[i].base;
    }
  }
  if(!first_time) {
    sprintf(title2," are missing");
    strcat(title,title2);
  }

  if(extra_string != NULL && strcmp(extra_string,"")) {
    sprintf(title2,":C:%s", extra_string);
    strcat(title,title2);
  }
}

main()
{
/*
 * Declare variables for the HLU routine calls.
 */

  int wks;
  nglPlotId text, *carray, *carray_copy, *varray, *varray_copy;
  int wk_rlist, sf_rlist, vf_rlist, tx_rlist, cn_rlist, vc_rlist;
  int srlist, lb_rlist, cmap_len[2];
  float *xf, *yf, cmap[NCOLORS][3];
  int nplots, panel_dims[2], first_time, *row_spec;
  nglRes special_res, special_pres, special_tres;
  int vccolors[]  = {2,16,30,44,58,52,86,100,114,128,142,156,170};

/*
 * Declare variables for getting information from netCDF file.
 */

  int ncid_T2, ncid_U, ncid_V, id_T2, id_U, id_V;
  int lonid_T2, latid_T2, timeid_T2;
  int nlat_T2, nlon_T2, ntime_T2, nlatlon_T2, ntimelatlon_T2;
  int ndims_T2;
  int is_missing_T2, is_missing_U, is_missing_V;
  int attid, status;

/*
 * Declare variables to hold values, missing values, and types that
 * are read in from netCDF files.
 */
  void  *T2, *U, *V, *lat_T2, *lon_T2, *time_T2;
  void  *T2new, *Unew, *Vnew, *FillValue_T2, *FillValue_U, *FillValue_V;
  int is_lat_coord_T2, is_lon_coord_T2;
  int is_time_coord_T2;

  nc_type nctype_T2, nctype_lat_T2, nctype_lon_T2, nctype_time_T2;
  char type_T2[TYPE_LEN], type_lat_T2[TYPE_LEN], type_lon_T2[TYPE_LEN];
  char type_time_T2[TYPE_LEN];

  size_t i, j, *start, *count;
  char  filename_U[256], filename_V[256], filename_T2[256], title[100];
  const char *dir = _NGGetNCARGEnv("data");

  float xmsg, ymsg;
  int nlines, *indices;

  extern void create_title(char *, nglPlotId *, nglPlotId *, int, int *,
                           int, const char *, int);

  xmsg = ymsg = -999;

/*
 * Open the netCDF files for contour and vector data.
 */

  sprintf(filename_T2, "%s/cdf/Tstorm.cdf", dir );
  sprintf(filename_U,  "%s/cdf/Ustorm.cdf", dir );
  sprintf(filename_V,  "%s/cdf/Vstorm.cdf", dir );

  nc_open(filename_T2,NC_NOWRITE,&ncid_T2);
  nc_open(filename_U, NC_NOWRITE,&ncid_U);
  nc_open(filename_V, NC_NOWRITE,&ncid_V);

/*
 * Get the lat/lon/time dimension ids so we can retrieve their lengths.
 */

  nc_inq_dimid(ncid_T2,"lat",&latid_T2);
  nc_inq_dimid(ncid_T2,"lon",&lonid_T2);
  nc_inq_dimid(ncid_T2,"timestep",&timeid_T2);

  nc_inq_dimlen(ncid_T2,latid_T2,(size_t*)&nlat_T2);  
  nc_inq_dimlen(ncid_T2,lonid_T2,(size_t*)&nlon_T2);
  nc_inq_dimlen(ncid_T2,timeid_T2,(size_t*)&ntime_T2);

  nlatlon_T2     = nlat_T2 * nlon_T2;
  ntimelatlon_T2 = ntime_T2 * nlatlon_T2;

/*
 * Get temperature, u, v, lat, and lon ids.
 */

  nc_inq_varid(ncid_T2, "t",&id_T2);
  nc_inq_varid(ncid_U,  "u",&id_U);
  nc_inq_varid(ncid_V,  "v",&id_V);
  nc_inq_varid(ncid_T2,"lat",&latid_T2);
  nc_inq_varid(ncid_T2,"lon",&lonid_T2);
  nc_inq_varid(ncid_T2,"timestep",&timeid_T2);

/*
 * Check if T2, U, V have _FillValue attributes set.  If so,
 * retrieve them later.
 */

  status = nc_inq_attid (ncid_T2, id_T2, "_FillValue", &attid); 
  if(status == NC_NOERR) {
    is_missing_T2 = 1;
  }
  else {
    is_missing_T2 = 0;
  }

  status = nc_inq_attid (ncid_U, id_U, "_FillValue", &attid); 
  if(status == NC_NOERR) {
    is_missing_U = 1;
  }
  else {
    is_missing_U = 0;
  }

  status = nc_inq_attid (ncid_V, id_V, "_FillValue", &attid); 
  if(status == NC_NOERR) {
    is_missing_V = 1;
  }
  else {
    is_missing_V = 0;
  }


/*
 * Get type and number of dimensions of T2, then read in first
 * ntime_T2 x nlat_T2 x nlon_T2 subsection of "t", "u" and "v".
 * Also, read in missing values if ones are set.
 */

  nc_inq_vartype  (ncid_T2, id_T2, &nctype_T2);
  nc_inq_varndims (ncid_T2, id_T2, &ndims_T2);

  start = (size_t *)malloc(ndims_T2*sizeof(size_t));
  count = (size_t *)malloc(ndims_T2*sizeof(size_t));
  for(i = 0; i < ndims_T2; i++)   start[i] = 0;
  for(i = 0; i < ndims_T2-3; i++) count[i] = 1;

  count[ndims_T2-3] = ntime_T2;
  count[ndims_T2-2] = nlat_T2;
  count[ndims_T2-1] = nlon_T2;

  switch(nctype_T2) {

  case NC_DOUBLE:
    strcpy(type_T2,"double");
    T2      = (double *)malloc(ntimelatlon_T2*sizeof(double));
/*
 * Get double values.
 */
    nc_get_vara_double(ncid_T2,id_T2,start,count,(double*)T2);

/*
 * Get double missing value.
 */
    if(is_missing_T2) {
      FillValue_T2 = (int *)malloc(sizeof(int));
      nc_get_att_double (ncid_T2, id_T2, "_FillValue", (double*)FillValue_T2); 
    }
/*
 * Convert from K to F.
 */
    for(i = 0; i < ntimelatlon_T2; i++) {
      if(!is_missing_T2 ||
         (is_missing_T2 && ((double*)T2)[i] != ((double*)FillValue_T2)[0])) {
        ((double*)T2)[i] = (((double*)T2)[i]-273.15)*(9./5.) + 32.;
      }
    }

    break;

  case NC_FLOAT:
    strcpy(type_T2,"float");
    T2      = (float *)malloc(ntimelatlon_T2*sizeof(float));

/*
 * Get float values.
 */
    nc_get_vara_float(ncid_T2,id_T2,start,count,(float*)T2);

/*
 * Get float missing value.
 */
    if(is_missing_T2) {
      FillValue_T2 = (int *)malloc(sizeof(int));
      nc_get_att_float (ncid_T2, id_T2, "_FillValue", (float*)FillValue_T2); 
    }
/*
 * Convert from K to F.
 */
    for(i = 0; i < ntimelatlon_T2; i++) {
      if(!is_missing_T2 ||
         (is_missing_T2 && ((float*)T2)[i] != ((float*)FillValue_T2)[0])) {
        ((float*)T2)[i] = (((float*)T2)[i]-273.15)*(9./5.) + 32.;
      }
    }
    break;

  case NC_INT:
    strcpy(type_T2,"integer");
    T2      = (int *)malloc(ntimelatlon_T2*sizeof(int));

/*
 * Get integer values.
 */
    nc_get_vara_int(ncid_T2,id_T2,start,count,(int*)T2);

/*
 * Get integer missing value.
 */
    if(is_missing_T2) {
      FillValue_T2 = (int *)malloc(sizeof(int));
      nc_get_att_int (ncid_T2, id_T2, "_FillValue", (int*)FillValue_T2); 
    }
    break;
  }

  switch(nctype_T2) {

  case NC_DOUBLE:
    strcpy(type_T2,"double");
    U      = (double *)malloc(ntimelatlon_T2*sizeof(double));
/*
 * Get double values.
 */
    nc_get_vara_double(ncid_U,id_U,start,count,(double*)U);

/*
 * Get double missing value.
 */
    if(is_missing_U) {
      FillValue_U = (int *)malloc(sizeof(int));
      nc_get_att_double (ncid_U, id_U, "_FillValue", (double*)FillValue_U); 
    }
    break;

  case NC_FLOAT:
    strcpy(type_T2,"float");
    U      = (float *)malloc(ntimelatlon_T2*sizeof(float));

/*
 * Get float values.
 */
    nc_get_vara_float(ncid_U,id_U,start,count,(float*)U);

/*
 * Get float missing value.
 */
    if(is_missing_U) {
      FillValue_U = (int *)malloc(sizeof(int));
      nc_get_att_float (ncid_U, id_U, "_FillValue", (float*)FillValue_U); 
    }
    break;

  case NC_INT:
    strcpy(type_T2,"integer");
    U      = (int *)malloc(ntimelatlon_T2*sizeof(int));

/*
 * Get integer values.
 */
    nc_get_vara_int(ncid_U,id_U,start,count,(int*)U);

/*
 * Get integer missing value.
 */
    if(is_missing_U) {
      FillValue_U = (int *)malloc(sizeof(int));
      nc_get_att_int (ncid_U, id_U, "_FillValue", (int*)FillValue_U); 
    }
    break;
  }

  switch(nctype_T2) {

  case NC_DOUBLE:
    strcpy(type_T2,"double");
    V      = (double *)malloc(ntimelatlon_T2*sizeof(double));
/*
 * Get double values.
 */
    nc_get_vara_double(ncid_V,id_V,start,count,(double*)V);

/*
 * Get double missing value.
 */
    if(is_missing_V) {
      FillValue_V = (int *)malloc(sizeof(int));
      nc_get_att_double (ncid_V, id_V, "_FillValue", (double*)FillValue_V); 
    }
    break;

  case NC_FLOAT:
    strcpy(type_T2,"float");
    V      = (float *)malloc(ntimelatlon_T2*sizeof(float));

/*
 * Get float values.
 */
    nc_get_vara_float(ncid_V,id_V,start,count,(float*)V);

/*
 * Get float missing value.
 */
    if(is_missing_V) {
      FillValue_V = (int *)malloc(sizeof(int));
      nc_get_att_float (ncid_V, id_V, "_FillValue", (float*)FillValue_V); 
    }
    break;

  case NC_INT:
    strcpy(type_T2,"integer");
    V      = (int *)malloc(ntimelatlon_T2*sizeof(int));

/*
 * Get integer values.
 */
    nc_get_vara_int(ncid_V,id_V,start,count,(int*)V);

/*
 * Get integer missing value.
 */
    if(is_missing_V) {
      FillValue_V = (int *)malloc(sizeof(int));
      nc_get_att_int (ncid_V, id_V, "_FillValue", (int*)FillValue_V); 
    }
    break;
  }

/*
 * Read in lat coordinate arrays for "t", "u", and "v".
 */

  nc_inq_vartype  (ncid_T2, latid_T2, &nctype_lat_T2);

  is_lat_coord_T2 = 1;
  switch(nctype_lat_T2) {

  case NC_DOUBLE:
    strcpy(type_lat_T2,"double");
    lat_T2      = (double *)malloc(nlat_T2*sizeof(double));

/*
 * Get double values.
 */
    nc_get_var_double(ncid_T2,latid_T2,(double*)lat_T2);
    break;

  case NC_FLOAT:
    strcpy(type_lat_T2,"float");
    lat_T2      = (float *)malloc(nlat_T2*sizeof(float));

/*
 * Get float values.
 */
    nc_get_var_float(ncid_T2,latid_T2,(float*)lat_T2);

  case NC_INT:
    strcpy(type_lat_T2,"integer");
    lat_T2      = (int *)malloc(nlat_T2*sizeof(int));

/*
 * Get integer values.
 */
    nc_get_var_int(ncid_T2,latid_T2,(int*)lat_T2);
    break;
  }

/*
 * Read in lon coordinate arrays for "u" and "v".
 */

  nc_inq_vartype  (ncid_T2, lonid_T2, &nctype_lon_T2);

  is_lon_coord_T2 = 1;
  switch(nctype_lon_T2) {

  case NC_DOUBLE:
    strcpy(type_lon_T2,"double");
    lon_T2      = (double *)malloc(nlon_T2*sizeof(double));

    nc_get_var_double(ncid_T2,lonid_T2,(double*)lon_T2);
    break;

  case NC_FLOAT:
    strcpy(type_lon_T2,"float");
    lon_T2      = (float *)malloc(nlon_T2*sizeof(float));

    nc_get_var_float(ncid_T2,lonid_T2,(float*)lon_T2);

  case NC_INT:
    strcpy(type_lon_T2,"integer");
    lon_T2      = (int *)malloc(nlon_T2*sizeof(int));

    nc_get_var_int(ncid_T2,lonid_T2,(int*)lon_T2);
    break;
  }

/*
 * Read in time coordinate arrays for "u" and "v".
 */

  nc_inq_vartype  (ncid_T2, timeid_T2, &nctype_time_T2);

  is_time_coord_T2 = 1;
  switch(nctype_time_T2) {

  case NC_DOUBLE:
    strcpy(type_time_T2,"double");
    time_T2      = (double *)malloc(ntime_T2*sizeof(double));

    nc_get_var_double(ncid_T2,timeid_T2,(double*)time_T2);
    break;

  case NC_FLOAT:
    strcpy(type_time_T2,"float");
    time_T2      = (float *)malloc(ntime_T2*sizeof(float));

    nc_get_var_float(ncid_T2,timeid_T2,(float*)time_T2);

  case NC_INT:
    strcpy(type_time_T2,"integer");
    time_T2      = (int *)malloc(ntime_T2*sizeof(int));

    nc_get_var_int(ncid_T2,timeid_T2,(int*)time_T2);
    break;
  }

/*
 * Close the netCDF files.
 */

  ncclose(ncid_T2);
  ncclose(ncid_U);
  ncclose(ncid_V);


/*----------------------------------------------------------------------*
 *
 * Start graphics portion of code.
 *
 *----------------------------------------------------------------------*/
/*
 * Initialize special resources.  For the plotting routines, draw, frame,
 * and maximize default to True (1).  For primitive and text routines,
 * only draw defaults to True. For panel routines, draw and frame default
 * to True.
 */
  special_res.nglDraw     = 1;
  special_res.nglFrame    = 1;
  special_res.nglMaximize = 1;
  special_res.nglScale    = 0;
  special_res.nglDebug    = 0;

/*
 * Resources for fill colors.
 */
  special_res.nglSpreadColors     =  1;
  special_res.nglSpreadColorStart =  2;
  special_res.nglSpreadColorEnd   = -1;

  special_tres.nglDraw    = 1;
  special_tres.nglFrame   = 0;
  special_tres.nglMaximize= 0;
  special_tres.nglDebug   = 0;

/*
 * Paper orientation: -1 is auto, 0 is portrait, and 6 is landscape.
 */
  special_pres.nglPaperOrientation = -1;
  special_pres.nglPaperWidth       =  8.5;
  special_pres.nglPaperHeight      = 11.0;
  special_pres.nglPaperMargin      =  0.5;

/*
 * Special resources for paneling. These are the default values.
 */
  special_pres.nglPanelSave               = 0;
  special_pres.nglDebug                   = 0;
  special_pres.nglPanelCenter             = 1;
  special_pres.nglPanelRowSpec            = 0;
  special_pres.nglPanelXWhiteSpacePercent = 1.;
  special_pres.nglPanelYWhiteSpacePercent = 1.;
  special_pres.nglPanelBoxes              = 0;
  special_pres.nglPanelLeft               = 0.;
  special_pres.nglPanelRight              = 1.;
  special_pres.nglPanelBottom             = 0.;
  special_pres.nglPanelTop                = 1.;
  special_pres.nglPanelInvsblTop          = -999;
  special_pres.nglPanelInvsblLeft         = -999;
  special_pres.nglPanelInvsblRight        = -999;
  special_pres.nglPanelInvsblBottom       = -999;

/*
 * Default settings for PanelLabelBar resources.
 */
  special_pres.nglPanelLabelBar               = 0;  
  special_pres.nglPanelLabelBarPerimOn        = 0; 
  special_pres.nglPanelLabelBarAlignment      = NhlINTERIOREDGES;
  special_pres.nglPanelLabelBarOrientation    = NhlHORIZONTAL;
  special_pres.nglPanelLabelBarXF             = -999.;
  special_pres.nglPanelLabelBarYF             = -999.;
  special_pres.nglPanelLabelBarWidthF         = -999.;
  special_pres.nglPanelLabelBarHeightF        = -999.;
  special_pres.nglPanelLabelBarOrthogonalPosF = -999;
  special_pres.nglPanelLabelBarParallelPosF   = -999;

/*
 * This resource isn't recognized yet.
 */
  special_pres.nglMaximize = 1;
  special_pres.nglFrame    = 1;
  special_pres.nglDraw     = 1;
    
/*
 * Initialize color map for later.
 */
   cmap[ 0][0] = 1.00;   cmap[ 0][1] = 1.00;   cmap[ 0][2] = 1.00;
   cmap[ 1][0] = 0.00;   cmap[ 1][1] = 0.00;   cmap[ 1][2] = 0.00;
   cmap[ 2][0] = .560;   cmap[ 2][1] = .500;   cmap[ 2][2] = .700;
   cmap[ 3][0] = .300;   cmap[ 3][1] = .300;   cmap[ 3][2] = .700;
   cmap[ 4][0] = .100;   cmap[ 4][1] = .100;   cmap[ 4][2] = .700;
   cmap[ 5][0] = .000;   cmap[ 5][1] = .100;   cmap[ 5][2] = .700;
   cmap[ 6][0] = .000;   cmap[ 6][1] = .300;   cmap[ 6][2] = .700;
   cmap[ 7][0] = .000;   cmap[ 7][1] = .500;   cmap[ 7][2] = .500;
   cmap[ 8][0] = .000;   cmap[ 8][1] = .700;   cmap[ 8][2] = .100;
   cmap[ 9][0] = .060;   cmap[ 9][1] = .680;   cmap[ 9][2] = .000;
   cmap[10][0] = .550;   cmap[10][1] = .550;   cmap[10][2] = .000;
   cmap[11][0] = .570;   cmap[11][1] = .420;   cmap[11][2] = .000;
   cmap[12][0] = .700;   cmap[12][1] = .285;   cmap[12][2] = .000;
   cmap[13][0] = .700;   cmap[13][1] = .180;   cmap[13][2] = .000;
   cmap[14][0] = .870;   cmap[14][1] = .050;   cmap[14][2] = .000;
   cmap[15][0] = 1.00;   cmap[15][1] = .000;   cmap[15][2] = .000;
   cmap[16][0] = .800;   cmap[16][1] = .800;   cmap[16][2] = .800;

/*
 * Set up workstation resource list.
 */

  wk_rlist = NhlRLCreate(NhlSETRL);
  wks = ngl_open_wks_wrap("x11","panel", wk_rlist);

/*
 * Initialize and clear resource lists.
 */

  srlist   = NhlRLCreate(NhlSETRL);
  sf_rlist = NhlRLCreate(NhlSETRL);
  vf_rlist = NhlRLCreate(NhlSETRL);
  tx_rlist = NhlRLCreate(NhlSETRL);
  cn_rlist = NhlRLCreate(NhlSETRL);
  vc_rlist = NhlRLCreate(NhlSETRL);
  lb_rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(sf_rlist);
  NhlRLClear(vf_rlist);
  NhlRLClear(tx_rlist);
  NhlRLClear(cn_rlist);
  NhlRLClear(vc_rlist);
  NhlRLClear(srlist);

/*
 * Set some colormap resources.
 */
  if(!special_res.nglSpreadColors) {
    cmap_len[0] = NCOLORS;
    cmap_len[1] = 3;
    NhlRLSetMDFloatArray(srlist,NhlNwkColorMap,&cmap[0][0],2,cmap_len);
  }
  else {
    NhlRLSetString(srlist,"wkColorMap","rainbow");
  }
  (void)NhlSetValues(wks, srlist);

/*
 * Set some contour resources.
 */
  NhlRLSetString (cn_rlist,"pmLabelBarDisplayMode"  , "Never");
  NhlRLSetString (cn_rlist,"pmLabelBarSide"         , "Bottom");
  NhlRLSetString (cn_rlist,"lbOrientation"          , "Horizontal");
  NhlRLSetString (cn_rlist,"lbPerimOn"              , "False");
  NhlRLSetInteger(cn_rlist,"lbTitleFont"            , 25);
  NhlRLSetString (cn_rlist,"lbTitleString"          , "temperature (:S:o:N:F)");
  NhlRLSetFloat  (cn_rlist,"tiMainFontHeightF"      , 0.03);
  NhlRLSetString (cn_rlist,"tiMainString"           , ":F25:Wind velocity vectors");
  
  NhlRLSetString (cn_rlist,"cnLevelSelectionMode"   , "ManualLevels");
  NhlRLSetFloat  (cn_rlist,"cnLevelSpacingF"        ,  10.);
  NhlRLSetFloat  (cn_rlist,"cnMinLevelValF"         , -20.);
  NhlRLSetFloat  (cn_rlist,"cnMaxLevelValF"         ,  80.);
  NhlRLSetFloat  (cn_rlist,"cnLevelSpacingF"        ,  10.);
  NhlRLSetString (cn_rlist,"cnFillOn"               , "True");

/*
 * Set some vector resources.
 */
  NhlRLSetIntegerArray(vc_rlist, "vcLevelColors"         , vccolors,13);

  NhlRLSetFloat  (vc_rlist, "vcRefLengthF",         0.045);
  NhlRLSetFloat  (vc_rlist, "vcRefMagnitudeF",      20.0);
  NhlRLSetFloat  (vc_rlist, "vcMinMagnitudeF",      0.001);
  NhlRLSetString (vc_rlist, "vcFillArrowsOn",       "True");
  NhlRLSetString (vc_rlist, "vcMonoFillArrowFillColor", "False");
  NhlRLSetFloat  (vc_rlist, "vcMinFracLengthF",      0.33);
  NhlRLSetString (vc_rlist, "pmLabelBarDisplayMode", "Never"); 
  NhlRLSetString (vc_rlist, "vcLevelSelectionMode",  "ManualLevels");
  NhlRLSetFloat  (vc_rlist, "vcLevelSpacingF", 2.0);
  NhlRLSetFloat  (vc_rlist, "vcMinLevelValF",  0.0);
  NhlRLSetFloat  (vc_rlist, "vcMaxLevelValF",  20.0);

/*
 * Loop across time dimension and create a contour and vector plot for
 * each one. Then, we'll create several panel plots.
 *
 * Also, create a separate array to hold the same plots, but set some of
 * them to "0" (missing).
 */
  carray      = (nglPlotId*)malloc(ntime_T2*sizeof(nglPlotId));
  carray_copy = (nglPlotId*)malloc(ntime_T2*sizeof(nglPlotId));
  varray      = (nglPlotId*)malloc(ntime_T2*sizeof(nglPlotId));
  varray_copy = (nglPlotId*)malloc(ntime_T2*sizeof(nglPlotId));

  special_res.nglMaximize = 0;
  special_res.nglFrame    = 0;
  special_res.nglDraw     = 0;
/*
  for(i = 0; i < ntime_T2; i++) { 
 */
  for(i = 0; i < 17; i++) {
    T2new = &((float*)T2)[nlatlon_T2*i];
    Unew  = &((float*)U)[nlatlon_T2*i];
    Vnew  = &((float*)V)[nlatlon_T2*i];
    special_res.nglSpreadColors = 1;
    carray[i] = ngl_contour_wrap(wks, T2new, type_T2, nlat_T2, nlon_T2, 
                                 is_lat_coord_T2, lat_T2, 
                                 type_lat_T2, is_lon_coord_T2, 
                                 lon_T2, type_lon_T2, 
                                 is_missing_T2, FillValue_T2, 
                                 sf_rlist, cn_rlist, &special_res);
    

    special_res.nglSpreadColors = 0;
    /*
    varray[i] = ngl_vector_wrap(wks, Unew, Vnew, type_T2, type_T2, 
                                nlat_T2, nlon_T2, is_lat_coord_T2, lat_T2,
                                type_lat_T2, is_lon_coord_T2, lon_T2,
                                type_lon_T2, is_missing_U, is_missing_V, 
                                FillValue_U, FillValue_V,
                                vf_rlist, vc_rlist, &special_res);
    */
    varray[i] = ngl_vector_scalar_wrap(wks, Unew, Vnew, T2new, type_T2,
                                       type_T2, type_T2, nlat_T2, nlon_T2, 
                                       is_lat_coord_T2, lat_T2,
                                       type_lat_T2, is_lon_coord_T2,
                                       lon_T2, type_lon_T2, is_missing_U,
                                       is_missing_V, is_missing_T2, 
                                       FillValue_U, FillValue_V, 
                                       FillValue_T2, vf_rlist, sf_rlist,
                                       vc_rlist, &special_res);

    carray_copy[i] = carray[i];
    varray_copy[i] = varray[i];
  }
/*
 * Initialize stuff for text string.
 */
  xf = (float*)malloc(sizeof(float));
  yf = (float*)malloc(sizeof(float));
  *xf = 0.5;
  *yf = 0.5;

  NhlRLClear(tx_rlist);
  NhlRLSetFloat(tx_rlist,"txFontHeightF" , 0.03);
  NhlRLSetString(tx_rlist,"txBackgroundFillColor","white");
  NhlRLSetString(tx_rlist,"txPerimOn","True");

/*
 * Begin panel plots. Set nglPanelLabelBar to True (default is False);
 */
  nplots = 6;

  special_pres.nglPanelLabelBar = 1;
  special_pres.nglPanelRowSpec = 0;
  panel_dims[0] = 3;
  panel_dims[1] = 2;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);

/*
 * Tests for gsnPanelRowSpec.
 */

  row_spec = (int*)malloc(3*sizeof(int));
  row_spec[0] = 3;
  row_spec[1] = 2;
  row_spec[2] = 1;

  special_pres.nglPanelCenter = 1;
  special_pres.nglPanelRowSpec = 1;
  special_pres.nglFrame = 0;
  special_tres.nglFrame = 1;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, &carray[0], nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  special_pres.nglPanelCenter = 1;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[5].base = 0;
  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[5].base = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[1].base = 0;
  carray[2].base = 0;
  special_pres.nglPanelCenter = 1;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[1].base = 0;
  carray[2].base = 0;
  special_pres.nglPanelCenter = 0;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[3].base = 0;
  carray[4].base = 0;
  special_pres.nglPanelCenter = 1;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[3].base = 0;
  carray[4].base = 0;
  special_pres.nglPanelCenter = 0;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[1].base = 0;
  carray[2].base = 0;
  carray[3].base = 0;
  carray[4].base = 0;
  special_pres.nglPanelCenter = 1;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[1].base = 0;
  carray[2].base = 0;
  carray[3].base = 0;
  carray[4].base = 0;
  special_pres.nglPanelCenter = 0;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Start with a single row of plots, then make the first plot and the last
 * plot missing.
 */
  special_pres.nglPanelRowSpec = 0;
  panel_dims[0] = 1;
  panel_dims[1] = 3;
  nplots = panel_dims[0] * panel_dims[1];

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[2].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Single column of plots, then make the first plot and the last
 * plot missing.
 */
  panel_dims[0] = 3;          /* rows    */
  panel_dims[1] = 1;          /* columns */
  nplots = panel_dims[0] * panel_dims[1];
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[2].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * 2 rows x 3 columns of plots. Then make the top, bottom, right,
 * and left rows missing.
 */
  panel_dims[0] = 2;          /* rows    */
  panel_dims[1] = 3;          /* columns */
  nplots = panel_dims[0] * panel_dims[1];
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[1].base = 0;
  carray[2].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[3].base = 0;
  carray[4].base = 0;
  carray[5].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[3].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[2].base = 0;
  carray[5].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * 3 rows x 2 columns of plots. Then make the top, bottom, right,
 * and left rows missing.
 */
  panel_dims[0] = 3;          /* rows    */
  panel_dims[1] = 2;          /* columns */
  nplots = panel_dims[0] * panel_dims[1];
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[1].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[2].base = 0;
  carray[3].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[4].base = 0;
  carray[5].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[2].base = 0;
  carray[4].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[1].base = 0;
  carray[3].base = 0;
  carray[5].base = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * 3 rows x 2 columns of plots, but only pass in 4 plots, then 2, then 1.
 */
  panel_dims[0] = 3;          /* rows    */
  panel_dims[1] = 2;          /* columns */
  nplots = 4;
  ngl_panel_wrap(wks, carray, 4, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "4 plots passed in",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[1].base = 0;
  ngl_panel_wrap(wks, carray, 4, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "4 plots passed in",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  nplots = 2;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "2 plots passed in",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  nplots = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "1 plot passed in",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * 3 rows x 4 columns of plots, but only pass in 8, 5, 3, then 1 plot.
 * On this one, we alternate between setting nglPanelCenter to True
 * and False.
 */

  panel_dims[0] = 3;          /* rows    */
  panel_dims[1] = 4;          /* columns */
  nplots = 8;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "8 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "8 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);
/*
 * Same thing as above, only with some missing plots thrown in.
 */
  carray[0].base = 0;
  carray[1].base = 0;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "8 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[1].base = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "8 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Now do 5 plots.
 */
  nplots = 5;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "5 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "5 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Same thing as above, only with some missing plots thrown in.
 */
  carray[3].base = 0;
  carray[4].base = 0;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "5 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[3].base = 0;
  carray[4].base = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "5 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Now 3.
 */
  nplots = 3;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "3 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "3 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Same thing as above, only with some missing plots thrown in.
 */ 
  carray[0].base = 0;
  carray[1].base = 0;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "3 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].base = 0;
  carray[1].base = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "3 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * One plot
 */
  nplots = 1;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "1 plot passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, carray_copy, nplots, panel_dims,2,
                       "1 plot passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * NhlDestroy destroys the given id and all of its children.
 */

  NhlDestroy(wks);

/*
 * Restores state.
 */

  NhlClose();
/*
 * Free up memory.
 */
  free(T2);
  free(lat_T2);
  free(lon_T2);
  exit(0);

}
