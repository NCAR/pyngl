#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gsun.h"

#define NCOLORS 17
#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))
#define NG  5

char *create_title(int *parray, int *parray_copy, int nplots, int panel_dims[2])
{
  int i, first_time;
  char *title, title2[10];

  title = (char *)malloc(100*sizeof(char));

  first_time = 1;
  for(i = 0; i < nplots; i++) {
    if(parray[i] == 0) {
      if(first_time) {
        sprintf(title,":F25:%d rows/%d columns:C:plots %d",
                panel_dims[0], panel_dims[1], i);
        first_time = 0;
      }
      else {
        sprintf(title2,", %d", i);
        strcat(title,title2);
      }
      parray[i] = parray_copy[i];
    }
  }
  if(!first_time) {
    sprintf(title2," are missing");
    strcat(title,title2);
  }
  else {
    sprintf(title,":F25:%d rows/%d columns",panel_dims[0], panel_dims[1]);
  }
  return(title);
}

main()
{
/*
 * Declare variables for the HLU routine calls.
 */

  int wks, text, *varray, *varray_copy;
  int wk_rlist, sf_rlist, vf_rlist, tx_rlist, vc_rlist;
  int srlist, cmap_len[2];
  float *xf, *yf, cmap[NCOLORS][3];
  int nplots, panel_dims[2], first_time;
  nglRes special_res, special_pres, special_tres;

/*
 * Declare variables for getting information from netCDF file.
 */

  int ncid_U, ncid_V, ncid_T2;
  int id_U, id_V, id_T2;
  int lonid_UV, latid_UV, timeid_UV;
  int nlat_UV, nlon_UV, ntime_UV, nlatlon_UV, ntimelatlon_UV;
  int ndims_UV;
  int is_missing_U, is_missing_V, is_missing_T2;
  int attid, status;

/*
 * Declare variables to hold values, missing values, and types that
 * are read in from netCDF files.
 */
  void  *U, *V, *T2, *lat_UV, *lon_UV, *time_UV;
  void  *Unew, *Vnew, *T2new;
  void  *FillValue_T2, *FillValue_U, *FillValue_V;
  int is_lat_coord_UV, is_lon_coord_UV;
  int is_time_coord_UV;

  nc_type nctype_U, nctype_V, nctype_T2;
  nc_type nctype_lat_UV, nctype_lon_UV, nctype_time_UV;
  char type_T2[TYPE_LEN], type_U[TYPE_LEN], type_V[TYPE_LEN];
  char type_lat_UV[TYPE_LEN], type_lon_UV[TYPE_LEN];
  char type_time_UV[TYPE_LEN];

  size_t i, j, *start, *count;
  char  filename_T2[256], filename_U[256], filename_V[256];
  char *title;
  const char *dir = _NGGetNCARGEnv("data");

  float xmsg, ymsg;
  int nlines, *indices;

  extern char *create_title(int *, int *, int, int *);

  xmsg = ymsg = -999;

/*
 * Open the netCDF files for contour and vector data.
 */

  sprintf(filename_U, "%s/cdf/Ustorm.cdf", dir );
  sprintf(filename_V, "%s/cdf/Vstorm.cdf", dir );
  sprintf(filename_T2, "%s/cdf/Tstorm.cdf", dir );

  nc_open(filename_U,NC_NOWRITE,&ncid_U);
  nc_open(filename_V,NC_NOWRITE,&ncid_V);
  nc_open(filename_T2,NC_NOWRITE,&ncid_T2);

/*
 * Get the lat/lon/time dimension ids so we can retrieve their lengths.
 * The lat/lon/time arrays for the U, V, and T2 data files are the same,
 * so we only need to retrieve one set of them.
 */

  nc_inq_dimid(ncid_U,"lat",&latid_UV);
  nc_inq_dimid(ncid_U,"lon",&lonid_UV);
  nc_inq_dimid(ncid_U,"timestep",&timeid_UV);

  nc_inq_dimlen(ncid_U,latid_UV,(size_t*)&nlat_UV);  
  nc_inq_dimlen(ncid_U,lonid_UV,(size_t*)&nlon_UV);
  nc_inq_dimlen(ncid_U,timeid_UV,(size_t*)&ntime_UV);

  nlatlon_UV     = nlat_UV * nlon_UV;
  ntimelatlon_UV = ntime_UV * nlatlon_UV;

/*
 * Get temperature, u, v, lat, and lon ids.
 */

  nc_inq_varid(ncid_U,  "u",&id_U);
  nc_inq_varid(ncid_V,  "v",&id_V);
  nc_inq_varid(ncid_T2, "t",&id_T2);
  nc_inq_varid(ncid_U,"lat",&latid_UV);
  nc_inq_varid(ncid_U,"lon",&lonid_UV);
  nc_inq_varid(ncid_U,"timestep",&timeid_UV);

/*
 * Check if T, U, V, or T2 has a _FillValue attribute set.  If so,
 * retrieve them later.
 */

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

  status = nc_inq_attid (ncid_T2, id_T2, "_FillValue", &attid); 
  if(status == NC_NOERR) {
    is_missing_T2 = 1;
  }
  else {
    is_missing_T2 = 0;
  }


/*
 * Get type and number of dimensions of U, V, and T2, then read in first
 * ntime_UV x nlat_UV x nlon_UV subsection of "u", "v", and "t".
 * Also, read in missing values if ones are set.
 *
 * U, V, and T2 are assumed to have the same dimensions.
 */

  nc_inq_vartype  (ncid_U, id_U, &nctype_U);
  nc_inq_vartype  (ncid_V, id_V, &nctype_V);
  nc_inq_vartype  (ncid_T2, id_T2, &nctype_T2);
  nc_inq_varndims (ncid_U, id_U, &ndims_UV);

  start = (size_t *)malloc(ndims_UV*sizeof(size_t));
  count = (size_t *)malloc(ndims_UV*sizeof(size_t));
  for(i = 0; i < ndims_UV; i++)   start[i] = 0;
  for(i = 0; i < ndims_UV-3; i++) count[i] = 1;
  count[ndims_UV-3] = ntime_UV;
  count[ndims_UV-2] = nlat_UV;
  count[ndims_UV-1] = nlon_UV;

  switch(nctype_U) {

  case NC_DOUBLE:
    strcpy(type_U,"double");
    U      = (double *)malloc(ntimelatlon_UV*sizeof(double));

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
    strcpy(type_U,"float");
    U      = (float *)malloc(ntimelatlon_UV*sizeof(float));

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
    strcpy(type_U,"integer");
    U      = (int *)malloc(ntimelatlon_UV*sizeof(int));

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

  switch(nctype_V) {

  case NC_DOUBLE:
    strcpy(type_V,"double");
    V      = (double *)malloc(ntimelatlon_UV*sizeof(double));

/*
 * Get double values.
 */
    nc_get_vara_double(ncid_V,id_V,start,count,(double*)V);

/*
 * Get double missing value.
 */
    if(is_missing_V) {
      FillValue_V = (int *)malloc(sizeof(int));
      nc_get_att_float (ncid_V, id_V, "_FillValue", (float*)FillValue_V); 
    }
    break;

  case NC_FLOAT:
    strcpy(type_V,"float");
    V      = (float *)malloc(ntimelatlon_UV*sizeof(float));

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
    strcpy(type_V,"integer");
    V      = (int *)malloc(ntimelatlon_UV*sizeof(int));

/*
 * Get integer values.
 */
    nc_get_vara_int(ncid_V,id_V,start,count,(int*)V);

/*
 * Get integer missing value.
 */
    if(is_missing_V) {
      FillValue_V = (int *)malloc(sizeof(int));
      nc_get_att_float (ncid_V, id_V, "_FillValue", (float*)FillValue_V); 
    }
    break;
  }

  switch(nctype_T2) {

  case NC_DOUBLE:
    strcpy(type_T2,"double");
    T2      = (double *)malloc(ntimelatlon_UV*sizeof(double));
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
    for(i = 0; i < ntimelatlon_UV; i++) {
      if(!is_missing_T2 ||
         (is_missing_T2 && ((double*)T2)[i] != ((double*)FillValue_T2)[0])) {
        ((double*)T2)[i] = (((double*)T2)[i]-273.15)*(9./5.) + 32.;
      }
    }

    break;

  case NC_FLOAT:
    strcpy(type_T2,"float");
    T2      = (float *)malloc(ntimelatlon_UV*sizeof(float));

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
    for(i = 0; i < ntimelatlon_UV; i++) {
      if(!is_missing_T2 ||
         (is_missing_T2 && ((float*)T2)[i] != ((float*)FillValue_T2)[0])) {
        ((float*)T2)[i] = (((float*)T2)[i]-273.15)*(9./5.) + 32.;
      }
    }
    break;

  case NC_INT:
    strcpy(type_T2,"integer");
    T2      = (int *)malloc(ntimelatlon_UV*sizeof(int));

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

/*
 * Read in lat coordinate arrays for "u", "v", and "t".
 */

  nc_inq_vartype  (ncid_U, latid_UV, &nctype_lat_UV);

  is_lat_coord_UV = 1;
  switch(nctype_lat_UV) {

  case NC_DOUBLE:
    strcpy(type_lat_UV,"double");
    lat_UV      = (double *)malloc(nlat_UV*sizeof(double));

/*
 * Get double values.
 */
    nc_get_var_double(ncid_U,latid_UV,(double*)lat_UV);
    break;

  case NC_FLOAT:
    strcpy(type_lat_UV,"float");
    lat_UV      = (float *)malloc(nlat_UV*sizeof(float));

/*
 * Get float values.
 */
    nc_get_var_float(ncid_U,latid_UV,(float*)lat_UV);

  case NC_INT:
    strcpy(type_lat_UV,"integer");
    lat_UV      = (int *)malloc(nlat_UV*sizeof(int));

/*
 * Get integer values.
 */
    nc_get_var_int(ncid_U,latid_UV,(int*)lat_UV);
    break;
  }

/*
 * Read in lon coordinate arrays for "u" and "v".
 */

  nc_inq_vartype  (ncid_U, lonid_UV, &nctype_lon_UV);

  is_lon_coord_UV = 1;
  switch(nctype_lon_UV) {

  case NC_DOUBLE:
    strcpy(type_lon_UV,"double");
    lon_UV      = (double *)malloc(nlon_UV*sizeof(double));

    nc_get_var_double(ncid_U,lonid_UV,(double*)lon_UV);
    break;

  case NC_FLOAT:
    strcpy(type_lon_UV,"float");
    lon_UV      = (float *)malloc(nlon_UV*sizeof(float));

    nc_get_var_float(ncid_U,lonid_UV,(float*)lon_UV);

  case NC_INT:
    strcpy(type_lon_UV,"integer");
    lon_UV      = (int *)malloc(nlon_UV*sizeof(int));

    nc_get_var_int(ncid_U,lonid_UV,(int*)lon_UV);
    break;
  }

/*
 * Read in time coordinate arrays for "u" and "v".
 */

  nc_inq_vartype  (ncid_U, timeid_UV, &nctype_time_UV);

  is_time_coord_UV = 1;
  switch(nctype_time_UV) {

  case NC_DOUBLE:
    strcpy(type_time_UV,"double");
    time_UV      = (double *)malloc(ntime_UV*sizeof(double));

    nc_get_var_double(ncid_U,timeid_UV,(double*)time_UV);
    break;

  case NC_FLOAT:
    strcpy(type_time_UV,"float");
    time_UV      = (float *)malloc(ntime_UV*sizeof(float));

    nc_get_var_float(ncid_U,timeid_UV,(float*)time_UV);

  case NC_INT:
    strcpy(type_time_UV,"integer");
    time_UV      = (int *)malloc(ntime_UV*sizeof(int));

    nc_get_var_int(ncid_U,timeid_UV,(int*)time_UV);
    break;
  }

/*
 * Close the netCDF files.
 */

  ncclose(ncid_U);
  ncclose(ncid_V);
  ncclose(ncid_T2);


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
 * Special resources for paneling.
 */
  special_pres.nglDebug                   = 1;
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
  special_pres.nglPanelSave               = 0;
    
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
  wks = ngl_open_wks_wrap("x11","atest", wk_rlist);

/*
 * Initialize and clear resource lists.
 */

  sf_rlist = NhlRLCreate(NhlSETRL);
  vf_rlist  = NhlRLCreate(NhlSETRL);
  tx_rlist  = NhlRLCreate(NhlSETRL);
  vc_rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(sf_rlist);
  NhlRLClear(vf_rlist);
  NhlRLClear(tx_rlist);
  NhlRLClear(vc_rlist);

  cmap_len[0] = NCOLORS;
  cmap_len[1] = 3;
  srlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(srlist);
  NhlRLSetMDFloatArray(srlist,NhlNwkColorMap,&cmap[0][0],2,cmap_len);
  (void)NhlSetValues(wks, srlist);

  NhlRLSetString (vc_rlist,"pmLabelBarDisplayMode"  , "Always");
  NhlRLSetString (vc_rlist,"pmLabelBarSide"         , "Bottom");
  NhlRLSetString (vc_rlist,"lbOrientation"          , "Horizontal");
  NhlRLSetString (vc_rlist,"lbPerimOn"              , "False");
  NhlRLSetInteger(vc_rlist,"lbTitleFont"            , 25);
  NhlRLSetString (vc_rlist,"lbTitleString"          , "TEMPERATURE (:S:o:N:F)");
  NhlRLSetFloat  (vc_rlist,"tiMainFontHeightF"      , 0.03);
  NhlRLSetString (vc_rlist,"tiMainString"           , ":F25:Wind velocity vectors");
  
  NhlRLSetInteger(vc_rlist,"vcFillArrowEdgeColor"    , 1);
  NhlRLSetString (vc_rlist,"vcFillArrowsOn"          , "True");
  NhlRLSetFloat  (vc_rlist,"vcMinFracLengthF"        , 0.33);
  NhlRLSetFloat  (vc_rlist,"vcMinMagnitudeF"         , 0.001);
  NhlRLSetString (vc_rlist,"vcMonoFillArrowFillColor", "False");
  NhlRLSetString (vc_rlist,"vcMonoLineArrowColor"    , "False");
  NhlRLSetFloat  (vc_rlist,"vcRefLengthF"            , 0.045);
  NhlRLSetFloat  (vc_rlist,"vcRefMagnitudeF"         , 20.0);

/*
 * Loop across time dimension and create a vector plot for each one.
 * Then, we'll create several panel plots.
 * Create a separate array to hold the same plots, but set some of
 * them to "0" (missing).
 */
  varray      = (int*)malloc(ntime_UV*sizeof(int));
  varray_copy = (int*)malloc(ntime_UV*sizeof(int));

  special_res.nglMaximize = 0;
  special_res.nglFrame    = 0;
  special_res.nglDraw     = 0;
/*
  for(i = 0; i < ntime_UV; i++) { 
 */
  for(i = 0; i < 20; i++) {
    Unew  = &((float*)U)[nlatlon_UV*i];
    Vnew  = &((float*)V)[nlatlon_UV*i];
    T2new = &((float*)T2)[nlatlon_UV*i];
    varray[i] = ngl_vector_scalar_wrap(wks, Unew, Vnew, T2new,
                                       type_U, type_V, type_T2,
                                       nlat_UV, nlon_UV, 
                                       is_lat_coord_UV, lat_UV, 
                                       type_lat_UV, is_lon_coord_UV, 
                                       lon_UV, type_lon_UV, 
                                       is_missing_U, is_missing_V, 
                                       is_missing_T2, FillValue_U, 
                                       FillValue_V, FillValue_T2, 
                                       vf_rlist, sf_rlist,
                                       vc_rlist, &special_res);
    
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

  special_pres.nglFrame = 0;
  special_tres.nglFrame = 1;

/*
 * Begin panel plots.
 *
 * Start with a single row of plots, then make the first plot and the last
 * plot missing.
 */
  panel_dims[0] = 1;
  panel_dims[1] = 3;
  nplots = panel_dims[0] * panel_dims[1];
  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[0] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[2] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Single column of plots, then make the first plot and the last
 * plot missing.
 */
  panel_dims[0] = 3;          /* rows    */
  panel_dims[1] = 1;          /* columns */
  nplots = panel_dims[0] * panel_dims[1];
  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[0] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[2] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * 2 rows x 3 columns of plots. Then make the top, bottom, right,
 * and left rows missing.
 */
  panel_dims[0] = 2;          /* rows    */
  panel_dims[1] = 3;          /* columns */
  nplots = panel_dims[0] * panel_dims[1];
  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[0] = 0;
  varray[1] = 0;
  varray[2] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[3] = 0;
  varray[4] = 0;
  varray[5] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[0] = 0;
  varray[3] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[2] = 0;
  varray[5] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * 3 rows x 2 columns of plots. Then make the top, bottom, right,
 * and left rows missing.
 */
  panel_dims[0] = 3;          /* rows    */
  panel_dims[1] = 2;          /* columns */
  nplots = panel_dims[0] * panel_dims[1];
  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[0] = 0;
  varray[1] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[2] = 0;
  varray[3] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[4] = 0;
  varray[5] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[0] = 0;
  varray[2] = 0;
  varray[4] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[1] = 0;
  varray[3] = 0;
  varray[5] = 0;

  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * 3 rows x 2 columns of plots, but only pass in 4 plots, then 2, then 1.
 */
  panel_dims[0] = 3;          /* rows    */
  panel_dims[1] = 2;          /* columns */
  nplots = 4;
  ngl_panel_wrap(wks, varray, 4, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  varray[0] = 0;
  varray[1] = 0;
  ngl_panel_wrap(wks, varray, 4, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  nplots = 2;
  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  nplots = 1;
  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, &special_pres);
  title = create_title(varray, varray_copy, nplots, panel_dims);
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
  free(U);
  free(V);
  free(T2);
  free(lat_UV);
  free(lon_UV);
  exit(0);

}
