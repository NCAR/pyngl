#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gsun.h"

main()
{
/*
 * Declare variables for the HLU routine calls.
 */

  int wks, contour, xy, vector, streamline, map, cntrmap;
  int wk_rlist, sf_rlist, ca_rlist, vf_rlist;
  int cn_rlist, xy_rlist, xyd_rlist, vc_rlist, st_rlist, mp_rlist;
  int cn2_rlist, mp2_rlist;
  int cncolors[]  = {2,16,30,44,58,52,86,100,114,128,142,156,170};
  int mpcolors[]  = {0, -1, 238, -1};
  int pttrns[]  = {0,1,2,3,4,5,6,7,8,9,10,11,12};

/*
 * Declare variables for getting information from netCDF file.
 */

  int ncid_T, ncid_U, ncid_V;
  int id_T, id_U, id_V;
  int lonid_T, latid_T, lonid_UV, latid_UV;
  int nlon_T, nlat_T, nlat_UV, nlon_UV;
  int ndims_T, dsizes_T[2], ndims_UV;
  int is_missing_T, is_missing_U, is_missing_V;
  int attid, status;

/*
 * Declare variables to hold values, missing values, and types that
 * are read in from netCDF files.
 */
  void  *T, *U, *V, *lat_T, *lon_T, *lat_UV, *lon_UV;
  void  *FillValue_T, *FillValue_U, *FillValue_V;
  int is_lat_coord_T, is_lon_coord_T, is_lat_coord_UV, is_lon_coord_UV;

  nc_type nctype_T, nctype_lat_T, nctype_lon_T;
  nc_type nctype_U, nctype_V, nctype_lat_UV, nctype_lon_UV;
  char type_T[TYPE_LEN], type_U[TYPE_LEN], type_V[TYPE_LEN];
  char type_lat_T[TYPE_LEN], type_lon_T[TYPE_LEN];
  char type_lat_UV[TYPE_LEN], type_lon_UV[TYPE_LEN];

  size_t i, *start, *count;
  char  filename_T[256], filename_U[256], filename_V[256];
  const char *dir = _NGGetNCARGEnv("data");

/*
 * Open the netCDF files for contour and vector data.
 */

  sprintf(filename_T, "%s/cdf/meccatemp.cdf", dir );
  sprintf(filename_U, "%s/cdf/Ustorm.cdf", dir );
  sprintf(filename_V, "%s/cdf/Vstorm.cdf", dir );

  nc_open(filename_T,NC_NOWRITE,&ncid_T);
  nc_open(filename_U,NC_NOWRITE,&ncid_U);
  nc_open(filename_V,NC_NOWRITE,&ncid_V);

/*
 * Get the lat/lon dimension ids so we can retrieve their lengths.
 * The lat/lon arrays for the U and V data files are the same, so we
 * only need to retrieve one set of them.
 */

  nc_inq_dimid(ncid_T,"lat",&latid_T);
  nc_inq_dimid(ncid_T,"lon",&lonid_T);
  nc_inq_dimid(ncid_U,"lat",&latid_UV);
  nc_inq_dimid(ncid_U,"lon",&lonid_UV);

  nc_inq_dimlen(ncid_T,latid_T, (size_t*)&nlat_T);
  nc_inq_dimlen(ncid_T,lonid_T, (size_t*)&nlon_T);
  nc_inq_dimlen(ncid_U,latid_UV,(size_t*)&nlat_UV);  
  nc_inq_dimlen(ncid_U,lonid_UV,(size_t*)&nlon_UV);

/*
 * Get temperature, u, v, lat, and lon ids.
 */

  nc_inq_varid(ncid_T,  "t",&id_T);
  nc_inq_varid(ncid_U,  "u",&id_U);
  nc_inq_varid(ncid_V,  "v",&id_V);
  nc_inq_varid(ncid_T,"lat",&latid_T);
  nc_inq_varid(ncid_T,"lon",&lonid_T);
  nc_inq_varid(ncid_U,"lat",&latid_UV);
  nc_inq_varid(ncid_U,"lon",&lonid_UV);

/*
 * Check if T, U, or V has a _FillValue attribute set.  If so,
 * retrieve them later.
 */

  status = nc_inq_attid (ncid_T, id_T, "_FillValue", &attid); 
  if(status == NC_NOERR) {
    is_missing_T = 1;
  }
  else {
    is_missing_T = 0;
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
 * Get type and number of dimensions of T, then read in first
 * nlat_T x nlon_T subsection of T. Also, read in missing value
 * if one is set.
 */

  nc_inq_vartype  (ncid_T, id_T, &nctype_T);
  nc_inq_varndims (ncid_T, id_T, &ndims_T);

  start    = (size_t *)calloc(ndims_T,sizeof(size_t));
  count    = (size_t *)calloc(ndims_T,sizeof(size_t));
  for(i = 0; i < ndims_T; i++)   start[i] = 0;
  for(i = 0; i < ndims_T-2; i++) count[i] = 1;
  count[ndims_T-2] = nlat_T;
  count[ndims_T-1] = nlon_T;

  switch(nctype_T) {

  case NC_DOUBLE:
    strcpy(type_T,"double");
    T      = (double *)calloc(nlat_T,sizeof(double));

    nc_get_vara_double(ncid_T,id_T,start,count,(double*)T);

/*
 * Get double missing value.
 */
    if(is_missing_T) {
      FillValue_T = (double *)calloc(1,sizeof(double));
      nc_get_att_double (ncid_T, id_T, "_FillValue", (double*)FillValue_T); 
    }
    break;

  case NC_FLOAT:
    strcpy(type_T,"float");
    T      = (float *)calloc(nlat_T*nlon_T,sizeof(float));

    nc_get_vara_float(ncid_T,id_T,start,count,(float*)T);

/*
 * Get float missing value.
 */
    if(is_missing_T) {
      FillValue_T = (float *)calloc(1,sizeof(float));
      nc_get_att_float (ncid_T, id_T, "_FillValue", (float*)FillValue_T); 
    }
    break;

  case NC_INT:
    strcpy(type_T,"integer");
    T      = (int *)calloc(nlat_T*nlon_T,sizeof(int));

    nc_get_vara_int(ncid_T,id_T,start,count,(int*)T);

/*
 * Get integer missing value.
 */
    if(is_missing_T) {
      FillValue_T = (int *)calloc(1,sizeof(int));
      nc_get_att_int (ncid_T, id_T, "_FillValue", (int*)FillValue_T); 
    }
    break;
  }

/*
 * Read in lat and lon coordinate arrays for "t".
 */

  nc_inq_vartype  (ncid_T, latid_T, &nctype_lat_T);
  nc_inq_vartype  (ncid_T, lonid_T, &nctype_lon_T);

  is_lat_coord_T = 1;
  switch(nctype_lat_T) {

  case NC_DOUBLE:
    strcpy(type_lat_T,"double");
    lat_T      = (double *)calloc(nlat_T,sizeof(double));

    nc_get_var_double(ncid_T,latid_T,(double*)lat_T);
    break;

  case NC_FLOAT:
    strcpy(type_lat_T,"float");
    lat_T      = (float *)calloc(nlat_T,sizeof(float));

    nc_get_var_float(ncid_T,latid_T,(float*)lat_T);

  case NC_INT:
    strcpy(type_lat_T,"integer");
    lat_T      = (int *)calloc(nlat_T,sizeof(int));

    nc_get_var_int(ncid_T,latid_T,(int*)lat_T);
    break;
  }

  is_lon_coord_T = 1;
  switch(nctype_lon_T) {

  case NC_DOUBLE:
    strcpy(type_lon_T,"double");
    lon_T      = (double *)calloc(nlon_T,sizeof(double));

    nc_get_var_double(ncid_T,lonid_T,(double*)lon_T);
    break;

  case NC_FLOAT:
    strcpy(type_lon_T,"float");
    lon_T      = (float *)calloc(nlon_T,sizeof(float));

    nc_get_var_float(ncid_T,lonid_T,(float*)lon_T);

  case NC_INT:
    strcpy(type_lon_T,"integer");
    lon_T      = (int *)calloc(nlon_T,sizeof(int));

    nc_get_var_int(ncid_T,lonid_T,(int*)lon_T);
    break;
  }

/*
 * Get type and number of dimensions of U and V, then read in first
 * nlat_UV x nlon_UV subsection of "u" and "v". Also, read in missing
 * values if ones are set.
 *
 * U and V are assumed to have the same dimensions.
 */

  free(start);
  free(count);

  nc_inq_vartype  (ncid_U, id_U, &nctype_U);
  nc_inq_vartype  (ncid_V, id_V, &nctype_V);
  nc_inq_varndims (ncid_U, id_U, &ndims_UV);

  start = (size_t *)calloc(ndims_UV,sizeof(size_t));
  count = (size_t *)calloc(ndims_UV,sizeof(size_t));
  for(i = 0; i < ndims_UV; i++)   start[i] = 0;
  for(i = 0; i < ndims_UV-2; i++) count[i] = 1;
  count[ndims_UV-2] = nlat_UV;
  count[ndims_UV-1] = nlon_UV;

  switch(nctype_U) {

  case NC_DOUBLE:
    strcpy(type_U,"double");
    U      = (double *)calloc(nlat_UV*nlon_UV,sizeof(double));

/*
 * Get double values.
 */
    nc_get_vara_double(ncid_U,id_U,start,count,(double*)U);

/*
 * Get double missing value.
 */
    if(is_missing_U) {
      FillValue_U = (int *)calloc(1,sizeof(int));
      nc_get_att_double (ncid_U, id_U, "_FillValue", (double*)FillValue_U); 
    }
    break;

  case NC_FLOAT:
    strcpy(type_U,"float");
    U      = (float *)calloc(nlat_UV*nlon_UV,sizeof(float));

/*
 * Get float values.
 */
    nc_get_vara_float(ncid_U,id_U,start,count,(float*)U);

/*
 * Get float missing value.
 */
    if(is_missing_U) {
      FillValue_U = (int *)calloc(1,sizeof(int));
      nc_get_att_float (ncid_U, id_U, "_FillValue", (float*)FillValue_U); 
    }
    break;

  case NC_INT:
    strcpy(type_U,"integer");
    U      = (int *)calloc(nlat_UV*nlon_UV,sizeof(int));

/*
 * Get integer values.
 */
    nc_get_vara_int(ncid_U,id_U,start,count,(int*)U);

/*
 * Get integer missing value.
 */
    if(is_missing_U) {
      FillValue_U = (int *)calloc(1,sizeof(int));
      nc_get_att_int (ncid_U, id_U, "_FillValue", (int*)FillValue_U); 
    }
    break;
  }

  switch(nctype_V) {

  case NC_DOUBLE:
    strcpy(type_V,"double");
    V      = (double *)calloc(nlat_UV*nlon_UV,sizeof(double));

/*
 * Get double values.
 */
    nc_get_vara_double(ncid_V,id_V,start,count,(double*)V);

/*
 * Get double missing value.
 */
    if(is_missing_V) {
      FillValue_V = (int *)calloc(1,sizeof(int));
      nc_get_att_float (ncid_V, id_V, "_FillValue", (float*)FillValue_V); 
    }
    break;

  case NC_FLOAT:
    strcpy(type_V,"float");
    V      = (float *)calloc(nlat_UV*nlon_UV,sizeof(float));

/*
 * Get float values.
 */
    nc_get_vara_float(ncid_V,id_V,start,count,(float*)V);

/*
 * Get float missing value.
 */
    if(is_missing_V) {
      FillValue_V = (int *)calloc(1,sizeof(int));
      nc_get_att_float (ncid_V, id_V, "_FillValue", (float*)FillValue_V); 
    }
    break;

  case NC_INT:
    strcpy(type_V,"integer");
    V      = (int *)calloc(nlat_UV*nlon_UV,sizeof(int));

/*
 * Get integer values.
 */
    nc_get_vara_int(ncid_V,id_V,start,count,(int*)V);

/*
 * Get integer missing value.
 */
    if(is_missing_V) {
      FillValue_V = (int *)calloc(1,sizeof(int));
      nc_get_att_float (ncid_V, id_V, "_FillValue", (float*)FillValue_V); 
    }
    break;
  }

/*
 * Read in lat coordinate arrays for "u" and "v".
 */

  nc_inq_vartype  (ncid_U, latid_UV, &nctype_lat_UV);

  is_lat_coord_UV = 1;
  switch(nctype_lat_UV) {

  case NC_DOUBLE:
    strcpy(type_lat_UV,"double");
    lat_UV      = (double *)calloc(nlat_UV,sizeof(double));

/*
 * Get double values.
 */
    nc_get_var_double(ncid_U,latid_UV,(double*)lat_UV);
    break;

  case NC_FLOAT:
    strcpy(type_lat_UV,"float");
    lat_UV      = (float *)calloc(nlat_UV,sizeof(float));

/*
 * Get float values.
 */
    nc_get_var_float(ncid_U,latid_UV,(float*)lat_UV);

  case NC_INT:
    strcpy(type_lat_UV,"integer");
    lat_UV      = (int *)calloc(nlat_UV,sizeof(int));

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
    lon_UV      = (double *)calloc(nlon_UV,sizeof(double));

    nc_get_var_double(ncid_U,lonid_UV,(double*)lon_UV);
    break;

  case NC_FLOAT:
    strcpy(type_lon_UV,"float");
    lon_UV      = (float *)calloc(nlon_UV,sizeof(float));

    nc_get_var_float(ncid_U,lonid_UV,(float*)lon_UV);

  case NC_INT:
    strcpy(type_lon_UV,"integer");
    lon_UV      = (int *)calloc(nlon_UV,sizeof(int));

    nc_get_var_int(ncid_U,lonid_UV,(int*)lon_UV);
    break;
  }

/*
 * Close the netCDF files.
 */

  ncclose(ncid_T);
  ncclose(ncid_U);
  ncclose(ncid_V);

/*----------------------------------------------------------------------*
 *
 * Start graphics portion of code.
 *
 *----------------------------------------------------------------------*/

/*
 * Set up workstation resource list.
 */

  wk_rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(wk_rlist);

/* 
 * Set color map resource and open workstation.
 */

  NhlRLSetString(wk_rlist,"wkColorMap","rainbow+gray");
  wks = gsn_open_wks("ncgm","test", wk_rlist);

/*
 * Initialize contour, vector, scalar field, and vector field 
 * resource lists.
 */

  sf_rlist  = NhlRLCreate(NhlSETRL);
  ca_rlist  = NhlRLCreate(NhlSETRL);
  vf_rlist  = NhlRLCreate(NhlSETRL);
  cn_rlist  = NhlRLCreate(NhlSETRL);
  cn2_rlist = NhlRLCreate(NhlSETRL);
  xy_rlist  = NhlRLCreate(NhlSETRL);
  xyd_rlist = NhlRLCreate(NhlSETRL);
  vc_rlist  = NhlRLCreate(NhlSETRL);
  st_rlist  = NhlRLCreate(NhlSETRL);
  mp_rlist  = NhlRLCreate(NhlSETRL);
  mp2_rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(sf_rlist);
  NhlRLClear(ca_rlist);
  NhlRLClear(vf_rlist);
  NhlRLClear(cn_rlist);
  NhlRLClear(cn2_rlist);
  NhlRLClear(xy_rlist);
  NhlRLClear(xyd_rlist);
  NhlRLClear(vc_rlist);
  NhlRLClear(st_rlist);
  NhlRLClear(mp_rlist);
  NhlRLClear(mp2_rlist);

/*
 * Set some contour resources.
 */

  NhlRLSetString      (cn_rlist, "cnFillOn",             "True");
  NhlRLSetIntegerArray(cn_rlist, "cnFillColors",         cncolors,13);
  NhlRLSetString      (cn_rlist, "cnLineLabelsOn",       "False");
  NhlRLSetString      (cn_rlist, "lbPerimOn",            "False");
  NhlRLSetString      (cn_rlist, "pmLabelBarDisplayMode","ALWAYS");

/*
 * Create and draw contour plot, and advance frame.
 */

  contour = gsn_contour_wrap(wks, T, type_T, nlat_T, nlon_T, 
                             is_lat_coord_T, lat_T, type_lat_T, 
                             is_lon_coord_T, lon_T, type_lon_T, 
                             is_missing_T, FillValue_T, sf_rlist, cn_rlist);

/*
 * Set some XY and XY data spec resources.
 */

  NhlRLSetFloat  (xy_rlist,  "trXMinF",          -180);
  NhlRLSetFloat  (xy_rlist,  "trXMaxF",           180);
  NhlRLSetString (xy_rlist,  "tiMainFont",       "helvetica-bold");
  NhlRLSetString (xy_rlist,  "tiMainFontColor",  "red");
  NhlRLSetString (xy_rlist,  "tiMainString",     "This is a boring red title");

/*
 * Resources for a single line. 
 */
/*
   NhlRLSetString (xyd_rlist, "xyLineColor",      "green");
   NhlRLSetFloat  (xyd_rlist, "xyLineThicknessF", 3.0);
 */

/*
 * Resources for multiple lines.
 */
  NhlRLSetString (xyd_rlist,      "xyMonoLineColor", "False");
  NhlRLSetIntegerArray(xyd_rlist, "xyLineColors",    cncolors,10);
  NhlRLSetIntegerArray(xyd_rlist, "xyDashPatterns",  pttrns,10);

/*
 * Create and draw XY plot, and advance frame. In this case, the X data
 * contains no missing values, but the Y data possibly does.
 *
 * Plot only the first 10 lines.
 */
  dsizes_T[0] = 10;           /* Could be up to nlat_T lines. */
  dsizes_T[1] = nlon_T;

  xy = gsn_xy_wrap(wks, lon_T, T, type_lon_T, type_T, 1, &nlon_T, 
                   2, &dsizes_T[0], 0, is_missing_T, NULL, FillValue_T, 
                   ca_rlist, xy_rlist, xyd_rlist);

/* 
 * To plot just a single line.
 */
/*
  xy = gsn_xy_wrap(wks, lon_T, lat_T, type_lon_T, type_lat_T, 1, &nlon_T, 
                   1, &nlat_T, 0, 0, NULL, NULL, 
                   ca_rlist, xy_rlist, xyd_rlist);
 */

/*
 * Create and draw streamline plot, and advance frame.
 */

  streamline = gsn_streamline_wrap(wks, U, V, type_U, type_V, 
                                   nlat_UV, nlon_UV, 
                                   is_lat_coord_UV, lat_UV, type_lat_UV, 
                                   is_lon_coord_UV, lon_UV, type_lon_UV, 
                                   is_missing_U, is_missing_V, 
                                   FillValue_U, FillValue_V,
                                   vf_rlist, st_rlist);

/*
 * Set some vector resources.
 */

  NhlRLSetFloat (vc_rlist, "vcRefAnnoOrthogonalPosF", -0.2);
  NhlRLSetFloat (vc_rlist, "vpXF",      0.10);
  NhlRLSetFloat (vc_rlist, "vpYF",      0.95);
  NhlRLSetFloat (vc_rlist, "vpWidthF",  0.85);
  NhlRLSetFloat (vc_rlist, "vpHeightF", 0.85);

/*
 * Create and draw vector plot, and advance frame.
 */

  vector = gsn_vector_wrap(wks, U, V, type_U, type_V, nlat_UV, nlon_UV, 
                           is_lat_coord_UV, lat_UV, type_lat_UV, 
                           is_lon_coord_UV, lon_UV, type_lon_UV, 
                           is_missing_U, is_missing_V, 
                           FillValue_U, FillValue_V,
                           vf_rlist, vc_rlist);

/*
 * Create and draw map plot, and advance frame.
 */

  NhlRLSetString  (mp_rlist, "mpGridAndLimbOn",    "False");
  NhlRLSetInteger (mp_rlist, "mpPerimOn",          1);
  NhlRLSetString  (mp_rlist, "pmTitleDisplayMode", "Always");
  NhlRLSetString  (mp_rlist, "tiMainString",       "CylindricalEquidistant");
  map = gsn_map_wrap(wks, mp_rlist);

/*
 * Create contours over a map.
 *
 * First set up some resources.
 */

  NhlRLSetString      (cn2_rlist, "cnFillOn",              "True");
  NhlRLSetIntegerArray(cn2_rlist, "cnFillColors",          cncolors,13);
  NhlRLSetString      (cn2_rlist, "cnLineLabelsOn",        "False");
  NhlRLSetString      (cn2_rlist, "cnInfoLabelOn",         "False");
  NhlRLSetString      (cn2_rlist, "pmLabelBarDisplayMode", "ALWAYS");
  NhlRLSetString      (cn2_rlist, "lbOrientation",         "Horizontal");
  NhlRLSetString      (cn2_rlist, "lbPerimOn",             "False");
  NhlRLSetString      (cn2_rlist, "pmLabelBarSide",        "Bottom");

  NhlRLSetString      (mp2_rlist, "mpFillOn",              "True");
  NhlRLSetIntegerArray(mp2_rlist, "mpFillColors",          mpcolors,4);
  NhlRLSetString      (mp2_rlist, "mpFillDrawOrder",       "PostDraw");
  NhlRLSetString      (mp2_rlist, "mpGridAndLimbOn",       "False");
  NhlRLSetInteger     (mp2_rlist, "mpPerimOn",             1);

  cntrmap = gsn_contour_map_wrap(wks, T, type_T, nlat_T, nlon_T, 
                                 is_lat_coord_T, lat_T, type_lat_T, 
                                 is_lon_coord_T, lon_T, type_lon_T, 
                                 is_missing_T, FillValue_T, 
                                 sf_rlist, cn2_rlist, mp2_rlist);
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
  free(T);
  free(U);
  free(V);
  free(lat_T);
  free(lon_T);
  free(lat_UV);
  free(lon_UV);
  exit(0);
}


/*
 * This function computes the PostScript device coordinates needed to
 * make a plot fill up the full page.
 *
 * bb     : bounding box that contains all graphical objects. It should
 *          be a n x 4 float array with values between 0 and 1.
 *            (top,bottom,left,right)
 *
 * res : list of optional resources. Ones accepted include:
 *
 * "gsnPaperOrientation" - orientation of paper. Can be "landscape",
 *                         "portrait", or "auto". Default is "auto".
 *
 *       "gsnPaperWidth"  - width of paper (in inches, default is 8.5)
 *       "gsnPaperHeight" - height of paper (in inches, default is 11.0)
 *       "gsnPaperMargin" - margin to leave around plots (in inches,
 *                        default is 0.5)
 *
 */
void compute_ps_device_coords(int wks, int plot)
{
  NhlBoundingBox box;
  float top, bot, lft, rgt, dpi_pw, dpi_ph, dpi_margin;
  float paper_width, paper_height, paper_margin;
  float pw, ph, lx, ly, ux, uy, dw, dh, ndc2du;
  int srlist, is_debug, dpi, coords[4];
  NhlWorkOrientation lc_orient, paper_orient; 

  NhlGetBB(plot,&box);

/*
 * These next four paper* variables should be settable by the user. Right
 * now, they are being hard-coded.
 */
  paper_orient = -1;         /* Auto */
  paper_height = 11.0;
  paper_width  = 8.5;
  paper_margin = 0.5;
  is_debug     = 0;

/*
 * Get the bounding box that encompasses the plot. Note that even
 * though the bounding box coordinates should be positive, it is
 * possible for them to be negative, and we need to keep these
 * negative values in our calculations later to preserve the 
 * aspect ratio.
 */

  top = box.t;
  bot = box.b;
  lft = box.l;
  rgt = box.r;

/*
 * Debug prints
 */

  if(is_debug) {
    printf("-------Bounding box values for PostScript/PDF-------\n");
    printf("    top = %g bot = %g lft = %g rgt = %g\n", top, bot, lft, rgt);
  }
/*
 * Initialization
 */
  dpi        = 72;                       /* Dots per inch. */
  dpi_pw     = paper_width  * dpi;
  dpi_ph     = paper_height * dpi;
  dpi_margin = paper_margin * dpi;

/*
 * Get paper height/width in dpi units
 */
  pw = rgt - lft;
  ph = top - bot;

  lx = dpi_margin;
  ly = dpi_margin;

  ux = dpi_pw - dpi_margin;
  uy = dpi_ph - dpi_margin;

  dw = ux - lx;
  dh = uy - ly;

/*
 * Determine orientation, and then calculate device coordinates based
 * on this.
 */
 
  if( (lc_orient == NhlPORTRAIT) || ((lc_orient == -1) &&
     (ph / pw) >= 1.0)) {
/*
 * If plot is higher than it is wide, then default to portrait if
 * orientation is not specified.
 */
    lc_orient = NhlPORTRAIT;

    if ( (ph/pw) > (dh/dw) ) {
                                      /* paper height limits size */
      ndc2du = dh / ph;
    }
    else {
      ndc2du = dw / pw;
    }
/*
 * Compute device coordinates.
 */
    lx = dpi_margin + 0.5 * ( dw - pw * ndc2du) - lft * ndc2du;
    ly = dpi_margin + 0.5 * ( dh - ph * ndc2du) - bot * ndc2du;
    ux = lx + ndc2du;
    uy = ly + ndc2du;
  }
  else {
/*
 * If plot is wider than it is high, then default to landscape if
 * orientation is not specified.
 */
    lc_orient = NhlLANDSCAPE;

    if ( (pw/ph) > (dh/dw) ) {
                                      /* paper height limits size */
      ndc2du = dh / pw;
    }
    else {
      ndc2du = dw / ph;
    }

/*
 * Compute device coordinates.
 */
    ly = dpi_margin + 0.5 * (dh - pw * ndc2du) - (1.0 - rgt) * ndc2du;
    lx = dpi_margin + 0.5 * (dw - ph * ndc2du) - bot * ndc2du;
    ux = lx + ndc2du;
    uy = ly + ndc2du;
  }

/*
 * Return device coordinates and the orientation.
 */
  coords[0] = (int)lx;
  coords[1] = (int)ly;
  coords[2] = (int)ux;
  coords[3] = (int)uy;
/*
 * Debug prints.
 */
    
  if(is_debug) {
    printf("-------Device coordinates for PostScript/PDF-------\n");
    printf("    wkDeviceLowerX = %d\n", coords[0]);
    printf("    wkDeviceLowerY = %d\n", coords[1]);
    printf("    wkDeviceUpperX = %d\n", coords[2]);
    printf("    wkDeviceUpperY = %d\n", coords[3]);
    printf("    wkOrientation  = %d\n", lc_orient);
  } 

/*
 * Initialize setting and retrieving resource lists.
 */
  srlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(srlist);

/*
 * Set the device coordinates.
 */
  NhlRLSetInteger(srlist,"wkDeviceLowerX", coords[0]);
  NhlRLSetInteger(srlist,"wkDeviceLowerY", coords[1]);
  NhlRLSetInteger(srlist,"wkDeviceUpperX", coords[2]);
  NhlRLSetInteger(srlist,"wkDeviceUpperY", coords[3]);
  NhlRLSetInteger(srlist,"wkOrientation",  lc_orient);
  (void)NhlSetValues(wks, srlist);

}


/*
 * This function maximizes the size of the plot in the viewport.
 */

void maximize_plot(int wks, int plot)
{
  NhlBoundingBox box; 
  float top, bot, lft, rgt, uw, uh;
  float scale, vpx, vpy, vpw, vph, dx, dy, new_uw, new_uh, new_ux, new_uy;
  float new_vpx, new_vpy, new_vpw, new_vph, margin = 0.02;
  int srlist, grlist, *coords;

/*
 * Initialize setting and retrieving resource lists.
 */
  srlist = NhlRLCreate(NhlSETRL);
  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(srlist);
  NhlRLClear(grlist);

/*
 * Get bounding box of plot.
 */

  NhlGetBB(plot,&box);

  top = box.t;
  bot = box.b;
  lft = box.l;
  rgt = box.r;

/*
 * Get height/width of plot in NDC units.
 */

  uw  = rgt - lft;
  uh = top - bot;

/*
 * Calculate scale factor needed to make plot larger (or smaller, if it's
 * outside the viewport).
 */

  scale = (1 - 2*margin)/max(uw,uh);

/*
 * Get the viewport.
 */

  NhlRLGetFloat(grlist,"vpXF",     &vpx);
  NhlRLGetFloat(grlist,"vpYF",     &vpy);
  NhlRLGetFloat(grlist,"vpWidthF", &vpw);
  NhlRLGetFloat(grlist,"vpHeightF",&vph);
  (void)NhlGetValues(plot, grlist);

/* 
 * Calculate distance from plot's left position to its leftmost
 * annotation and from plot's top position to its topmost 
 * annotation.
 */

  dx = scale * (vpx - lft); 
  dy = scale * (top - vpy);

/*
 * Calculate new viewport coordinates.
 */ 

  new_uw = uw * scale;
  new_uh = uh * scale;
  new_ux =     .5 * (1-new_uw);
  new_uy = 1 - .5 * (1-new_uh);

  new_vpx = new_ux + dx;
  new_vpy = new_uy - dy;
  new_vpw = vpw * scale;
  new_vph = vph * scale;

/*
 * Return new coordinates 
 */

  NhlRLSetFloat(srlist,"vpXF",      new_vpx);
  NhlRLSetFloat(srlist,"vpYF",      new_vpy);
  NhlRLSetFloat(srlist,"vpWidthF",  new_vpw);
  NhlRLSetFloat(srlist,"vpHeightF", new_vph);
  (void)NhlSetValues(plot, srlist);

  if(!strcmp(NhlClassName(wks),"psWorkstationClass") ||
     !strcmp(NhlClassName(wks),"pdfWorkstationClass")) {
/*
 * Compute and set device coordinates that will make plot fill the 
 * whole page.
 */
    compute_ps_device_coords(wks,plot);
  }
}

/*
 * This function creates a scalar field object that will get
 * used with the contour object.
 */

int scalar_field(void *data, const char *type_data, int ylen, int xlen, 
                 int is_ycoord, void *ycoord, const char *type_ycoord,
                 int is_xcoord, void *xcoord, const char *type_xcoord,
                 int is_missing_data, void *FillValue_data, int sf_rlist)
{
  int app, field, length[2];

/*
 * Retrieve application id.
 */
  app = NhlAppGetDefaultParentId();

/*
 * Create a scalar field object that will be used as the
 * dataset for the contour object. Check for missing values
 * here as well.
 */

  length[0] = ylen;
  length[1] = xlen;

  if(!strcmp(type_data,"double")) {
    NhlRLSetMDDoubleArray(sf_rlist,"sfDataArray",(double*)data,2,
                          length);
    
    if(is_missing_data) {
      NhlRLSetDouble(sf_rlist,"sfMissingValueV",((double*)FillValue_data)[0]);
    }
  }
  else if(!strcmp(type_data,"float")) {
    NhlRLSetMDFloatArray(sf_rlist,"sfDataArray",(float*)data,2,length);

    if(is_missing_data) {
      NhlRLSetFloat(sf_rlist,"sfMissingValueV",((float*)FillValue_data)[0]);
    }
  }
  else if(!strcmp(type_data,"integer")) {
    NhlRLSetMDIntegerArray(sf_rlist,"sfDataArray",(int*)data,2,length);
    if(is_missing_data) {
      NhlRLSetInteger(sf_rlist,"sfMissingValueV",((int*)FillValue_data)[0]);
    }
  }

/*
 * Check for coordinate arrays.
 */
 
  if(is_ycoord) {
    if(!strcmp(type_ycoord,"double")) {
       NhlRLSetDoubleArray(sf_rlist,"sfYArray",(double*)ycoord,ylen);
   }
   else if(!strcmp(type_ycoord,"float")) {
     NhlRLSetFloatArray(sf_rlist,"sfYArray",(float*)ycoord,ylen);
   }
   else if(!strcmp(type_ycoord,"integer")) {
     NhlRLSetIntegerArray(sf_rlist,"sfYArray",(int*)ycoord,ylen);
   }
  }
  if(is_xcoord) {
    if(!strcmp(type_xcoord,"double")) {
      NhlRLSetDoubleArray(sf_rlist,"sfXArray",(double*)xcoord,xlen);
    }
    else if(!strcmp(type_xcoord,"float")) {
      NhlRLSetFloatArray(sf_rlist,"sfXArray",(float*)xcoord,xlen);
    }
    else if(!strcmp(type_xcoord,"integer")) {
      NhlRLSetIntegerArray(sf_rlist,"sfXArray",(int*)xcoord,xlen);
    }
  }

/*
 * Create the object.
 */
   NhlCreate(&field,"field",NhlscalarFieldClass,app,sf_rlist);
   
   return(field);
}

/*
 * This function creates a coord arrays object that will get
 * used with the XY object.
 */

int coord_array(void *x, void *y, const char *type_x, const char *type_y, 
                int ndims_x, int *dsizes_x, int ndims_y, int *dsizes_y, 
                int is_missing_x, int is_missing_y,
                void *FillValue_x, void *FillValue_y, int ca_rlist)
{
  int app, carray;

/*
 * Retrieve application id.
 */
  app = NhlAppGetDefaultParentId();

/*
 * Create a coord arrays object that will be used as the
 * dataset for an XY object.
 */

  if(!strcmp(type_x,"double")) {
    NhlRLSetMDDoubleArray(ca_rlist,"caXArray",(double*)x,ndims_x,dsizes_x);
    
    if(is_missing_x) {
      NhlRLSetDouble(ca_rlist,"caXMissingV",((double*)FillValue_x)[0]);
    }
  }
  else if(!strcmp(type_x,"float")) {
    NhlRLSetMDFloatArray(ca_rlist,"caXArray",(float*)x,ndims_x,dsizes_x);

    if(is_missing_x) {
      NhlRLSetFloat(ca_rlist,"caXMissingV",((float*)FillValue_x)[0]);
    }
  }
  else if(!strcmp(type_x,"integer")) {
    NhlRLSetMDIntegerArray(ca_rlist,"caXArray",(int*)x,ndims_x,dsizes_x);

    if(is_missing_x) {
      NhlRLSetInteger(ca_rlist,"caXMissingV",((int*)FillValue_x)[0]);
    }
  }

  if(!strcmp(type_y,"double")) {
    NhlRLSetMDDoubleArray(ca_rlist,"caYArray",(double*)y,ndims_y,dsizes_y);

    if(is_missing_y) {
      NhlRLSetDouble(ca_rlist,"caYMissingV",((double*)FillValue_y)[0]);
    }
  }
  else if(!strcmp(type_y,"float")) {
    NhlRLSetMDFloatArray(ca_rlist,"caYArray",(float*)y,ndims_y,dsizes_y);

    if(is_missing_y) {
      NhlRLSetFloat(ca_rlist,"caYMissingV",((float*)FillValue_y)[0]);
    }
  }
  else if(!strcmp(type_y,"integer")) {
    NhlRLSetMDIntegerArray(ca_rlist,"caYArray",(int*)y,ndims_y,dsizes_y);

    if(is_missing_y) {
      NhlRLSetInteger(ca_rlist,"caYMissingV",((int*)FillValue_y)[0]);
    }
  }

  NhlCreate(&carray,"carray",NhlcoordArraysClass,app,ca_rlist);

  return(carray);
}


/*
 * This function creates a vector field object that will get
 * used with the vector or streamline object.
 */

int vector_field(void *u, void *v, const char *type_u, const char *type_v, 
                 int ylen, int xlen, 
                 int is_ycoord, void *ycoord, const char *type_ycoord, 
                 int is_xcoord, void *xcoord, const char *type_xcoord,
                 int is_missing_u, int is_missing_v,
                 void *FillValue_u, void *FillValue_v, int vf_rlist)
{
  int app, field, length[2];

/*
 * Retrieve application id.
 */
  app = NhlAppGetDefaultParentId();

/*
 * Create a vector field object that will be used as the
 * dataset for the vector or streamline object.
 */

  length[0] = ylen;
  length[1] = xlen;

  if(!strcmp(type_u,"double")) {
    NhlRLSetMDDoubleArray(vf_rlist,"vfUDataArray",(double*)u,2,length);
    
    if(is_missing_u) {
      NhlRLSetDouble(vf_rlist,"vfMissingUValueV",((double*)FillValue_u)[0]);
    }
  }
  else if(!strcmp(type_u,"float")) {
    NhlRLSetMDFloatArray(vf_rlist,"vfUDataArray",(float*)u,2,length);

    if(is_missing_u) {
      NhlRLSetFloat(vf_rlist,"vfMissingUValueV",((float*)FillValue_u)[0]);
    }
  }
  else if(!strcmp(type_u,"integer")) {
    NhlRLSetMDIntegerArray(vf_rlist,"vfUDataArray",(int*)u,2,length);

    if(is_missing_u) {
      NhlRLSetInteger(vf_rlist,"vfMissingUValueV",((int*)FillValue_u)[0]);
    }
  }

  if(!strcmp(type_v,"double")) {
    NhlRLSetMDDoubleArray(vf_rlist,"vfVDataArray",(double*)v,2,length);

    if(is_missing_v) {
      NhlRLSetDouble(vf_rlist,"vfMissingVValueV",((double*)FillValue_v)[0]);
    }
  }
  else if(!strcmp(type_v,"float")) {
    NhlRLSetMDFloatArray(vf_rlist,"vfVDataArray",(float*)v,2,length);

    if(is_missing_v) {
      NhlRLSetFloat(vf_rlist,"vfMissingVValueV",((float*)FillValue_v)[0]);
    }
  }
  else if(!strcmp(type_v,"integer")) {
    NhlRLSetMDIntegerArray(vf_rlist,"vfVDataArray",(int*)v,2,length);

    if(is_missing_v) {
      NhlRLSetInteger(vf_rlist,"vfMissingVValueV",((int*)FillValue_v)[0]);
    }
  }

/*
 * Check for coordinate arrays.
 */

  if(is_ycoord) {
    if(!strcmp(type_ycoord,"double")) {
      NhlRLSetDoubleArray(vf_rlist,"vfYArray",(double*)ycoord,ylen);
    }
    else if(!strcmp(type_ycoord,"float")) {
      NhlRLSetFloatArray(vf_rlist,"vfYArray",(float*)ycoord,ylen);
    }
    else if(!strcmp(type_ycoord,"integer")) {
      NhlRLSetIntegerArray(vf_rlist,"vfYArray",(int*)ycoord,ylen);
    }
  }

  if(is_xcoord) {
    if(!strcmp(type_xcoord,"double")) {
      NhlRLSetDoubleArray(vf_rlist,"vfXArray",(double*)xcoord,xlen);
    }
    else if(!strcmp(type_xcoord,"float")) {
      NhlRLSetFloatArray(vf_rlist,"vfXArray",(float*)xcoord,xlen);
    }
    else if(!strcmp(type_xcoord,"integer")) {
      NhlRLSetIntegerArray(vf_rlist,"vfXArray",(int*)xcoord,xlen);
    }
  }

  NhlCreate(&field,"field",NhlvectorFieldClass,app,vf_rlist);

  return(field);
}


/*
 * This function uses the HLUs to create an Application object
 * and to open a workstation.
 */

int gsn_open_wks(const char *type, const char *name, int wk_rlist)
{
  int wks, len;
  char *filename = (char *) NULL;
  int srlist, app;

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
 * Remove resource list.
 */

  NhlRLDestroy(srlist);

/*
 * Load color maps. This is necessary for access to the color maps in
 * $NCARG_ROOT/lib/ncarg/colormaps, not for the 9 built-in color maps.
 */
  NhlPalLoadColormapFiles(NhlworkstationClass,True);

/*
 * Start workstation code.
 */

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

  if(filename != NULL) free(filename);
  return(wks);
}

/*
 * This function uses the HLUs to create a contour plot.
 */

int gsn_contour_wrap(int wks, void *data, const char *type, 
                     int ylen, int xlen,
                     int is_ycoord, void *ycoord, const char *ycoord_type,
                     int is_xcoord, void *xcoord, const char *xcoord_type,
                     int is_missing, void *FillValue, 
                     int sf_rlist, int cn_rlist)
{
  int field, contour;

/*
 * Create a scalar field object that will be used as the
 * dataset for the contour object.
 */

  field = scalar_field(data, type, ylen, xlen, is_ycoord, ycoord,
                       ycoord_type, is_xcoord, xcoord, xcoord_type,
                       is_missing, FillValue, sf_rlist);

/*
 * Assign the data object that was created earlier.
 */

  NhlRLSetInteger(cn_rlist,"cnScalarFieldData",field);

/*
 * Create plot.
 */

  NhlCreate(&contour,"contour",NhlcontourPlotClass,wks,cn_rlist);

/*
 * Draw contour plot and advance frame.
 */

  maximize_plot(wks, contour);

  NhlDraw(contour);
  NhlFrame(wks);

/*
 * Return.
 */

  return(contour);
}


/*
 * This function uses the HLUs to create an XY plot.
 */

int gsn_xy_wrap(int wks, void *x, void *y, const char *type_x,
                const char *type_y, int ndims_x, int *dsizes_x,
                int ndims_y, int *dsizes_y, 
                int is_missing_x, int is_missing_y, 
                void *FillValue_x, void *FillValue_y,
                int ca_rlist, int xy_rlist, int xyd_rlist)
{
  int carray, xy, grlist;
  int num_dspec, *xyds;

/*
 * Create a coord arrays object that will be used as the
 * dataset for the xy object.
 */

  carray = coord_array(x, y, type_x, type_y, ndims_x, dsizes_x, 
                       ndims_y, dsizes_y, is_missing_x, is_missing_y, 
                       FillValue_x, FillValue_y, ca_rlist);

/*
 * Assign the data object that was created earlier.
 */

  NhlRLSetInteger(xy_rlist,"xyCoordData",carray);

/*
 * Create plot.
 */

  NhlCreate(&xy,"xy",NhlxyPlotClass,wks,xy_rlist);

/*
 * Get the DataSpec object id. This object is needed in order to 
 * set some of the XY resources, like line color, thickness, dash
 * patterns, etc. 
 */
    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetIntegerArray(grlist,NhlNxyCoordDataSpec,&xyds,&num_dspec);
    NhlGetValues(xy,grlist);
/*
 * Now apply the data spec resources.
 */
    NhlSetValues(*xyds,xyd_rlist);
    NhlFree(xyds);
/*
 * Draw xy plot and advance frame.
 */

  maximize_plot(wks, xy);

  NhlDraw(xy);
  NhlFrame(wks);

/*
 * Return.
 */

  return(xy);
}


/*
 * This function uses the HLUs to create a vector plot.
 */

int gsn_vector_wrap(int wks, void *u, void *v, const char *type_u,
                    const char *type_v, int ylen, int xlen, 
                    int is_ycoord, void *ycoord, const char *type_ycoord,
                    int is_xcoord, void *xcoord, const char *type_xcoord,
                    int is_missing_u, int is_missing_v, 
                    void *FillValue_u, void *FillValue_v,
                    int vf_rlist, int vc_rlist)
{
  int field, vector;

/*
 * Create a vector field object that will be used as the
 * dataset for the vector object.
 */

  field = vector_field(u, v, type_u, type_v, ylen, xlen, is_ycoord,
                       ycoord, type_ycoord, is_xcoord, xcoord, 
                       type_xcoord, is_missing_u, is_missing_v, 
                       FillValue_u, FillValue_v, vf_rlist);

/*
 * Assign the data object that was created earlier.
 */

  NhlRLSetInteger(vc_rlist,"vcVectorFieldData",field);

/*
 * Create plot.
 */

  NhlCreate(&vector,"vector",NhlvectorPlotClass,wks,vc_rlist);

/*
 * Draw vector plot and advance frame.
 */

  maximize_plot(wks, vector);

  NhlDraw(vector);
  NhlFrame(wks);

/*
 * Return.
 */

  return(vector);
}


/*
 * This function uses the HLUs to create a streamline plot.
 */

int gsn_streamline_wrap(int wks, void *u, void *v, const char *type_u,
                        const char *type_v, int ylen, int xlen, 
                        int is_ycoord, void *ycoord, const char *type_ycoord,
                        int is_xcoord, void *xcoord, const char *type_xcoord,
                        int is_missing_u, int is_missing_v, 
                        void *FillValue_u, void *FillValue_v, 
                        int vf_rlist, int st_rlist)
{
  int field, streamline;

/*
 * Create a vector field object that will be used as the
 * dataset for the streamline object.
 */

  field = vector_field(u, v, type_u, type_v, ylen, xlen, is_ycoord, ycoord,
                       type_ycoord, is_xcoord, xcoord, type_xcoord, 
                       is_missing_u, is_missing_v, FillValue_u, 
                       FillValue_v, vf_rlist);

/*
 * Assign the data object that was created earlier.
 */

  NhlRLSetInteger(st_rlist,"stVectorFieldData",field);

/*
 * Create plot.
 */

  NhlCreate(&streamline,"streamline",NhlstreamlinePlotClass,wks,st_rlist);

/*
 * Draw streamline plot and advance frame.
 */

  maximize_plot(wks, streamline);

  NhlDraw(streamline);
  NhlFrame(wks);

/*
 * Return.
 */

  return(streamline);
}

/*
 * This function uses the HLUs to create a map plot.
 */

int gsn_map_wrap(int wks, int mp_rlist)
{
  int map;

/*
 * Create plot.
 */

  NhlCreate(&map,"map",NhlmapPlotClass,wks,mp_rlist);

/*
 * Draw map plot and advance frame.
 */

  maximize_plot(wks, map);

  NhlDraw(map);
  NhlFrame(wks);

/*
 * Return.
 */

  return(map);
}



/*
 * This function uses the HLUs to overlay contours on a map.
 */

int gsn_contour_map_wrap(int wks, void *data, const char *type, 
                        int ylen, int xlen,
                        int is_ycoord, void *ycoord, const char *ycoord_type,
                        int is_xcoord, void *xcoord, const char *xcoord_type,
                        int is_missing, void *FillValue, 
                        int sf_rlist, int cn_rlist, int mp_rlist)
{
  int contour, map;

/*
 * Create contour plot.
 */

  contour = gsn_contour_wrap(wks, data, type, ylen, xlen,
                             is_ycoord, ycoord, ycoord_type,
                             is_xcoord, xcoord, xcoord_type,
                             is_missing, FillValue, sf_rlist, cn_rlist);


/*
 * Create map plot.
 */
  map = gsn_map_wrap(wks, mp_rlist);

/*
 * Overlay contour plot on map plot.
 */
  NhlAddOverlay(map,contour,-1);

/*
 * Draw plots and advance frame.
 */

  maximize_plot(wks, map);

  NhlDraw(map);
  NhlFrame(wks);

/*
 * Return.
 */

  return(map);
}


