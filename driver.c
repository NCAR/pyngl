#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gsun.h"

#define NCOLORS 17
#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))
#define NG  5

#define NCOLORMAPS 68

char *colormaps[NCOLORMAPS] = { \
                          "3gauss",  "3saw", "BkBlAqGrYeOrReViWh200",  \
                          "BkBlAqGrYeOrReViWh200", "BlAqGrYeOrRe", \
                          "BlAqGrYeOrRe",  "BlAqGrYeOrReVi200", \
                          "BlAqGrYeOrReVi200",  "BlGrYeOrReVi200", \
                          "BlGrYeOrReVi200",  "BlRe",  "BlWhRe", \
                          "BlWhRe",  "BlueRed",  "BlueRedGray", \
                          "GreenYellow",  "ViBlGrWhYeOrRe", \
                          "ViBlGrWhYeOrRe",  "WhBlGrYeRe", \
                          "WhBlGrYeRe",  "WhBlReWh", "WhViBlGrYeOrRe", \
                          "WhViBlGrYeOrRe", "WhViBlGrYeOrReWh", \
                          "WhViBlGrYeOrReWh", "amwg", "cosam", "cosam12", \
                          "cyclic", "default", "detail", "example", \
                          "extrema", "gscyclic", "gsltod", "gsdtol", \
                          "gui_default", "helix",  "helix1", \
                          "hlu_default", "hotres","ncview_default", \
                          "nrl_sirkes", "psgcap", "rainbow+gray", \
                          "rainbow+white+gray","rainbow+white", \
                          "rainbow",  "so4_21", "so4_23",  "tbrAvg1", \
                          "tbrStd1", "tbrStd1",  "tbrVar1",  "tbrVar1", \
                          "tbr_240-300",  "tbr_240-300", "tbr_stdev_0-30", \
                          "tbr_stdev_0-30", "tbr_var_0-500", \
                          "tbr_var_0-500", "temp1", "testcmap", "thelix", \
                          "uniform", "wgne15", "wh-bl-gr-ye-re", "wxpEnIR"}; 

/*
 * The type is being mixed here (double & float) for testing
 * purposes.
 */

double xmark1[6] = {0.02,0.03,0.04,0.05,0.06,0.07};
float  ymark1[6] = {0.01,0.01,0.01,0.01,0.01,0.01};
float  xmark2[3] = {0.06,0.05,0.04};
double ymark2[3] = {0.02,0.03,0.04};
float  xmark3[3] = {0.03,0.02,0.01};
float  ymark3[3] = {0.03,0.02,0.01};

float xline1[2]  = {0.01,0.99};
float yline1[2]  = {0.05,0.05};
float xline2[5]  = { 6, 15, 15, 6, 6 };
float yline2[5]  = { 271.5, 271.5, 273.1, 273.1, 271.5 };


float xgon[NG] = {0.80,0.90,0.90,0.80,0.80};
float ygon[NG] = {0.05,0.05,0.10,0.10,0.05};

int latgon[5] = {  20,  60,  60,  20,  20};
int longon[5] = {-125,-125, -65, -65, -125};

main()
{
/*
 * Declare variables for the HLU routine calls.
 */

  int wks, contour, xy, vector, streamline, map, text, text1, text2;
  int cntrmap, vctrmap, strmlnmap, *varray, *varray_copy;
  int wk_rlist, sf_rlist, sf2_rlist, ca_rlist, vf_rlist, tx_rlist;
  int cn_rlist, xy_rlist, xyd_rlist, vc_rlist, st_rlist, mp_rlist;
  int cn2_rlist, vc2_rlist, mp2_rlist, gs_rlist, am_rlist;
  int cncolors[]  = {2,16,30,44,58,52,86,100,114,128,142,156,170};
  int mpcolors[]  = {0, -1, 238, -1};
  int pttrns[]    = {0,1,2,3,4,5,6,7,8,9,10,11,12};
  int srlist, cmap_len[2];
  float xf, yf, cmap[NCOLORS][3], xm, ym;
  int   ixf, iyf, i1, i2, i3;
  int panel_dims[2];
  gsnRes special_res, special_pres;


/*
 * Declare variables for determining which plots to draw.
 */
  int do_colormaps, do_contour, do_xy_single, do_xy_multi, do_y, do_vector;
  int do_streamline, do_map, do_contour_map, do_contour_map2, do_vector_map;
  int do_streamline_map, do_vector_scalar, do_vector_scalar_map;
  int do_panel;

/*
 * Declare variables for getting information from netCDF file.
 */

  int ncid_T, ncid_U, ncid_V, ncid_P, ncid_T2;
  int id_T, id_U, id_V, id_P, id_T2;
  int lonid_T, latid_T, lonid_UV, latid_UV, timeid_UV;
  int nlon_T, nlat_T, nlat_UV, nlon_UV, ntime_UV, nlatlon_UV, ntimelatlon_UV;
  int ndims_T, dsizes_T[2], ndims_UV;
  int is_missing_T, is_missing_U, is_missing_V, is_missing_P, is_missing_T2;
  int attid, status;

/*
 * Declare variables to hold values, missing values, and types that
 * are read in from netCDF files.
 */
  void  *T, *U, *V, *P, *T2, *lat_T, *lon_T, *lat_UV, *lon_UV, *time_UV;
  void  *Unew, *Vnew, *T2new;
  void  *FillValue_T, *FillValue_U, *FillValue_V;
  void  *FillValue_P, *FillValue_T2;
  int is_lat_coord_T, is_lon_coord_T, is_lat_coord_UV, is_lon_coord_UV;
  int is_time_coord_UV;

  nc_type nctype_T, nctype_lat_T, nctype_lon_T;
  nc_type nctype_U, nctype_V, nctype_P;
  nc_type nctype_lat_UV, nctype_lon_UV, nctype_time_UV, nctype_T2;
  char type_T[TYPE_LEN], type_U[TYPE_LEN], type_V[TYPE_LEN];
  char type_P[TYPE_LEN], type_T2[TYPE_LEN];
  char type_lat_T[TYPE_LEN], type_lon_T[TYPE_LEN];
  char type_lat_UV[TYPE_LEN], type_lon_UV[TYPE_LEN];
  char type_time_UV[TYPE_LEN];

  size_t i, j, *start, *count;
  char  filename_T[256], filename_U[256], filename_V[256];
  char  filename_P[256], filename_T2[256];
  const char *dir = _NGGetNCARGEnv("data");

  float xmsg, ymsg;
  int nlines, *indices;

  xmsg = ymsg = -999;

/*
 * Open the netCDF files for contour and vector data.
 */

  sprintf(filename_T, "%s/cdf/meccatemp.cdf", dir );
  sprintf(filename_U, "%s/cdf/Ustorm.cdf", dir );
  sprintf(filename_V, "%s/cdf/Vstorm.cdf", dir );
  sprintf(filename_P, "%s/cdf/Pstorm.cdf", dir );
  sprintf(filename_T2, "%s/cdf/Tstorm.cdf", dir );

  nc_open(filename_T,NC_NOWRITE,&ncid_T);
  nc_open(filename_U,NC_NOWRITE,&ncid_U);
  nc_open(filename_V,NC_NOWRITE,&ncid_V);
  nc_open(filename_P,NC_NOWRITE,&ncid_P);
  nc_open(filename_T2,NC_NOWRITE,&ncid_T2);

/*
 * Get the lat/lon/time dimension ids so we can retrieve their lengths.
 * The lat/lon/time arrays for the U, V, and T2 data files are the same,
 * so we only need to retrieve one set of them.
 */

  nc_inq_dimid(ncid_T,"lat",&latid_T);
  nc_inq_dimid(ncid_T,"lon",&lonid_T);
  nc_inq_dimid(ncid_U,"lat",&latid_UV);
  nc_inq_dimid(ncid_U,"lon",&lonid_UV);
  nc_inq_dimid(ncid_U,"timestep",&timeid_UV);

  nc_inq_dimlen(ncid_T,latid_T, (size_t*)&nlat_T);
  nc_inq_dimlen(ncid_T,lonid_T, (size_t*)&nlon_T);
  nc_inq_dimlen(ncid_U,latid_UV,(size_t*)&nlat_UV);  
  nc_inq_dimlen(ncid_U,lonid_UV,(size_t*)&nlon_UV);
  nc_inq_dimlen(ncid_U,timeid_UV,(size_t*)&ntime_UV);

  nlatlon_UV     = nlat_UV * nlon_UV;
  ntimelatlon_UV = ntime_UV * nlatlon_UV;

/*
 * Get temperature, u, v, lat, and lon ids.
 */

  nc_inq_varid(ncid_T,  "t",&id_T);
  nc_inq_varid(ncid_U,  "u",&id_U);
  nc_inq_varid(ncid_V,  "v",&id_V);
  nc_inq_varid(ncid_P,  "p",&id_P);
  nc_inq_varid(ncid_T2, "t",&id_T2);
  nc_inq_varid(ncid_T,"lat",&latid_T);
  nc_inq_varid(ncid_T,"lon",&lonid_T);
  nc_inq_varid(ncid_U,"lat",&latid_UV);
  nc_inq_varid(ncid_U,"lon",&lonid_UV);
  nc_inq_varid(ncid_U,"timestep",&timeid_UV);

/*
 * Check if T, U, V, or T2 has a _FillValue attribute set.  If so,
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

  status = nc_inq_attid (ncid_P, id_P, "_FillValue", &attid); 
  if(status == NC_NOERR) {
    is_missing_P = 1;
  }
  else {
    is_missing_P = 0;
  }

  status = nc_inq_attid (ncid_T2, id_T2, "_FillValue", &attid); 
  if(status == NC_NOERR) {
    is_missing_T2 = 1;
  }
  else {
    is_missing_T2 = 0;
  }


/*
 * Get type and number of dimensions of T, then read in first
 * nlat_T x nlon_T subsection of T. Also, read in missing value
 * if one is set.
 */

  nc_inq_vartype  (ncid_T, id_T, &nctype_T);
  nc_inq_varndims (ncid_T, id_T, &ndims_T);

  start    = (size_t *)malloc(ndims_T*sizeof(size_t));
  count    = (size_t *)malloc(ndims_T*sizeof(size_t));
  for(i = 0; i < ndims_T; i++)   start[i] = 0;
  for(i = 0; i < ndims_T-2; i++) count[i] = 1;
  count[ndims_T-2] = nlat_T;
  count[ndims_T-1] = nlon_T;

  switch(nctype_T) {

  case NC_DOUBLE:
    strcpy(type_T,"double");
    T      = (double *)malloc(nlat_T*sizeof(double));

    nc_get_vara_double(ncid_T,id_T,start,count,(double*)T);

/*
 * Get double missing value.
 */
    if(is_missing_T) {
      FillValue_T = (double *)malloc(sizeof(double));
      nc_get_att_double (ncid_T, id_T, "_FillValue", (double*)FillValue_T); 
    }
    break;

  case NC_FLOAT:
    strcpy(type_T,"float");
    T      = (float *)malloc(nlat_T*nlon_T*sizeof(float));

    nc_get_vara_float(ncid_T,id_T,start,count,(float*)T);

/*
 * Get float missing value.
 */
    if(is_missing_T) {
      FillValue_T = (float *)malloc(sizeof(float));
      nc_get_att_float (ncid_T, id_T, "_FillValue", (float*)FillValue_T); 
    }
    break;

  case NC_INT:
    strcpy(type_T,"integer");
    T      = (int *)malloc(nlat_T*nlon_T*sizeof(int));

    nc_get_vara_int(ncid_T,id_T,start,count,(int*)T);

/*
 * Get integer missing value.
 */
    if(is_missing_T) {
      FillValue_T = (int *)malloc(sizeof(int));
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
    lat_T      = (double *)malloc(nlat_T*sizeof(double));

    nc_get_var_double(ncid_T,latid_T,(double*)lat_T);
    break;

  case NC_FLOAT:
    strcpy(type_lat_T,"float");
    lat_T      = (float *)malloc(nlat_T*sizeof(float));

    nc_get_var_float(ncid_T,latid_T,(float*)lat_T);

  case NC_INT:
    strcpy(type_lat_T,"integer");
    lat_T      = (int *)malloc(nlat_T*sizeof(int));

    nc_get_var_int(ncid_T,latid_T,(int*)lat_T);
    break;
  }

  is_lon_coord_T = 1;
  switch(nctype_lon_T) {

  case NC_DOUBLE:
    strcpy(type_lon_T,"double");
    lon_T      = (double *)malloc(nlon_T*sizeof(double));

    nc_get_var_double(ncid_T,lonid_T,(double*)lon_T);
    break;

  case NC_FLOAT:
    strcpy(type_lon_T,"float");
    lon_T      = (float *)malloc(nlon_T*sizeof(float));

    nc_get_var_float(ncid_T,lonid_T,(float*)lon_T);

  case NC_INT:
    strcpy(type_lon_T,"integer");
    lon_T      = (int *)malloc(nlon_T*sizeof(int));

    nc_get_var_int(ncid_T,lonid_T,(int*)lon_T);
    break;
  }

/*
 * Get type and number of dimensions of U, V, and T2, then read in first
 * ntime_UV x nlat_UV x nlon_UV subsection of "u", "v", and "t".
 * Also, read in missing values if ones are set.
 *
 * U, V, and T2 are assumed to have the same dimensions.
 */

  free(start);
  free(count);

  nc_inq_vartype  (ncid_U, id_U, &nctype_U);
  nc_inq_vartype  (ncid_V, id_V, &nctype_V);
  nc_inq_vartype  (ncid_P, id_P, &nctype_P);
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

  switch(nctype_P) {

  case NC_DOUBLE:
    strcpy(type_P,"double");
    P      = (double *)malloc(ntimelatlon_UV*sizeof(double));

/*
 * Get double values.
 */
    nc_get_vara_double(ncid_P,id_P,start,count,(double*)P);

/*
 * Get double missing value.
 */
    if(is_missing_P) {
      FillValue_P = (int *)malloc(sizeof(int));
      nc_get_att_float (ncid_P, id_P, "_FillValue", (float*)FillValue_P); 
    }
    break;

  case NC_FLOAT:
    strcpy(type_P,"float");
    P      = (float *)malloc(ntimelatlon_UV*sizeof(float));

/*
 * Get float values.
 */
    nc_get_vara_float(ncid_P,id_P,start,count,(float*)P);

/*
 * Get float missing value.
 */
    if(is_missing_P) {
      FillValue_P = (int *)malloc(sizeof(int));
      nc_get_att_float (ncid_P, id_P, "_FillValue", (float*)FillValue_P); 
    }
    for(i = 0; i < ntimelatlon_UV; i++) {
      if(!is_missing_P ||
         (is_missing_P && ((float*)P)[i] != ((float*)FillValue_P)[0])) {
        ((float*)P)[i] = (((float*)P)[i] * 0.01);
      }
    }
    break;

  case NC_INT:
    strcpy(type_P,"integer");
    P      = (int *)malloc(ntimelatlon_UV*sizeof(int));

/*
 * Get integer values.
 */
    nc_get_vara_int(ncid_P,id_P,start,count,(int*)P);

/*
 * Get integer missing value.
 */
    if(is_missing_P) {
      FillValue_P = (int *)malloc(sizeof(int));
      nc_get_att_float (ncid_P, id_P, "_FillValue", (float*)FillValue_P); 
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

  ncclose(ncid_T);
  ncclose(ncid_U);
  ncclose(ncid_V);
  ncclose(ncid_P);
  ncclose(ncid_T2);


/*----------------------------------------------------------------------*
 *
 * Start graphics portion of code.
 *
 *----------------------------------------------------------------------*/
/*
 * Initialize special resources.  For the plotting routines, draw, frame,
 * and maximize default to True (1).  For primitive and text routines,
 * only draw defaults to True.
 */
  special_res.gsnDraw     = 1;
  special_res.gsnFrame    = 1;
  special_res.gsnMaximize = 1;
  special_res.gsnScale    = 0;
  special_res.gsnDebug    = 0;

  special_pres.gsnDraw    = 1;
  special_pres.gsnFrame   = 0;
  special_pres.gsnMaximize= 0;
  special_pres.gsnDebug   = 0;

/*
 * Initialize which plots to draw.
 */
  do_colormaps         = 0;
  do_contour           = 0;
  do_xy_single         = 0;
  do_xy_multi          = 0;
  do_y                 = 0;
  do_vector            = 0;
  do_streamline        = 0;
  do_map               = 0;
  do_contour_map       = 0;
  do_contour_map2      = 0;
  do_vector_map        = 0;
  do_streamline_map    = 0;
  do_vector_scalar     = 0;
  do_vector_scalar_map = 0;
  do_panel             = 1;

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
  wks = ngl_open_wks_wrap("ps","test", wk_rlist);

/* 
 * Set color map resource and open workstation.
 */

  if(do_colormaps) {
    for(i = 0; i < NCOLORMAPS; i++) {
      NhlRLClear(wk_rlist);
      NhlRLSetString(wk_rlist,"wkColorMap",colormaps[i]);
      (void)NhlSetValues(wks, wk_rlist);
      ngl_draw_colormap_wrap(wks);
    }
  }

/*
 * Initialize colormap back to "rainbow+gray".
 */
  NhlRLClear(wk_rlist);
  NhlRLSetString(wk_rlist,"wkColorMap","rainbow+gray");
  (void)NhlSetValues(wks, wk_rlist);

/*
 * Initialize and clear resource lists.
 */

  sf_rlist  = NhlRLCreate(NhlSETRL);
  sf2_rlist = NhlRLCreate(NhlSETRL);
  ca_rlist  = NhlRLCreate(NhlSETRL);
  vf_rlist  = NhlRLCreate(NhlSETRL);
  tx_rlist  = NhlRLCreate(NhlSETRL);
  am_rlist  = NhlRLCreate(NhlSETRL);
  cn_rlist  = NhlRLCreate(NhlSETRL);
  cn2_rlist = NhlRLCreate(NhlSETRL);
  xy_rlist  = NhlRLCreate(NhlSETRL);
  xyd_rlist = NhlRLCreate(NhlSETRL);
  vc_rlist  = NhlRLCreate(NhlSETRL);
  vc2_rlist = NhlRLCreate(NhlSETRL);
  st_rlist  = NhlRLCreate(NhlSETRL);
  mp_rlist  = NhlRLCreate(NhlSETRL);
  mp2_rlist = NhlRLCreate(NhlSETRL);
  gs_rlist  = NhlRLCreate(NhlSETRL);
  NhlRLClear(sf_rlist);
  NhlRLClear(sf2_rlist);
  NhlRLClear(ca_rlist);
  NhlRLClear(vf_rlist);
  NhlRLClear(tx_rlist);
  NhlRLClear(am_rlist);
  NhlRLClear(cn_rlist);
  NhlRLClear(cn2_rlist);
  NhlRLClear(xy_rlist);
  NhlRLClear(xyd_rlist);
  NhlRLClear(vc_rlist);
  NhlRLClear(vc2_rlist);
  NhlRLClear(st_rlist);
  NhlRLClear(mp_rlist);
  NhlRLClear(mp2_rlist);
  NhlRLClear(gs_rlist);

/*
 * ngl_contour section
 */

  if(do_contour) {

/*
 * Set some contour resources.
 */

    NhlRLSetString      (cn_rlist, "cnFillOn"             , "True");
    NhlRLSetIntegerArray(cn_rlist, "cnFillColors"         , cncolors,13);
    NhlRLSetString      (cn_rlist, "cnLineLabelsOn"       , "False");
    NhlRLSetString      (cn_rlist, "lbPerimOn"            , "False");
    NhlRLSetString      (cn_rlist, "pmLabelBarDisplayMode", "ALWAYS");

/*
 * Set some text resources and draw a text string.
 */
    NhlRLClear(tx_rlist);
    NhlRLSetFloat  (tx_rlist,"txFontHeightF", 0.02);
    NhlRLSetInteger(tx_rlist,"txFont"       , 21);
    NhlRLSetString (tx_rlist,"txJust"       , "CenterLeft");
    NhlRLSetString (tx_rlist,"txFuncCode"   , "~");

    xf = 0.01;
    yf = 0.05;
    text = ngl_text_ndc_wrap(wks,"ngl_text_ndc: bottom feeder",
                             &xf,&yf,"float","float",tx_rlist,&special_pres);

/*
 * Draw a polygon before we draw the plot.
 */

    NhlRLClear(gs_rlist);
    NhlRLSetString (gs_rlist,"gsFillColor"     , "SlateBlue");
    NhlRLSetString (gs_rlist,"gsEdgesOn"       , "True");
    NhlRLSetFloat  (gs_rlist,"gsEdgeThicknessF", 3.0);
    NhlRLSetString (gs_rlist,"gsEdgeColor"     , "Salmon");
    ngl_polygon_ndc_wrap(wks, (void *)xgon, (void *)ygon, "float", "float", 
                         NG, 0, 0, NULL, NULL, gs_rlist, &special_pres);

/*
 * Create and draw contour plot, and advance frame.
 */
    contour = ngl_contour_wrap(wks, T, type_T, nlat_T, nlon_T, 
                               is_lat_coord_T, lat_T, type_lat_T, 
                               is_lon_coord_T, lon_T, type_lon_T, 
                               is_missing_T, FillValue_T, sf_rlist, cn_rlist,
                               &special_res);
  }

/*
 * ngl_y section
 *
 * Index values are used for the X axis.
 */

  if(do_y) {
    NhlRLClear(tx_rlist);
    NhlRLSetString(tx_rlist,"txFuncCode"   , "~");
    NhlRLSetFloat (tx_rlist,"txFontHeightF", 0.03);
    NhlRLSetString (tx_rlist,"txJust"      , "TopLeft");
    NhlRLSetString(tx_rlist,"txDirection"  , "Down");

    xf = 0.3;
    yf = 0.8;
    text = ngl_text_ndc_wrap(wks,"ngl_text_ndc",&xf,&yf,"float","float",
                             tx_rlist,&special_pres);
    xf = 0.35;
    text = ngl_text_ndc_wrap(wks,"Down",&xf,&yf,"float","float",tx_rlist,
                             &special_pres);


    special_res.gsnFrame = 0;

    xy = ngl_y_wrap(wks, T, type_T, 1, &nlon_T, is_missing_T, FillValue_T, 
                    ca_rlist, xy_rlist, xyd_rlist, &special_res);

    special_res.gsnFrame = 1;

    NhlRLClear(gs_rlist);
    NhlRLSetString (gs_rlist,"gsLineColor",       "red");
    NhlRLSetInteger(gs_rlist,"gsLineDashPattern", 11);
    ngl_polyline_wrap(wks, xy, (void *)xline2, (void *)yline2, "float",
                      "float", 5, 0, 0, NULL, NULL, 
                      gs_rlist, &special_pres);

    NhlFrame(wks);
  }

/*
 * ngl_xy section
 */

  if(do_xy_multi) {
    NhlRLSetFloat  (xy_rlist,  "trXMinF",          -180);
    NhlRLSetFloat  (xy_rlist,  "trXMaxF",           180);
    NhlRLSetString (xy_rlist,  "tiMainFont",       "helvetica-bold");
    NhlRLSetString (xy_rlist,  "tiMainFontColor",  "red");
    NhlRLSetString (xy_rlist,  "tiMainString",     "This is a boring red title");

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


    special_res.gsnFrame = 0;
    special_res.gsnDraw  = 0;

    xy = ngl_xy_wrap(wks, lon_T, T, type_lon_T, type_T, 1, &nlon_T, 
                     2, &dsizes_T[0], 0, is_missing_T, NULL, FillValue_T, 
                     ca_rlist, xy_rlist, xyd_rlist, &special_res);

    special_res.gsnDraw  = 1;
    special_res.gsnFrame = 1;

/*
 * Set some text resources and draw a text string.
 */
    NhlRLClear(tx_rlist);
    NhlRLSetFloat (tx_rlist,"txAngleF"     ,   90.);
    NhlRLSetFloat (tx_rlist,"txFontHeightF", 0.03);
    NhlRLSetString(tx_rlist,"txFuncCode"   , "~");

    ixf = -130;
    iyf =  268;
    text1 = ngl_add_text_wrap(wks, xy, "~F26~ngl_add_text:~C~sideways", 
                             &ixf, &iyf, "integer","integer",tx_rlist,
                             am_rlist, &special_pres);

    NhlDraw(xy);

    NhlRLSetFloat  (tx_rlist,"txAngleF",     270.);
    NhlRLSetString (tx_rlist,"txFontColor",  "green");
    ixf = -80;
    text2 = ngl_text_wrap(wks, xy, "~F26~ngl_text:~C~green, sideways", 
                         &ixf, &iyf,"integer","integer",tx_rlist,
                          &special_pres);

    NhlFrame(wks);
  }

/* 
 * Plot just a single line.
 */
  if(do_xy_single) {
    NhlRLClear     (xy_rlist);
    NhlRLClear     (xyd_rlist);
    NhlRLSetString (xyd_rlist, "xyLineColor"     , "green");
    NhlRLSetFloat  (xyd_rlist, "xyLineThicknessF", 3.0);

    special_res.gsnFrame = 0;

    xy = ngl_xy_wrap(wks, lat_T, T, type_lat_T, type_T, 1, &nlat_T, 
                     1, &nlon_T, 0, 0, NULL, NULL, 
                     ca_rlist, xy_rlist, xyd_rlist, &special_res);

    special_res.gsnFrame = 1;

    NhlRLClear(gs_rlist);
    NhlRLSetString (gs_rlist,"gsMarkerColor", "Red");
    NhlRLSetInteger(gs_rlist,"gsMarkerIndex", 17);
    NhlRLSetFloat  (gs_rlist,"gsMarkerSizeF", 0.02);
    xm = -70.;
    ym = 271.5;
    ngl_polymarker_wrap(wks, xy, (void *)&xm, (void *)&ym, "float", "float", 
                        1, 0, 0, NULL, NULL,  gs_rlist, &special_pres);

    xm = -50.;
    NhlRLSetFloat  (gs_rlist,"gsMarkerSizeF", 0.03);
    ngl_polymarker_wrap(wks, xy, (void *)&xm, (void *)&ym, "float", "float", 
                        1, 0, 0, NULL, NULL,  gs_rlist, &special_pres);

    xm = -30.;
    NhlRLSetFloat  (gs_rlist,"gsMarkerSizeF", 0.04);
    ngl_polymarker_wrap(wks, xy, (void *)&xm, (void *)&ym, "float", "float", 
                        1, 0, 0, NULL, NULL,  gs_rlist, &special_pres);

    xm = -10.;
    NhlRLSetFloat  (gs_rlist,"gsMarkerSizeF", 0.05);
    ngl_polymarker_wrap(wks, xy, (void *)&xm, (void *)&ym, "float", "float", 
                        1, 0, 0, NULL, NULL,  gs_rlist, &special_pres);


    xm =  10.;
    NhlRLSetFloat  (gs_rlist,"gsMarkerSizeF", 0.06);
    ngl_polymarker_wrap(wks, xy, (void *)&xm, (void *)&ym, "float", "float", 
                        1, 0, 0, NULL, NULL,  gs_rlist, &special_pres);

    NhlFrame(wks);
  }

/*
 * ngl_streamline section
 */

  if(do_streamline) {

/*
 * Create and draw streamline plot, and advance frame.
 */
    streamline = ngl_streamline_wrap(wks, U, V, type_U, type_V, 
                                     nlat_UV, nlon_UV, 
                                     is_lat_coord_UV, lat_UV, type_lat_UV, 
                                     is_lon_coord_UV, lon_UV, type_lon_UV, 
                                     is_missing_U, is_missing_V, 
                                     FillValue_U, FillValue_V,
                                     vf_rlist, st_rlist, &special_res);
  }

/*
 * ngl_vector section
 */

  if(do_vector) {

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

    vector = ngl_vector_wrap(wks, U, V, type_U, type_V, 
                             nlat_UV, nlon_UV, 
                             is_lat_coord_UV, lat_UV, type_lat_UV, 
                             is_lon_coord_UV, lon_UV, type_lon_UV, 
                             is_missing_U, is_missing_V, 
                             FillValue_U, FillValue_V,
                             vf_rlist, vc_rlist, &special_res);
  }

/*
 * ngl_map section
 */

  if(do_map) {

/*
 * Set some map resources.
 */

    NhlRLSetString  (mp_rlist, "mpGridAndLimbOn",    "False");
    NhlRLSetInteger (mp_rlist, "mpPerimOn",          1);
    NhlRLSetString  (mp_rlist, "pmTitleDisplayMode", "Always");
    NhlRLSetString  (mp_rlist, "tiMainString",       "CylindricalEquidistant");

/*
 * Create and draw map plot, and advance frame.
 */

    map = ngl_map_wrap(wks, mp_rlist, &special_res);
  }

/*
 * ngl_contour_map section
 */

  if(do_contour_map) {

/*
 * Set up some resources.
 */
    NhlRLSetString      (cn2_rlist, "cnFillOn",              "True");
    NhlRLSetIntegerArray(cn2_rlist, "cnFillColors",          cncolors,13);
    NhlRLSetString      (cn2_rlist, "cnLinesOn",             "False");
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
    NhlRLSetString      (mp2_rlist, "mpGeophysicalLineColor","black");
    NhlRLSetInteger     (mp2_rlist, "mpPerimOn",             1);

    cntrmap = ngl_contour_map_wrap(wks, T, type_T, nlat_T, nlon_T, 
                                   is_lat_coord_T, lat_T, type_lat_T, 
                                   is_lon_coord_T, lon_T, type_lon_T, 
                                   is_missing_T, FillValue_T, 
                                   sf_rlist, cn2_rlist, mp2_rlist,
                                   &special_res);
  }

/*
 * ngl_contour_map section
 */

  if(do_contour_map2) {
/* 
 * Ser color map back to the default.
 */
    srlist = NhlRLCreate(NhlSETRL);
    NhlRLClear(srlist);
    NhlRLSetString(srlist,NhlNwkColorMap,"hlu_default");
    NhlRLSetString(srlist,NhlNwkForegroundColor,"black");
    NhlRLSetString(srlist,NhlNwkBackgroundColor,"white");
    (void)NhlSetValues(wks, srlist);

    NhlRLClear(cn_rlist);
    NhlRLClear(sf_rlist);
    NhlRLClear(mp_rlist);

/*
 * Set up some resources.
 */
    NhlRLSetString(cn_rlist,"tiXAxisString" , ":F25:longitude");
    NhlRLSetString(cn_rlist,"tiYAxisString" , ":F25:latitude");

    NhlRLSetString(cn_rlist,"cnFillOn"              , "True");
    NhlRLSetString(cn_rlist,"cnLineLabelsOn"        , "False");
    NhlRLSetString(cn_rlist,"cnInfoLabelOn"         , "False");
    NhlRLSetString(cn_rlist,"pmLabelBarDisplayMode" , "Always");
    NhlRLSetString(cn_rlist,"lbPerimOn"             , "False");

    NhlRLSetFloat(sf_rlist,"sfXCStartV" , -140.0);
    NhlRLSetFloat(sf_rlist,"sfXCEndV"   ,  -52.5);
    NhlRLSetFloat(sf_rlist,"sfYCStartV" ,   20.0);
    NhlRLSetFloat(sf_rlist,"sfYCEndV"   ,   60.0);

    NhlRLSetString(mp_rlist,"mpProjection" , "LambertEqualArea");
    NhlRLSetFloat(mp_rlist,"mpCenterLonF" , -96.25);
    NhlRLSetFloat(mp_rlist,"mpCenterLatF" ,  40.0);

    NhlRLSetString(mp_rlist,"mpLimitMode" , "LatLon");
    NhlRLSetFloat(mp_rlist,"mpMinLonF"   , -140.0);
    NhlRLSetFloat(mp_rlist,"mpMaxLonF"   ,  -52.5);
    NhlRLSetFloat(mp_rlist,"mpMinLatF"   ,   20.0);
    NhlRLSetFloat(mp_rlist,"mpMaxLatF"   ,   60.0);
    NhlRLSetString(mp_rlist,"mpPerimOn"  , "True");

    NhlRLSetString(cn_rlist,"tiMainString" , ":F26:January 1996 storm");

    NhlRLSetFloat(mp_rlist,"vpXF"      , 0.1);
    NhlRLSetFloat(mp_rlist,"vpYF"      , 0.9);
    NhlRLSetFloat(mp_rlist,"vpWidthF"  , 0.7);
    NhlRLSetFloat(mp_rlist,"vpHeightF" , 0.7);

    special_res.gsnScale    = 1;
    special_res.gsnFrame    = 0;
    special_res.gsnMaximize = 0;

    cntrmap = ngl_contour_map_wrap(wks, P, type_P, nlat_UV, nlon_UV, 
                                   is_lat_coord_UV, lat_UV, type_lat_UV, 
                                   is_lon_coord_UV, lon_UV, type_lon_UV, 
                                   is_missing_P, FillValue_P, 
                                   sf_rlist, cn_rlist, mp_rlist,
                                   &special_res);
    special_res.gsnMaximize = 1;
    special_res.gsnFrame    = 1;
    special_res.gsnScale    = 0;

/*
 * Set some text resources.
 */

    NhlRLClear (tx_rlist);
    NhlRLSetFloat(tx_rlist,"txFontHeightF" , 0.025);
    NhlRLSetInteger(tx_rlist,"txFontColor"   , 4);
    xf = 0.45;
    yf = 0.25;
    text = ngl_text_ndc_wrap(wks,":F25:Pressure (mb)",&xf,&yf,"float",
                             "float", tx_rlist,&special_pres);
    
    NhlFrame(wks);
  }

/* 
 * Define a new color map.
 */
  cmap_len[0] = NCOLORS;
  cmap_len[1] = 3;
  srlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(srlist);
  NhlRLSetMDFloatArray(srlist,NhlNwkColorMap,&cmap[0][0],2,cmap_len);
 (void)NhlSetValues(wks, srlist);
/*
 * ngl_vector_map section
 */

  if(do_vector_scalar || do_panel) {

/*
 * First set up some resources.
 */

    NhlRLSetString (vc2_rlist,"pmLabelBarDisplayMode"  , "Always");
    NhlRLSetString (vc2_rlist,"pmLabelBarSide"         , "Bottom");
    NhlRLSetString (vc2_rlist,"lbOrientation"          , "Horizontal");
    NhlRLSetString (vc2_rlist,"lbPerimOn"              , "False");
    NhlRLSetInteger(vc2_rlist,"lbTitleFont"            , 25);
    NhlRLSetString (vc2_rlist,"lbTitleString"          , "TEMPERATURE (:S:o:N:F)");
    NhlRLSetFloat  (vc2_rlist,"tiMainFontHeightF"      , 0.03);
    NhlRLSetString (vc2_rlist,"tiMainString"           , ":F25:Wind velocity vectors");
    
    NhlRLSetInteger(vc2_rlist,"vcFillArrowEdgeColor"    , 1);
    NhlRLSetString (vc2_rlist,"vcFillArrowsOn"          , "True");
    NhlRLSetFloat  (vc2_rlist,"vcMinFracLengthF"        , 0.33);
    NhlRLSetFloat  (vc2_rlist,"vcMinMagnitudeF"         , 0.001);
    NhlRLSetString (vc2_rlist,"vcMonoFillArrowFillColor", "False");
    NhlRLSetString (vc2_rlist,"vcMonoLineArrowColor"    , "False");
    NhlRLSetFloat  (vc2_rlist,"vcRefLengthF"            , 0.045);
    NhlRLSetFloat  (vc2_rlist,"vcRefMagnitudeF"         , 20.0);
/*
 * Draw some polymarkers before we draw the plot.
 */

    if(do_vector_scalar) {
      NhlRLClear(gs_rlist);
      NhlRLSetInteger(gs_rlist,"gsMarkerIndex", 16);
      NhlRLSetFloat  (gs_rlist,"gsMarkerSizeF", 10.5);
      NhlRLSetString (gs_rlist,"gsMarkerColor", "red");
      ngl_polymarker_ndc_wrap(wks, (void *)xmark1, (void *)ymark1, "double",
                              "float", 6, 0, 0, NULL, NULL, gs_rlist,
                              &special_pres);
      NhlRLSetString (gs_rlist,"gsMarkerColor", "green");
      ngl_polymarker_ndc_wrap(wks, (void *)xmark2, (void *)ymark2, "float", 
                              "double", 3, 0, 0, NULL, NULL, gs_rlist, 
                              &special_pres);
      NhlRLSetString (gs_rlist,"gsMarkerColor", "blue");
      ngl_polymarker_ndc_wrap(wks, (void *)xmark3, (void *)ymark3, "float",
                              "float", 3, 0, 0, NULL, NULL, gs_rlist,
                              &special_pres);

/*
 * Now create and draw plot.
 */
      vctrmap = ngl_vector_scalar_wrap(wks, U, V, T2, type_U, type_V,
                                       type_T2, nlat_UV, nlon_UV, 
                                       is_lat_coord_UV, lat_UV, type_lat_UV, 
                                       is_lon_coord_UV, lon_UV, type_lon_UV,
                                       is_missing_U, is_missing_V, 
                                       is_missing_T2, FillValue_U, 
                                       FillValue_V, FillValue_T2,
                                       vf_rlist, sf2_rlist, vc2_rlist, 
                                       &special_res);
    }

    if(do_panel) {
/*
 * Loop across time dimension and create a vector plot for each one.
 * Then, we'll create several panel plots.
 * Create a separate array to hold the same plots, but set some of
 * them to "0" (missing).
 */
      varray      = (int*)malloc(ntime_UV*sizeof(int));
      varray_copy = (int*)malloc(ntime_UV*sizeof(int));

      special_res.gsnMaximize = 0;
      special_res.gsnFrame    = 0;
      special_res.gsnDraw     = 0;
      for(i = 0; i < ntime_UV; i++) {
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
                           vf_rlist, sf2_rlist,
                           vc2_rlist, &special_res);

      }
      varray_copy[i] = varray[i];
/*
 * Paper orientation: -1 is auto, 0 is portrait, and 6 is landscape.
 */
      special_res.gsnPaperOrientation = -1;
      special_res.gsnPaperWidth       =  8.5;
      special_res.gsnPaperHeight      = 11.0;
      special_res.gsnPaperMargin      =  0.5;

/*
 * Special resources for paneling.
 */
      special_res.gsnDebug                   = 1;
      special_res.gsnPanelCenter             = 1;
      special_res.gsnPanelRowSpec            = 0;
      special_res.gsnPanelXWhiteSpacePercent = 1.;
      special_res.gsnPanelYWhiteSpacePercent = 1.;
      special_res.gsnPanelBoxes              = 0;
      special_res.gsnPanelLeft               = 0.;
      special_res.gsnPanelRight              = 1.;
      special_res.gsnPanelBottom             = 0.;
      special_res.gsnPanelTop                = 1.;
      special_res.gsnPanelInvsblTop          = -999;
      special_res.gsnPanelInvsblLeft         = -999;
      special_res.gsnPanelInvsblRight        = -999;
      special_res.gsnPanelInvsblBottom       = -999;
      special_res.gsnPanelSave               = 0;

      special_res.gsnMaximize = 1;
      special_res.gsnFrame    = 1;
      special_res.gsnDraw     = 1;

/*
 * This first loop is to do an initial run-through where no plots
 * are missing, and then the second run-through will set some plots
 * to missing.
 */
      for(i = 0; i <= 1; i++) {
/*
 * Make sure text strings default to horizontal orientation.
 */
        NhlRLSetFloat(tx_rlist,"txAngleF" , 0.);

        if(i) {
/*
 * We're going to be doing missing plots, so for these, we always
 * want a text string. Since the text string has to be drawn after
 * the paneled plots, we need to turn off frame for the paneled plots,
 * and turn it on for the text strings.
 */
          special_res.gsnFrame  = 0;
          special_pres.gsnFrame = 1;
        }

        panel_dims[0] = 3;          /* rows    */
        panel_dims[1] = 1;          /* columns */
        if(i) {
          varray[0] = 0;
        }
        ngl_panel_wrap(wks, varray, 3, panel_dims, 2, 
                       &special_res);

/*
        ngl_panel_wrap(wks, varray, ntime_UV, panel_dims, 2, 
                       &special_res);
 */
        if(i) {
          varray[0] = varray_copy[0];
          xf = 0.5;
          yf = 0.80;
          text = ngl_text_ndc_wrap(wks,":F25:3 rows/1 column:C:plot 0 is missing",
                       &xf,&yf,"float","float",
                       tx_rlist,&special_pres);
        }
      
        panel_dims[0] = 1;          /* rows    */
        panel_dims[1] = 3;          /* columns */
        if(i) {
          varray[0] = 0;
        }
        ngl_panel_wrap(wks, varray, 3, panel_dims, 2, 
                   &special_res);
/*
        ngl_panel_wrap(wks, varray, ntime_UV, panel_dims, 2, 
                   &special_res);
 */

        if(i) {
          varray[0] = varray_copy[0];
          xf = 0.5;
          yf = 0.98;
          text = ngl_text_ndc_wrap(wks,":F25:1 row/3 columns:C:Plot 0 is missing",
                       &xf,&yf,"float","float",tx_rlist,
                       &special_pres);
        }

        panel_dims[0] = 2;          /* rows    */
        panel_dims[1] = 2;          /* columns */

        if(i) {
          varray[2] = varray[3] = 0;
        }
        ngl_panel_wrap(wks, varray, 4, panel_dims, 2, 
                       &special_res);

/*
        ngl_panel_wrap(wks, varray, ntime_UV, panel_dims, 2, 
                       &special_res);
*/
        if(i) {
          varray[2] = varray_copy[2];
          varray[3] = varray_copy[3];
          xf = 0.5;
          yf = 0.98;
          text = ngl_text_ndc_wrap(wks,":F25:2 rows/2 columns:C:Plots 2/3 are missing",
                                   &xf,&yf,"float","float",tx_rlist,
                                   &special_pres);
        }

/*
 * We need to test various aspects of panelling:
 *   - adding a string at the top, bottom, left, and right
 *   - changing the margins
 *   - changing the spacing around the plots
 *   - playing w/gsnPanelCenter when you have missing plots
 *   - gsnPanelRowSpec
 */
        special_res.gsnFrame  = 0;
        special_pres.gsnFrame = 1;
        NhlRLClear(tx_rlist);
        NhlRLSetFloat(tx_rlist,"txFontHeightF" , 0.020);
/*
 * Testing gsnPanelBottom.
 */
        printf("Testing gsnPanelBottom...\n");

        special_res.gsnPanelBottom = 0.07;
        panel_dims[0] = 2;          /* rows    */
        panel_dims[1] = 3;          /* columns */

        if(i) {
          varray[0] = varray[2] = 0;
        }

        ngl_panel_wrap(wks, varray, 6, panel_dims, 2, 
                       &special_res);
/*
        ngl_panel_wrap(wks, varray, ntime_UV, panel_dims, 2, 
                       &special_res);
 */
        xf = 0.5;
        yf = 0.04;
        if(i) {
          varray[0] = varray_copy[0];
          varray[2] = varray_copy[2];
          text = ngl_text_ndc_wrap(wks,":F25:2 rows/3 columns:C:Plots 0/2 are missing",
                                   &xf,&yf,"float","float",tx_rlist,
                                   &special_pres);
        }
        else {
          text = ngl_text_ndc_wrap(wks,":F26:2 rows/3 columns:C:Panel string at the bottom",
                                   &xf,&yf,"float","float",tx_rlist,
                                   &special_pres);
        }

/*
 * Testing gsnPanelTop.
 */
        printf("Testing gsnPanelTop...\n");

        special_res.gsnPanelTop    = 0.94;
        special_res.gsnPanelBottom = 0.0;
        panel_dims[0] = 4;          /* rows    */
        panel_dims[1] = 2;          /* columns */

        if(i) {
          varray[4] = varray[5] = 0;
        }

        ngl_panel_wrap(wks, varray, 8, panel_dims, 2, &special_res);

        xf = 0.5;
        yf = 0.97;
        if(i) {
          varray[4] = varray_copy[4];
          varray[5] = varray_copy[5];
          text = ngl_text_ndc_wrap(wks,":F25:4 rows/2 columns:C:Plots 4/5 are missing",
                                   &xf,&yf,"float","float",tx_rlist,
                                   &special_pres);
        }
        else {
          text = ngl_text_ndc_wrap(wks,":F26:4 rows/2 columns:C:Panel string at the top",
                                   &xf,&yf,"float","float",tx_rlist,
                                   &special_pres);
        }

/*
 * Testing gsnPanelRight.
 */
        printf("Testing gsnPanelRight...\n");

        special_res.gsnPanelTop    = 1.0;
        special_res.gsnPanelRight  = 0.94;
        panel_dims[0] = 2;          /* rows    */
        panel_dims[1] = 4;          /* columns */
        if(i) {
          varray[3] = 0;
          varray[7] = 0;
        }
        ngl_panel_wrap(wks, varray, 8, panel_dims, 2, &special_res);
        
        xf = 0.97;
        yf = 0.5;
        NhlRLSetFloat(tx_rlist,"txAngleF" , 90.);
        if(i) {
          varray[3] = varray_copy[3];
          varray[7] = varray_copy[7];
          text = ngl_text_ndc_wrap(wks,":F25:2 rows/4 columns:C:Plots 3/7 are missing",
                                   &xf,&yf,"float","float",tx_rlist,
                                   &special_pres);
        }
        else {
          text = ngl_text_ndc_wrap(wks,":F26:2 rows/4 columns:C:Panel string on the right",
                                   &xf,&yf,"float","float",tx_rlist,
                                   &special_pres);
        }

/*
 * Testing gsnPanelLeft.
 */
        printf("Testing gsnPanelLeft...\n");

        special_res.gsnPanelRight  = 1.0;
        special_res.gsnPanelLeft   = 0.07;
        panel_dims[0] = 3;          /* rows    */
        panel_dims[1] = 4;          /* columns */

        if(i) {
          varray[2] = 0;
          varray[5] = 0;
          varray[7] = 0;
          varray[11] = 0;
        }

        ngl_panel_wrap(wks, varray, 12, panel_dims, 2, &special_res);
        
        xf = 0.04;
        yf = 0.5;
        if(i) {
          varray[2] = varray_copy[2];
          varray[5] = varray_copy[5];
          varray[7] = varray_copy[7];
          varray[11] = varray_copy[11];
          text = ngl_text_ndc_wrap(wks,":F25:3 rows/4 columns:C:Plots 2/5/7/11 are missing",
                                   &xf,&yf,"float","float",tx_rlist,
                                   &special_pres);
        }
        else {
          text = ngl_text_ndc_wrap(wks,":F26:3 rows/4 columns:C:Panel string on the left",
                                   &xf,&yf,"float","float",tx_rlist,
                                   &special_pres);
        }
        
        special_res.gsnPanelTop    = 1.0;
        special_res.gsnPanelBottom = 0.0;
        special_res.gsnPanelRight  = 1.0;
        special_res.gsnPanelLeft   = 0.0;
        special_pres.gsnFrame      = 0;
        special_res.gsnFrame       = 1;
      }
    }
  }

/*
 * ngl_vector_map section
 */

  if(do_vector_map) {

/*
 * Set some vector resources.
 */

    NhlRLClear(vf_rlist);
    NhlRLClear(vc2_rlist);

    NhlRLSetInteger(vc2_rlist,"vcFillArrowEdgeColor"    , 1);
    NhlRLSetString (vc2_rlist,"vcFillArrowsOn"          , "True");
    NhlRLSetFloat  (vc2_rlist,"vcMinFracLengthF"        , 0.33);
    NhlRLSetFloat  (vc2_rlist,"vcMinMagnitudeF"         , 0.001);
    NhlRLSetString (vc2_rlist,"vcMonoFillArrowFillColor", "False");
    NhlRLSetString (vc2_rlist,"vcMonoLineArrowColor"    , "False");
    NhlRLSetFloat  (vc2_rlist,"vcRefLengthF"            , 0.045);
    NhlRLSetFloat  (vc2_rlist,"vcRefMagnitudeF"         , 20.0);
    NhlRLSetFloat  (vc2_rlist, "vcRefAnnoOrthogonalPosF", -0.1);
    NhlRLSetString (vc2_rlist,"pmLabelBarDisplayMode"   , "Always");
    NhlRLSetString (vc2_rlist,"pmLabelBarSide"          , "Bottom");
    NhlRLSetString (vc2_rlist,"lbOrientation"           , "Horizontal");
    NhlRLSetString (vc2_rlist,"lbPerimOn"               , "False");
    NhlRLSetInteger(vc2_rlist,"lbLabelFont"             , 25);

/*
 * Set some map resources.
 */
    NhlRLClear(mp2_rlist);
    NhlRLSetString (mp2_rlist,"mpProjection"           , "Mercator");
    NhlRLSetString (mp2_rlist,"mpLimitMode"            , "LatLon");
    NhlRLSetFloat  (mp2_rlist,"mpMaxLatF"              ,  65.0);
    NhlRLSetFloat  (mp2_rlist,"mpMaxLonF"              , -58.);
    NhlRLSetFloat  (mp2_rlist,"mpMinLatF"              ,  18.0);
    NhlRLSetFloat  (mp2_rlist,"mpMinLonF"              , -128.);
    NhlRLSetFloat  (mp2_rlist,"mpCenterLatF"           ,   40.0);
    NhlRLSetFloat  (mp2_rlist,"mpCenterLonF"           , -100.0);
    NhlRLSetString (mp2_rlist,"mpFillOn"               , "True");
    NhlRLSetInteger(mp2_rlist,"mpInlandWaterFillColor" , -1);
    NhlRLSetInteger(mp2_rlist,"mpLandFillColor"        , 16);
    NhlRLSetInteger(mp2_rlist,"mpOceanFillColor"       , -1);
    NhlRLSetInteger(mp2_rlist,"mpGridLineDashPattern"  , 2);
    NhlRLSetString (mp2_rlist,"mpGridMaskMode"         , "MaskNotOcean");
    NhlRLSetString (mp2_rlist,"mpPerimOn"              , "True");
    NhlRLSetString (mp2_rlist,"mpOutlineBoundarySets"  ,
                    "GeophysicalAndUSStates");
    
    special_res.gsnFrame = 0;
    vctrmap = ngl_vector_map_wrap(wks, U, V, type_U, type_V, nlat_UV, 
                                  nlon_UV, is_lat_coord_UV, lat_UV, 
                                  type_lat_UV, is_lon_coord_UV, lon_UV,
                                  type_lon_UV, is_missing_U, is_missing_V,
                                  FillValue_U, FillValue_V, vf_rlist, 
                                  vc2_rlist, mp2_rlist, &special_res);
    
  
    special_res.gsnFrame = 1;

/*
 * Draw a polyline before we draw the plot.
 */

    NhlRLClear(gs_rlist);
    NhlRLSetString (gs_rlist,"gsLineColor"     , "red");
    NhlRLSetFloat  (gs_rlist,"gsLineThicknessF", 2.5);
    ngl_polyline_ndc_wrap(wks, (void *)xline1, (void *)yline2, "float",
                          "float", 2, 0, 0, NULL, NULL, gs_rlist,
                          &special_pres);

/*
 * Set some text resources and draw a text string.
 */

    NhlRLClear (tx_rlist);
    NhlRLSetFloat  (tx_rlist,"txFontHeightF", 0.03);
    NhlRLSetInteger(tx_rlist,"txFont"       , 22);
    NhlRLSetString (tx_rlist,"txFuncCode"   , "~");

    xf = 0.5;
    yf = 0.16;
    text = ngl_text_ndc_wrap(wks,"ngl_text_ndc: I'm a big labelbar",
                             &xf,&yf,"float","float", tx_rlist,&special_pres);
    NhlFrame(wks);

  }

/*
 * ngl_streamline_map section
 */

  if(do_streamline_map) {

/*
 * Set some resources for streamline over map plot.
 */

    NhlRLClear(vf_rlist);
    NhlRLClear(vc2_rlist);
    NhlRLClear(mp2_rlist);

    NhlRLSetString (mp2_rlist,"mpProjection"           , "LambertConformal");
    NhlRLSetFloat  (mp2_rlist,"mpLambertParallel1F"    ,   0.001); 
    NhlRLSetFloat  (mp2_rlist,"mpLambertParallel2F"    ,  89.999); 
    NhlRLSetFloat  (mp2_rlist,"mpLambertMeridianF"     , -93.0);
    NhlRLSetString (mp2_rlist,"mpLimitMode"            , "LatLon");
    NhlRLSetFloat  (mp2_rlist,"mpMaxLatF"              ,  65.0);
    NhlRLSetFloat  (mp2_rlist,"mpMaxLonF"              , -58.);
    NhlRLSetFloat  (mp2_rlist,"mpMinLatF"              ,  18.0);
    NhlRLSetFloat  (mp2_rlist,"mpMinLonF"              , -128.);
    NhlRLSetString (mp2_rlist,"mpPerimOn"              , "True");

/*
 * Set some streamline resources. 
 */
    NhlRLClear(st_rlist);
    NhlRLSetString (st_rlist,"stLineColor"             , "green");
    NhlRLSetFloat  (st_rlist,"stLineThicknessF"        , 2.0);
    NhlRLSetString (st_rlist,"tiMainString"            , "Green Streams");
    NhlRLSetInteger(st_rlist,"tiMainFont"              , 25);

    special_res.gsnFrame = 0;

    strmlnmap = ngl_streamline_map_wrap(wks, U, V, type_U, type_V, nlat_UV, 
                                        nlon_UV, is_lat_coord_UV, lat_UV, 
                                        type_lat_UV, is_lon_coord_UV, lon_UV,
                                        type_lon_UV, is_missing_U, 
                                        is_missing_V, FillValue_U, FillValue_V,
                                        vf_rlist, st_rlist, mp2_rlist, 
                                        &special_res);

    special_res.gsnFrame = 1;
/*
 * Set some text resources and draw a text string.
 */
    NhlRLClear(tx_rlist);
    NhlRLSetFloat  (tx_rlist,"txFontHeightF"        , 0.02);
    NhlRLSetString (tx_rlist,"txFuncCode"           , "~");
    NhlRLSetString (tx_rlist,"txFont"               , "helvetica-bold");
    NhlRLSetString (tx_rlist,"txFontColor"          , "Blue");
    NhlRLSetString (tx_rlist,"txPerimOn"            , "True");
    NhlRLSetString (tx_rlist,"txBackgroundFillColor", "LightGray");
    
    ixf = -93;
    iyf =  65;
    text = ngl_text_wrap(wks,strmlnmap,"ngl_text: lat=65,lon=-93",
                         &ixf, &iyf, "integer", "integer", tx_rlist,
                         &special_pres);
/*
 * Draw a polygon, markers, and line.
 */

    NhlRLClear(gs_rlist);
    NhlRLSetString (gs_rlist,"gsFillColor", "LightGray");
    ngl_polygon_ndc_wrap(wks, (void *)xgon, (void *)ygon, "float", "float",
                         NG, 0, 0, NULL, NULL, gs_rlist, &special_pres);

    NhlRLClear(gs_rlist);
    NhlRLSetString (gs_rlist,"gsMarkerColor", "red");
    ngl_polymarker_ndc_wrap(wks, (void *)xgon, (void *)ygon, "float",
                            "float", NG, 0, 0, NULL, NULL, gs_rlist,
                            &special_pres);

    NhlRLClear(gs_rlist);
    NhlRLSetString (gs_rlist,"gsLineColor", "Blue");
    ngl_polyline_ndc_wrap(wks, (void *)xgon, (void *)ygon, "float", 
                          "float", NG, 0, 0, NULL, NULL, gs_rlist,
                          &special_pres);

    NhlFrame(wks);
  }

/*
 * ngl_vector_scalar_map section
 */

  if(do_vector_scalar_map) {

/*
 * Set some map resources.
 */

    NhlRLClear(mp2_rlist);
    NhlRLSetString (mp2_rlist,"mpProjection"           , "Mercator");
    NhlRLSetString (mp2_rlist,"mpLimitMode"            , "LatLon");
    NhlRLSetFloat  (mp2_rlist,"mpMaxLatF"              ,  60.0);
    NhlRLSetFloat  (mp2_rlist,"mpMaxLonF"              , -62.);
    NhlRLSetFloat  (mp2_rlist,"mpMinLatF"              ,  18.0);
    NhlRLSetFloat  (mp2_rlist,"mpMinLonF"              , -128.);
    NhlRLSetFloat  (mp2_rlist,"mpCenterLatF"           ,   40.0);
    NhlRLSetFloat  (mp2_rlist,"mpCenterLonF"           , -100.0);
    NhlRLSetString (mp2_rlist,"mpFillOn"               , "True");
    NhlRLSetInteger(mp2_rlist,"mpInlandWaterFillColor" , -1);
    NhlRLSetString (mp2_rlist,"mpLandFillColor"        , "LightGray");
    NhlRLSetInteger(mp2_rlist,"mpOceanFillColor"       , -1);
    NhlRLSetInteger(mp2_rlist,"mpGridLineDashPattern"  , 2);
    NhlRLSetString (mp2_rlist,"mpGridMaskMode"         , "MaskNotOcean");
    NhlRLSetString (mp2_rlist,"mpOutlineOn"            , "False");
    NhlRLSetString (mp2_rlist,"mpPerimOn"              , "True");

/*
 * Set some vector resources.
 */
    NhlRLClear(vf_rlist);
    NhlRLClear(vc2_rlist);

    NhlRLSetString (vc2_rlist,"pmLabelBarDisplayMode"  , "Always");
    NhlRLSetString (vc2_rlist,"pmLabelBarSide"         , "Bottom");
    NhlRLSetString (vc2_rlist,"lbOrientation"          , "Horizontal");
    NhlRLSetString (vc2_rlist,"lbPerimOn"              , "False");
    NhlRLSetInteger(vc2_rlist,"lbTitleFont"            , 25);
    NhlRLSetString (vc2_rlist,"lbTitleString"          , "TEMPERATURE (:S:o:N:F)");
    NhlRLSetFloat  (vc2_rlist,"tiMainFontHeightF"      , 0.03);
    NhlRLSetString (vc2_rlist,"tiMainString"           , ":F25:Wind velocity vectors");

    NhlRLSetInteger(vc2_rlist,"vcFillArrowEdgeColor"    , 1);
    NhlRLSetString (vc2_rlist,"vcFillArrowsOn"          , "True");
    NhlRLSetFloat  (vc2_rlist,"vcMinFracLengthF"        , 0.33);
    NhlRLSetFloat  (vc2_rlist,"vcMinMagnitudeF"         , 0.001);
    NhlRLSetString (vc2_rlist,"vcMonoFillArrowFillColor", "False");
    NhlRLSetString (vc2_rlist,"vcMonoLineArrowColor"    , "False");
    NhlRLSetFloat  (vc2_rlist,"vcRefLengthF"            , 0.045);
    NhlRLSetFloat  (vc2_rlist,"vcRefMagnitudeF"         , 20.0);
    NhlRLSetString (vc2_rlist,"vcGlyphStyle"            , "CurlyVector");

    special_res.gsnFrame = 0;
    special_res.gsnDraw  = 0;

    vctrmap = ngl_vector_scalar_map_wrap(wks, U, V, T2, type_U, type_V, 
                                         type_T2, nlat_UV, nlon_UV, 
                                         is_lat_coord_UV, lat_UV, type_lat_UV,
                                         is_lon_coord_UV, lon_UV, type_lon_UV,
                                         is_missing_U, is_missing_V, 
                                         is_missing_T2, FillValue_U, 
                                         FillValue_V, FillValue_T2, vf_rlist,
                                         sf2_rlist, vc2_rlist, mp2_rlist,
                                         &special_res);
    special_res.gsnFrame = 1;
    special_res.gsnDraw  = 1;

/*
 * Attach a polygon and outline it using a polyline. Note: it is possible
 * to outline a polygon by setting the polygon resource "gsEdgesOn" to 
 * True.  We're doing it using a polyline for test purposes.
 */
    NhlRLClear(gs_rlist);
    NhlRLSetInteger (gs_rlist,"gsFillIndex", 17);
    i1 = ngl_add_polygon_wrap(wks, vctrmap, (void *)longon, (void *)latgon,
                              "integer", "integer", 5, 0, 0, NULL, NULL, 
                              gs_rlist, &special_pres);
    NhlRLSetInteger (gs_rlist,"gsLineThicknessF", 2.0);
    i2 = ngl_add_polyline_wrap(wks, vctrmap, (void *)longon, (void *)latgon,
                               "integer", "integer", 5, 0, 0, NULL, NULL, 
                               gs_rlist, &special_pres);

/*
 * Mark the four corners of the polygon with polymarkers.
 */
    NhlRLSetInteger(gs_rlist,"gsMarkerIndex", 16);
    NhlRLSetFloat  (gs_rlist,"gsMarkerSizeF", 10.5);
    i3 = ngl_add_polymarker_wrap(wks, vctrmap, (void *)longon, 
                                 (void *)latgon, "integer", "integer", 4, 0,
                                 0, NULL, NULL, gs_rlist,&special_pres);

/*
 * Label two corners of the polygon with text.
 */
    ixf = -125;
    yf  =   60;
    NhlRLClear(tx_rlist);
    NhlRLSetFloat  (tx_rlist,"txFontHeightF", 0.02);
    NhlRLSetString (tx_rlist,"txFont"       , "helvetica");
    NhlRLSetString (tx_rlist,"txJust"       , "BottomRight");

    text1 = ngl_add_text_wrap(wks, vctrmap, "lat=  60:C:lon=-125", &ixf, 
                             &yf, "integer", "float", tx_rlist, am_rlist,
                             &special_pres);

    ixf = -65;
    yf  =  60;
    NhlRLSetString (tx_rlist,"txJust", "BottomLeft");
    text2 = ngl_add_text_wrap(wks,vctrmap,"lat= 60:C:lon=-65",
                             &ixf, &yf, "integer", "float", tx_rlist,
                             am_rlist, &special_pres);
/*
 * Once you've "attached" all the primitives and text you want, go
 * ahead and draw the map.  You will see all the primitives and text
 * drawn too, since they are attached to the map.
 */

    NhlDraw(vctrmap);

/*
 * Label the other two corners of the polygon with text.
 */
    
    NhlRLSetString (tx_rlist,"txJust", "TopRight");
    ixf = -125;
    yf  =   20;
    text = ngl_text_wrap(wks,vctrmap,"lat=  20:C:lon=-125",
                         &ixf, &yf, "integer", "float", tx_rlist,
                         &special_pres);
    ixf = -65;
    yf  =  20;
    NhlRLSetString (tx_rlist,"txJust", "TopLeft");
    text = ngl_text_wrap(wks,vctrmap,"lat= 20:C:lon=-65", &ixf, &yf,
                         "integer", "float", tx_rlist, &special_pres);

    NhlFrame(wks);

/* 
 * Change viewport coordinates to show how some text gets resized, and
 * some doesn't.
 */ 
    srlist = NhlRLCreate(NhlSETRL);
    NhlRLClear(srlist);
    NhlRLSetFloat(srlist,"vpXF",      0.3);
    NhlRLSetFloat(srlist,"vpYF",      0.7);
    NhlRLSetFloat(srlist,"vpWidthF",  0.4);
    NhlRLSetFloat(srlist,"vpHeightF", 0.4);

    NhlSetValues(vctrmap,srlist);

  }
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
