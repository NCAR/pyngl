#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <netcdf.h>
#include "gsun.h"

#define NCOLORS 17
#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))
#define NG  5

void create_title(char *title, nglPlotId *parray, int nplots,
                  int *panel_dims, int ndims, const char *extra_string,
                  int row_spec)
{
  int i, first_time;
  char title2[100];

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
    if(parray[i].nbase == 0) {
      if(first_time) {
        sprintf(title2,":C:plots %d", i);
        strcat(title,title2);
        first_time = 0;
      }
      else {
        sprintf(title2,", %d", i);
        strcat(title,title2);
      }
      parray[i].nbase = 1;
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
  nglPlotId text, *carray;
  nglPlotId *varray, *varray_f, *varray_l, *varray_c, *varray_s;
  nglPlotId *varray_f_map, *varray_c_map, *varray_s_map;
  int wk_rlist, sf_rlist, vf_rlist, tx_rlist, cn_rlist, mp_rlist;
  int vc_rlist, vc_f_rlist, vc_l_rlist, vc_c_rlist, vc_s_rlist;
  int srlist, lb_rlist, cmap_len[2];
  float *xf, *yf, cmap[NCOLORS][3];
  int max_plots, nplots, panel_dims[2], first_time, *row_spec;
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
  int is_lat_coord_T2, is_lon_coord_T2, is_time_coord_T2;
  size_t i, j, *start, *count;
  char  filename_U[256], filename_V[256], filename_T2[256], title[100];
  const char *dir = _NGGetNCARGEnv("data");

  extern void create_title(char *, nglPlotId *, int, int *, int,
                           const char *, int);

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

/*
 * Retrieve coordinate var lengths.
 */

  nc_inq_dimlen(ncid_T2,latid_T2,(size_t*)&nlat_T2);  
  nc_inq_dimlen(ncid_T2,lonid_T2,(size_t*)&nlon_T2);
  nc_inq_dimlen(ncid_T2,timeid_T2,(size_t*)&ntime_T2);

  nlatlon_T2     = nlat_T2 * nlon_T2;
  ntimelatlon_T2 = ntime_T2 * nlatlon_T2;

/*
 * Get temperature, u, v, lat, and lon ids.
 */

  nc_inq_varid(ncid_T2, "t",        &id_T2);
  nc_inq_varid(ncid_U,  "u",        &id_U);
  nc_inq_varid(ncid_V,  "v",        &id_V);
  nc_inq_varid(ncid_T2, "lat",      &latid_T2);
  nc_inq_varid(ncid_T2, "lon",      &lonid_T2);
  nc_inq_varid(ncid_T2, "timestep", &timeid_T2);

  is_missing_T2    = 1;
  is_missing_U     = 1;
  is_missing_V     = 1;
  is_lat_coord_T2  = 1;
  is_lon_coord_T2  = 1;
  is_time_coord_T2 = 1;

/*
 * Read in all the data: T2, U, V, FillValues, and coordinate arrays.
 */

  nc_inq_varndims (ncid_T2, id_T2, &ndims_T2);

  start = (size_t *)malloc(ndims_T2*sizeof(size_t));
  count = (size_t *)malloc(ndims_T2*sizeof(size_t));
  for(i = 0; i < ndims_T2; i++)   start[i] = 0;
  for(i = 0; i < ndims_T2-3; i++) count[i] = 1;

  count[ndims_T2-3] = ntime_T2;
  count[ndims_T2-2] = nlat_T2;
  count[ndims_T2-1] = nlon_T2;

  T2      = (float *)malloc(ntimelatlon_T2*sizeof(float));
  U       = (float *)malloc(ntimelatlon_T2*sizeof(float));
  V       = (float *)malloc(ntimelatlon_T2*sizeof(float));
  lat_T2  = (float *)malloc(nlat_T2*sizeof(float));
  lon_T2  = (float *)malloc(nlon_T2*sizeof(float));
  time_T2 = (float *)malloc(ntime_T2*sizeof(float));
  FillValue_T2 = (float *)malloc(sizeof(float));
  FillValue_U  = (float *)malloc(sizeof(float));
  FillValue_V  = (float *)malloc(sizeof(float));

  if(T2 == NULL || U == NULL || V == NULL || lat_T2 == NULL || 
     lon_T2 == NULL || time_T2 == NULL || FillValue_T2 == NULL || 
     FillValue_U == NULL || FillValue_V == NULL) {
    printf("Not enough memory for data variables.\n");
    exit(0);
  }

  nc_get_vara_float(ncid_T2, id_T2,     start, count, (float*)T2);
  nc_get_vara_float(ncid_U,  id_U,      start, count, (float*)U);
  nc_get_vara_float(ncid_V,  id_V,      start, count, (float*)V);
  nc_get_var_float (ncid_T2, latid_T2,                (float*)lat_T2);
  nc_get_var_float (ncid_T2, lonid_T2,                (float*)lon_T2);
  nc_get_var_float (ncid_T2, timeid_T2,               (float*)time_T2);

/*
 * Get missing values.
 */
  nc_get_att_float (ncid_T2, id_T2, "_FillValue", (float*)FillValue_T2); 
  nc_get_att_float (ncid_U,  id_U,  "_FillValue", (float*)FillValue_U); 
  nc_get_att_float (ncid_V,  id_V,  "_FillValue", (float*)FillValue_V); 

/*
 * Convert T from K to F.
 */
  for(i = 0; i < ntimelatlon_T2; i++) {
    if(((float*)T2)[i] != ((float*)FillValue_T2)[0]) {
      ((float*)T2)[i] = (((float*)T2)[i]-273.15)*(9./5.) + 32.;
    }
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
 * Initialize special resources.
 */
  initialize_resources(&special_res,  nglPlot);
  initialize_resources(&special_tres, nglPrimitive);
  initialize_resources(&special_pres, nglPlot);
    
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
  wks = ngl_open_wks_wrap("ps","panel", wk_rlist);

/*
 * Initialize and clear resource lists.
 */

  srlist   = NhlRLCreate(NhlSETRL);
  sf_rlist = NhlRLCreate(NhlSETRL);
  vf_rlist = NhlRLCreate(NhlSETRL);
  tx_rlist = NhlRLCreate(NhlSETRL);
  cn_rlist = NhlRLCreate(NhlSETRL);
  mp_rlist = NhlRLCreate(NhlSETRL);
  lb_rlist = NhlRLCreate(NhlSETRL);
  vc_rlist        = NhlRLCreate(NhlSETRL);
  vc_l_rlist   = NhlRLCreate(NhlSETRL);
  vc_f_rlist   = NhlRLCreate(NhlSETRL);
  vc_c_rlist  = NhlRLCreate(NhlSETRL);
  vc_s_rlist = NhlRLCreate(NhlSETRL);

  NhlRLClear(srlist);
  NhlRLClear(sf_rlist);
  NhlRLClear(vf_rlist);
  NhlRLClear(tx_rlist);
  NhlRLClear(cn_rlist);
  NhlRLClear(mp_rlist);
  NhlRLClear(lb_rlist);
  NhlRLClear(vc_rlist);
  NhlRLClear(vc_f_rlist);
  NhlRLClear(vc_l_rlist);
  NhlRLClear(vc_c_rlist);
  NhlRLClear(vc_s_rlist);

/*
 * Set some colormap resources.
 */
  if(!special_res.nglSpreadColors) {
    cmap_len[0] = NCOLORS;
    cmap_len[1] = 3;
    NhlRLSetMDFloatArray(srlist,NhlNwkColorMap,&cmap[0][0],2,cmap_len);
  }
  else {
    NhlRLSetString(srlist,"wkColorMap","rainbow+gray");
  }
  (void)NhlSetValues(wks, srlist);

/*
 * Set some contour resources.
 */
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
  NhlRLSetFloat  (vc_rlist, "vcRefLengthF",         0.045);
  NhlRLSetFloat  (vc_rlist, "vcRefMagnitudeF",      20.0);
  NhlRLSetFloat  (vc_rlist, "vcMinMagnitudeF",      0.001);
  NhlRLSetFloat  (vc_rlist, "vcMinFracLengthF",      0.33);

  NhlRLSetString (vc_rlist, "vcLevelSelectionMode",  "ManualLevels");
  NhlRLSetFloat  (vc_rlist, "vcLevelSpacingF", 2.0);
  NhlRLSetFloat  (vc_rlist, "vcMinLevelValF",  0.0);
  NhlRLSetFloat  (vc_rlist, "vcMaxLevelValF",  20.0);

  NhlRLSetString (vc_rlist, "vcMonoLineArrowColor", "True");
  NhlRLSetString (vc_rlist, "tiMainString", "Plain Vectors");

  NhlRLSetFloat  (vc_f_rlist, "vcRefLengthF",         0.045);
  NhlRLSetFloat  (vc_f_rlist, "vcRefMagnitudeF",      20.0);
  NhlRLSetFloat  (vc_f_rlist, "vcMinMagnitudeF",      0.001);
  NhlRLSetFloat  (vc_f_rlist, "vcMinFracLengthF",      0.33);

  NhlRLSetString (vc_f_rlist, "vcLevelSelectionMode",  "ManualLevels");
  NhlRLSetFloat  (vc_f_rlist, "vcLevelSpacingF", 2.0);
  NhlRLSetFloat  (vc_f_rlist, "vcMinLevelValF",  0.0);
  NhlRLSetFloat  (vc_f_rlist, "vcMaxLevelValF",  20.0);

  NhlRLSetString (vc_f_rlist, "vcFillArrowsOn", "True");
  NhlRLSetString (vc_f_rlist, "vcMonoFillArrowFillColor", "False");
  NhlRLSetString (vc_f_rlist, "tiMainString", "Filled Vectors");

  NhlRLSetFloat  (vc_l_rlist, "vcRefLengthF",         0.045);
  NhlRLSetFloat  (vc_l_rlist, "vcRefMagnitudeF",      20.0);
  NhlRLSetFloat  (vc_l_rlist, "vcMinMagnitudeF",      0.001);
  NhlRLSetFloat  (vc_l_rlist, "vcMinFracLengthF",      0.33);
  NhlRLSetString (vc_l_rlist, "pmLabelBarDisplayMode", "Never"); 

  NhlRLSetString (vc_l_rlist, "vcLevelSelectionMode",  "ManualLevels");
  NhlRLSetFloat  (vc_l_rlist, "vcLevelSpacingF", 2.0);
  NhlRLSetFloat  (vc_l_rlist, "vcMinLevelValF",  0.0);
  NhlRLSetFloat  (vc_l_rlist, "vcMaxLevelValF",  20.0);

  NhlRLSetString (vc_l_rlist, "vcFillArrowsOn",       "False");
  NhlRLSetString (vc_l_rlist, "vcMonoLineArrowColor", "False");
  NhlRLSetString (vc_l_rlist, "tiMainString", "Line Vectors");

  NhlRLSetFloat  (vc_c_rlist, "vcRefLengthF",         0.045);
  NhlRLSetFloat  (vc_c_rlist, "vcRefMagnitudeF",      20.0);
  NhlRLSetFloat  (vc_c_rlist, "vcMinMagnitudeF",      0.001);
  NhlRLSetFloat  (vc_c_rlist, "vcMinFracLengthF",      0.33);
  NhlRLSetString (vc_c_rlist, "pmLabelBarDisplayMode", "Never"); 

  NhlRLSetString (vc_c_rlist, "vcLevelSelectionMode",  "ManualLevels");
  NhlRLSetFloat  (vc_c_rlist, "vcLevelSpacingF", 2.0);
  NhlRLSetFloat  (vc_c_rlist, "vcMinLevelValF",  0.0);
  NhlRLSetFloat  (vc_c_rlist, "vcMaxLevelValF",  20.0);

  NhlRLSetInteger(vc_c_rlist, "vcGlyphStyle", NhlCURLYVECTOR);
  NhlRLSetString (vc_c_rlist, "vcMonoLineArrowColor", "False");
  NhlRLSetString (vc_c_rlist, "tiMainString", "Curly Vectors");

  NhlRLSetFloat  (vc_s_rlist, "vcRefLengthF",         0.045);
  NhlRLSetFloat  (vc_s_rlist, "vcRefMagnitudeF",      20.0);
  NhlRLSetFloat  (vc_s_rlist, "vcMinMagnitudeF",      0.001);
  NhlRLSetFloat  (vc_s_rlist, "vcMinFracLengthF",      0.33);
  NhlRLSetString (vc_s_rlist, "pmLabelBarDisplayMode", "Never"); 
  NhlRLSetFloat  (vc_s_rlist, "vcRefAnnoOrthogonalPosF", -0.15);
  NhlRLSetString (vc_s_rlist, "vcLevelSelectionMode",  "ManualLevels");
  NhlRLSetFloat  (vc_s_rlist, "vcLevelSpacingF", 10.0);
  NhlRLSetFloat  (vc_s_rlist, "vcMinLevelValF",   0.0);
  NhlRLSetFloat  (vc_s_rlist, "vcMaxLevelValF",  80.0);

  NhlRLSetInteger(vc_s_rlist, "vcGlyphStyle", NhlCURLYVECTOR);
  NhlRLSetString (vc_s_rlist, "vcMonoLineArrowColor", "False");
  NhlRLSetString (vc_s_rlist, "tiMainString", "Scalar Vectors");

  NhlRLSetString (mp_rlist,"mpProjection"           , "Mercator");
  NhlRLSetString (mp_rlist,"mpLimitMode"            , "LatLon");
  NhlRLSetFloat  (mp_rlist,"mpMaxLatF"              ,  60.0);
  NhlRLSetFloat  (mp_rlist,"mpMaxLonF"              , -62.);
  NhlRLSetFloat  (mp_rlist,"mpMinLatF"              ,  18.0);
  NhlRLSetFloat  (mp_rlist,"mpMinLonF"              , -128.);
  NhlRLSetFloat  (mp_rlist,"mpCenterLatF"           ,   40.0);
  NhlRLSetFloat  (mp_rlist,"mpCenterLonF"           , -100.0);
  NhlRLSetString (mp_rlist,"mpFillOn"               , "True");
  NhlRLSetInteger(mp_rlist,"mpInlandWaterFillColor" , -1);
  NhlRLSetString (mp_rlist,"mpLandFillColor"        , "LightGray");
  NhlRLSetInteger(mp_rlist,"mpOceanFillColor"       , -1);
  NhlRLSetInteger(mp_rlist,"mpGridLineDashPattern"  , 2);
  NhlRLSetString (mp_rlist,"mpGridMaskMode"         , "MaskNotOcean");
  NhlRLSetString (mp_rlist,"mpOutlineOn"            , "False");
  NhlRLSetString (mp_rlist,"mpPerimOn"              , "True");

/*
 * Loop across time dimension and create a contour and vector plot for
 * each one. Then, we'll create several panel plots. We could potentially
 * create ntime plots, but this is too time-consuming, so we are just
 * going to create max_plots of them.
 */
  max_plots     = 12;
  
  carray       = (nglPlotId*)malloc(max_plots*sizeof(nglPlotId));
  varray       = (nglPlotId*)malloc(max_plots*sizeof(nglPlotId));
  varray_f     = (nglPlotId*)malloc(max_plots*sizeof(nglPlotId));
  varray_l     = (nglPlotId*)malloc(max_plots*sizeof(nglPlotId));
  varray_c     = (nglPlotId*)malloc(max_plots*sizeof(nglPlotId));
  varray_s     = (nglPlotId*)malloc(max_plots*sizeof(nglPlotId));
  varray_s_map = (nglPlotId*)malloc(max_plots*sizeof(nglPlotId));

  if( carray   == NULL || varray == NULL   || varray_f == NULL || 
      varray_l == NULL || varray_c == NULL || varray_s == NULL ||
     varray_s_map == NULL) {
    printf("Not enough memory for plot ids.\n");
    exit(0);
  }

/*
 * Change some of the default special resources, since we don't want to
 * draw each individual plot, and we also don't need to waste time
 * maximizing its size.
 */
  special_res.nglMaximize = 0;
  special_res.nglFrame    = 0;
  special_res.nglDraw     = 0;

/*
 * Loop through the number of plots, and create a bunch of different
 * kinds.
 */
  for(i = 0; i < max_plots; i++) {
    T2new = &((float*)T2)[nlatlon_T2*i];
    Unew  = &((float*)U)[nlatlon_T2*i];
    Vnew  = &((float*)V)[nlatlon_T2*i];

    carray[i] = ngl_contour_wrap(wks, T2new, "float", nlat_T2, nlon_T2, 
                                 is_lat_coord_T2, lat_T2, "float",
                                 is_lon_coord_T2, lon_T2, "float", 
                                 is_missing_T2, FillValue_T2, 
                                 sf_rlist, cn_rlist, &special_res);

    varray[i] = ngl_vector_wrap(wks, Unew, Vnew, "float", "float",
                                nlat_T2, nlon_T2, is_lat_coord_T2, lat_T2,
                                "float", is_lon_coord_T2, lon_T2, "float",
                                is_missing_U, is_missing_V, FillValue_U,
                                FillValue_V, vf_rlist, vc_rlist, 
                                &special_res);

    varray_f[i] = ngl_vector_wrap(wks, Unew, Vnew, "float", "float",
                                  nlat_T2, nlon_T2, is_lat_coord_T2,
                                  lat_T2, "float", is_lon_coord_T2,
                                  lon_T2, "float", is_missing_U,
                                  is_missing_V, FillValue_U, FillValue_V, 
                                  vf_rlist, vc_f_rlist, &special_res);
                             
    varray_l[i] = ngl_vector_wrap(wks, Unew, Vnew, "float", "float",
                                  nlat_T2, nlon_T2, is_lat_coord_T2,
                                  lat_T2, "float", is_lon_coord_T2,
                                  lon_T2, "float", is_missing_U,
                                  is_missing_V, FillValue_U, FillValue_V,
                                  vf_rlist, vc_l_rlist, &special_res);
                                            

    varray_c[i] = ngl_vector_wrap(wks, Unew, Vnew, "float", "float",
                                  nlat_T2, nlon_T2, is_lat_coord_T2,
                                  lat_T2, "float", is_lon_coord_T2,
                                  lon_T2, "float", is_missing_U,
                                  is_missing_V, FillValue_U, FillValue_V,
                                  vf_rlist, vc_c_rlist, &special_res);
                                            
    varray_s[i] = ngl_vector_scalar_wrap(wks, Unew, Vnew, T2new,
                                         "float", "float", "float",
                                         nlat_T2, nlon_T2, is_lat_coord_T2,
                                         lat_T2, "float", is_lon_coord_T2,
                                         lon_T2, "float", is_missing_U,
                                         is_missing_V, is_missing_T2, 
                                         FillValue_U, FillValue_V,
                                         FillValue_T2, vf_rlist, sf_rlist,
                                         vc_s_rlist, &special_res);

    varray_s_map[i] = ngl_vector_scalar_map_wrap(wks, Unew, Vnew, T2new,
                                                 "float", "float", "float",
                                                 nlat_T2, nlon_T2,
                                                 is_lat_coord_T2, lat_T2,
                                                 "float", is_lon_coord_T2,
                                                 lon_T2, "float",
                                                 is_missing_U, is_missing_V,
                                                 is_missing_T2, FillValue_U,
                                                 FillValue_V, FillValue_T2,
                                                 vf_rlist, sf_rlist,
                                                 vc_s_rlist, mp_rlist,
                                                 &special_res);
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
 * Begin panel plots. Set nglPanelLabelBar to True (default is False).
 */

  nplots = 6;

  special_pres.nglPanelLabelBar = 1;
  special_pres.nglPanelRowSpec  = 0;
  panel_dims[0] = 3;
  panel_dims[1] = 2;

  special_pres.nglPanelLabelBar = 1;
  ngl_panel_wrap(wks, varray, nplots, panel_dims, 2, lb_rlist,
                 &special_pres);

  special_pres.nglPanelLabelBar = 1;
  ngl_panel_wrap(wks, varray_l, nplots, panel_dims, 2, lb_rlist,
                 &special_pres);

  ngl_panel_wrap(wks, varray_f, nplots, panel_dims, 2, lb_rlist,
                 &special_pres);
  ngl_panel_wrap(wks, varray_c, nplots, panel_dims, 2, lb_rlist,
                 &special_pres);
  ngl_panel_wrap(wks, varray_s, nplots, panel_dims, 2, lb_rlist,
                 &special_pres);
  ngl_panel_wrap(wks, varray_s_map, nplots, panel_dims, 2, lb_rlist,
                 &special_pres);

  special_pres.nglPanelLabelBarOrientation    = NhlVERTICAL;
  ngl_panel_wrap(wks, varray_f, nplots, panel_dims, 2, lb_rlist, &special_pres);
  ngl_panel_wrap(wks, varray_l, nplots, panel_dims, 2, lb_rlist, &special_pres);

  special_pres.nglPanelLabelBarOrthogonalPosF = 0.02;
  ngl_panel_wrap(wks, varray_f, nplots, panel_dims, 2, lb_rlist, &special_pres);

  NhlRLSetInteger (lb_rlist,"lbLabelStride", 2);
  special_pres.nglPanelLabelBarWidthF = -999.;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);

  NhlRLSetInteger (lb_rlist,"lbLabelStride", 1);
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
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelLabelBarOrientation = NhlVERTICAL;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  special_pres.nglPanelCenter = 1;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[5].nbase = 0;
  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[5].nbase = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[1].nbase = 0;
  carray[2].nbase = 0;
  special_pres.nglPanelCenter = 1;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[1].nbase = 0;
  carray[2].nbase = 0;
  special_pres.nglPanelCenter = 0;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[3].nbase = 0;
  carray[4].nbase = 0;
  special_pres.nglPanelCenter = 1;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[3].nbase = 0;
  carray[4].nbase = 0;
  special_pres.nglPanelCenter = 0;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[1].nbase = 0;
  carray[2].nbase = 0;
  carray[3].nbase = 0;
  carray[4].nbase = 0;
  special_pres.nglPanelCenter = 1;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is True", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[1].nbase = 0;
  carray[2].nbase = 0;
  carray[3].nbase = 0;
  carray[4].nbase = 0;
  special_pres.nglPanelCenter = 0;

  ngl_panel_wrap(wks, carray, nplots, row_spec, 3, lb_rlist, &special_pres);
  create_title(title, carray, nplots, row_spec, 3, "PanelCenter is False", 1);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Start with a single row of plots, then make the first plot and the last
 * plot missing.
 */
  special_pres.nglPanelRowSpec                = 0;
  special_pres.nglPanelLabelBarOrthogonalPosF = -999.;

  panel_dims[0] = 1;
  panel_dims[1] = 3;
  nplots = panel_dims[0] * panel_dims[1];

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[2].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
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
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[2].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
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
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[1].nbase = 0;
  carray[2].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[3].nbase = 0;
  carray[4].nbase = 0;
  carray[5].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[3].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[2].nbase = 0;
  carray[5].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
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
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[1].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[2].nbase = 0;
  carray[3].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[4].nbase = 0;
  carray[5].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[2].nbase = 0;
  carray[4].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[1].nbase = 0;
  carray[3].nbase = 0;
  carray[5].nbase = 0;

  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,"",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * 3 rows x 2 columns of plots, but only pass in 4 plots, then 2, then 1.
 */
  panel_dims[0] = 3;          /* rows    */
  panel_dims[1] = 2;          /* columns */
  nplots = 4;
  ngl_panel_wrap(wks, carray, 4, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "4 plots passed in",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[1].nbase = 0;
  ngl_panel_wrap(wks, carray, 4, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "4 plots passed in",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  nplots = 2;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "2 plots passed in",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  nplots = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
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
  create_title(title, carray, nplots, panel_dims,2,
                       "8 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "8 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);
/*
 * Same thing as above, only with some missing plots thrown in.
 */
  carray[0].nbase = 0;
  carray[1].nbase = 0;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "8 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[1].nbase = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "8 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Now do 5 plots.
 */
  nplots = 5;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "5 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "5 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Same thing as above, only with some missing plots thrown in.
 */
  carray[3].nbase = 0;
  carray[4].nbase = 0;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "5 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[3].nbase = 0;
  carray[4].nbase = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "5 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Now 3.
 */
  nplots = 3;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "3 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "3 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * Same thing as above, only with some missing plots thrown in.
 */ 
  carray[0].nbase = 0;
  carray[1].nbase = 0;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "3 plots passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  carray[0].nbase = 0;
  carray[1].nbase = 0;
  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "3 plots passed in:C:center is False",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

/*
 * One plot
 */
  nplots = 1;

  special_pres.nglPanelCenter = 1;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
                       "1 plot passed in:C:center is True",0);
  text = ngl_text_ndc_wrap(wks,title,(void*)xf,(void*)yf,"float","float",
                           tx_rlist,&special_tres);

  special_pres.nglPanelCenter = 0;
  ngl_panel_wrap(wks, carray, nplots, panel_dims, 2, lb_rlist, &special_pres);
  create_title(title, carray, nplots, panel_dims,2,
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
  free(U);
  free(V);
  free(lat_T2);
  free(lon_T2);
  exit(0);

}
