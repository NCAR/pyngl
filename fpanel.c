#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <netcdf.h>
#include "gsun.h"

main()
{
/*
 * Declare variables for the HLU routine calls.
 */

  int wks;
  nglPlotId text, *carray;
  int wk_rlist, sf_rlist, tx_rlist, cn_rlist, mp_rlist, lb_rlist;
  int cmap_len[2];
  float *xf, *yf, cmap[33][3];
  int max_plots, panel_dims[2];
  nglRes special_res, special_pres, special_tres;

/*
 * Declare variables for getting information from netCDF file.
 */

  int ncid_T2, id_T2, lonid_T2, latid_T2, timeid_T2;
  int nlat_T2, nlon_T2, ntime_T2, nlatlon_T2, ntimelatlon_T2;
  int ndims_T2, is_missing_T2;
  int attid, status;

/*
 * Declare variables to hold values, missing values, and types that
 * are read in from netCDF files.
 */
  void  *T2, *lat_T2, *lon_T2, *time_T2;
  void  *T2new, *FillValue_T2;
  int is_lat_coord_T2, is_lon_coord_T2, is_time_coord_T2;
  size_t i, j, *start, *count;
  char  filename_T2[256];
  const char *dir = _NGGetNCARGEnv("data");

/*
 * Open the netCDF files for contour and vector data.
 */

  sprintf(filename_T2, "%s/cdf/Tstorm.cdf", dir );
  nc_open(filename_T2,NC_NOWRITE,&ncid_T2);

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
 * Get temperature id.
 */

  nc_inq_varid(ncid_T2, "t",        &id_T2);
  nc_inq_varid(ncid_T2, "lat",      &latid_T2);
  nc_inq_varid(ncid_T2, "lon",      &lonid_T2);
  nc_inq_varid(ncid_T2, "timestep", &timeid_T2);

  is_missing_T2    = 1;
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
  lat_T2  = (float *)malloc(nlat_T2*sizeof(float));
  lon_T2  = (float *)malloc(nlon_T2*sizeof(float));
  time_T2 = (float *)malloc(ntime_T2*sizeof(float));
  FillValue_T2 = (float *)malloc(sizeof(float));

  if(T2 == NULL || lat_T2 == NULL || lon_T2 == NULL || time_T2 == NULL 
     || FillValue_T2 == NULL) {
    printf("Not enough memory for data variables.\n");
    exit(0);
  }

  nc_get_vara_float(ncid_T2, id_T2,     start, count, (float*)T2);
  nc_get_var_float (ncid_T2, latid_T2,                (float*)lat_T2);
  nc_get_var_float (ncid_T2, lonid_T2,                (float*)lon_T2);
  nc_get_var_float (ncid_T2, timeid_T2,               (float*)time_T2);

/*
 * Get missing values.
 */
  nc_get_att_float (ncid_T2, id_T2, "_FillValue", (float*)FillValue_T2); 

/*
 * Close the netCDF file.
 */

  ncclose(ncid_T2);


/*----------------------------------------------------------------------*
 *
 * Start graphics portion of code.
 *
 *----------------------------------------------------------------------*/
/*
 * Initialize special resources.
 */
  initialize_resources(&special_res,  nglPlot);
  initialize_resources(&special_pres, nglPlot);
  initialize_resources(&special_tres, nglPrimitive);
    
/*
 * Initialize color map for later.
 */
   cmap[ 0][0] = 1.00;   cmap[ 0][1] = 1.00;   cmap[ 0][2] = 1.00;
   cmap[ 1][0] = 0.00;   cmap[ 1][1] = 0.00;   cmap[ 1][2] = 0.00;
   cmap[ 2][0] = 1.00;   cmap[ 2][1] = .000;   cmap[ 2][2] = .000;
   cmap[ 3][0] = .950;   cmap[ 3][1] = .010;   cmap[ 3][2] = .000;
   cmap[ 4][0] = .870;   cmap[ 4][1] = .050;   cmap[ 4][2] = .000;
   cmap[ 5][0] = .800;   cmap[ 5][1] = .090;   cmap[ 5][2] = .000;
   cmap[ 6][0] = .700;   cmap[ 6][1] = .090;   cmap[ 6][2] = .000;
   cmap[ 7][0] = .700;   cmap[ 7][1] = .120;   cmap[ 7][2] = .000;
   cmap[ 8][0] = .700;   cmap[ 8][1] = .180;   cmap[ 8][2] = .000;
   cmap[ 9][0] = .700;   cmap[ 9][1] = .260;   cmap[ 9][2] = .000;
   cmap[10][0] = .700;   cmap[10][1] = .285;   cmap[10][2] = .000;
   cmap[11][0] = .680;   cmap[11][1] = .330;   cmap[11][2] = .000;
   cmap[12][0] = .570;   cmap[12][1] = .420;   cmap[12][2] = .000;
   cmap[13][0] = .560;   cmap[13][1] = .530;   cmap[13][2] = .000;
   cmap[14][0] = .550;   cmap[14][1] = .550;   cmap[14][2] = .000;
   cmap[15][0] = .130;   cmap[15][1] = .570;   cmap[15][2] = .000;
   cmap[16][0] = .060;   cmap[16][1] = .680;   cmap[16][2] = .000;
   cmap[17][0] = .000;   cmap[17][1] = .690;   cmap[17][2] = .000;
   cmap[18][0] = .000;   cmap[18][1] = .700;   cmap[18][2] = .100;
   cmap[19][0] = .000;   cmap[19][1] = .600;   cmap[19][2] = .300;
   cmap[20][0] = .000;   cmap[20][1] = .500;   cmap[20][2] = .500;
   cmap[21][0] = .000;   cmap[21][1] = .400;   cmap[21][2] = .700;
   cmap[22][0] = .000;   cmap[22][1] = .300;   cmap[22][2] = .700;
   cmap[23][0] = .000;   cmap[23][1] = .200;   cmap[23][2] = .700;
   cmap[24][0] = .000;   cmap[24][1] = .100;   cmap[24][2] = .700;
   cmap[25][0] = .000;   cmap[25][1] = .000;   cmap[25][2] = .700;
   cmap[26][0] = .100;   cmap[26][1] = .100;   cmap[26][2] = .700;
   cmap[27][0] = .200;   cmap[27][1] = .200;   cmap[27][2] = .700;
   cmap[28][0] = .300;   cmap[28][1] = .300;   cmap[28][2] = .700;
   cmap[29][0] = .420;   cmap[29][1] = .400;   cmap[29][2] = .700;
   cmap[30][0] = .560;   cmap[30][1] = .500;   cmap[30][2] = .700;
   cmap[31][0] = .610;   cmap[31][1] = .600;   cmap[31][2] = .700;
   cmap[32][0] = .700;   cmap[32][1] = .700;   cmap[32][2] = .700;

/*
 * Set up workstation resource list.
 */

  wk_rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(wk_rlist);
  cmap_len[0] = 33;
  cmap_len[1] = 3;
  NhlRLSetMDFloatArray(wk_rlist,"wkColorMap",&cmap[0][0],2,cmap_len);
  wks = ngl_open_wks_wrap("x11","panel", wk_rlist);

/*
 * Initialize and clear resource lists.
 */
  sf_rlist = NhlRLCreate(NhlSETRL);
  cn_rlist = NhlRLCreate(NhlSETRL);
  mp_rlist = NhlRLCreate(NhlSETRL);


  NhlRLClear(sf_rlist);
  NhlRLClear(cn_rlist);
  NhlRLClear(mp_rlist);

/*
 * Set some contour resources.
 */
  NhlRLSetString(cn_rlist,"cnInfoLabelOn", "False");
  NhlRLSetString(cn_rlist,"cnLineLabelsOn", "False");
  NhlRLSetString(cn_rlist,"cnLinesOn", "False");
  NhlRLSetString(cn_rlist,"cnFillOn", "True");

  NhlRLSetString(cn_rlist,"cnLevelSelectionMode", "ManualLevels");
  NhlRLSetFloat(cn_rlist,"cnMinLevelValF", 245);
  NhlRLSetFloat(cn_rlist,"cnMaxLevelValF", 302.5);
  NhlRLSetFloat(cn_rlist,"cnLevelSpacingF",   2.5);

  NhlRLSetString(mp_rlist,"mpLimitMode", "LatLon");
  NhlRLSetFloat(mp_rlist,"mpMinLatF", 20.);
  NhlRLSetFloat(mp_rlist,"mpMaxLatF", 60.);
  NhlRLSetFloat(mp_rlist,"mpMinLonF", -140.);
  NhlRLSetFloat(mp_rlist,"mpMaxLonF", -52.5);

  NhlRLSetString(mp_rlist,"mpPerimOn", "True");
  NhlRLSetString(mp_rlist,"mpGridAndLimbOn", "False");

  max_plots = 6;
  carray    = (nglPlotId*)malloc(max_plots*sizeof(nglPlotId));

  if(carray   == NULL) {
    printf("Not enough memory for plot ids.\n");
    exit(0);
  }

/*
 * Change some of the default special resources, since we don't want to
 * draw each individual plot, and we also don't need to waste time
 * maximizing its size. Also, turn off gsnSpreadColors.
 */
  special_res.nglMaximize = 0;
  special_res.nglFrame    = 0;
  special_res.nglDraw     = 0;
  special_res.nglSpreadColors = 0;

/*
 * Loop through the number of plots, and create a bunch of contour
 * plots. 
 */
  for(i = 0; i < max_plots; i++) {
    T2new = &((float*)T2)[nlatlon_T2*i];

    carray[i] = ngl_contour_map_wrap(wks, T2new, "float", nlat_T2, nlon_T2, 
                                     is_lat_coord_T2, lat_T2, "float",
                                     is_lon_coord_T2, lon_T2, "float", 
                                     is_missing_T2, FillValue_T2, 
                                     sf_rlist, cn_rlist, mp_rlist,
                                     &special_res);

  }

/*
 * Panel stuff.
 */

  lb_rlist = NhlRLCreate(NhlSETRL);
  tx_rlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(lb_rlist);
  NhlRLClear(tx_rlist);

  special_pres.nglPanelTop      = 0.934;
  special_pres.nglPanelLabelBar = 1;
  special_pres.nglFrame         = 0;
  special_pres.nglPanelLabelBarWidthF  = 0.70;
  special_pres.nglPanelLabelBarHeightF = 0.15;

  panel_dims[0] = 3;
  panel_dims[1] = 2;

  NhlRLSetString(lb_rlist,"lbLabelFont","helvetica-bold");
  NhlRLSetString(lb_rlist,"lbLabelAutoStride","True");
  NhlRLSetFloat(lb_rlist,"lbLabelFontHeightF",0.015);

  ngl_panel_wrap(wks, carray, max_plots, panel_dims, 2, lb_rlist,
                 &special_pres);

  NhlRLSetFloat(tx_rlist,"txFontHeightF" , 0.025);

  xf = (float*)malloc(sizeof(float));
  yf = (float*)malloc(sizeof(float));
  *xf = 0.5;
  *yf = 0.97;
  special_tres.nglFrame = 0;
  text = ngl_text_ndc_wrap(wks,":F26:Temperature (K) at every six hours",
                           (void*)xf,(void*)yf,"float","float",tx_rlist,
                           &special_tres);

  NhlRLSetFloat(tx_rlist,"txFontHeightF" , 0.02);

  *yf = 0.935;
  text = ngl_text_ndc_wrap(wks,":F26:January 1996",
                           (void*)xf,(void*)yf,"float","float",tx_rlist,
                           &special_tres);

  NhlFrame(wks);  

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
