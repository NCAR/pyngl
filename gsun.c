#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gsun.h"

/*
 * Define some global strings.
 */
const char *polylinestr = "polyline";
  

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
void compute_ps_device_coords(int wks, int plot, gsnRes *special_res)
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
  is_debug     = special_res->gsnDebug;

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

void maximize_plot(int wks, int plot, gsnRes *special_res)
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

  if(special_res->gsnDebug) {
    printf("vpXF      = %g\n", new_vpx);
    printf("vpYF      = %g\n", new_vpy);
    printf("vpWidthF  = %g\n", new_vpw);
    printf("vpHeightF = %g\n", new_vph);
  }
  if(!strcmp(NhlClassName(wks),"psWorkstationClass") ||
     !strcmp(NhlClassName(wks),"pdfWorkstationClass")) {
/*
 * Compute and set device coordinates that will make plot fill the 
 * whole page.
 */
    compute_ps_device_coords(wks, plot, special_res);
  }
}

/*
 * This function "scales" the tickmarks and labels on the axes
 * so they are the same size/length.
 */

void scale_plot(int plot)
{
  int srlist, grlist;
  float xfont, yfont, xbfont, ylfont;
  float xlength, ylength, xmlength, ymlength;
  float major_length, minor_length;

  srlist = NhlRLCreate(NhlSETRL);
  grlist = NhlRLCreate(NhlGETRL);

  NhlRLClear(grlist);
  NhlRLGetFloat(grlist,"tiXAxisFontHeightF",    &xfont);
  NhlRLGetFloat(grlist,"tiYAxisFontHeightF",    &yfont);
  NhlRLGetFloat(grlist,"tmXBLabelFontHeightF",  &xbfont);
  NhlRLGetFloat(grlist,"tmXBMajorLengthF",      &xlength);
  NhlRLGetFloat(grlist,"tmXBMinorLengthF",      &xmlength);
  NhlRLGetFloat(grlist,"tmYLLabelFontHeightF",  &ylfont);
  NhlRLGetFloat(grlist,"tmYLMajorLengthF",      &ylength);
  NhlRLGetFloat(grlist,"tmYLMinorLengthF",      &ymlength);
  NhlGetValues(plot,grlist);

  if(xlength != 0. && ylength != 0.) {
    major_length = (ylength+xlength)/2.;
    xlength = major_length;
    ylength = major_length;
  }

  if(xmlength != 0. && ymlength != 0.) {
    minor_length = (ymlength+xmlength)/2.;
    xmlength = minor_length;
    ymlength = minor_length;
  }

/*
 * Reset all these resources, making the values the same for
 * both axes.
 */

  NhlRLClear(srlist);
  NhlRLSetFloat(srlist, "tiXAxisFontHeightF",   (xfont+yfont)/2.);
  NhlRLSetFloat(srlist, "tiYAxisFontHeightF",   (xfont+yfont)/2.);
  NhlRLSetFloat(srlist, "tmXBLabelFontHeightF", (xbfont+ylfont)/2.);
  NhlRLSetFloat(srlist, "tmXBMajorLengthF",     xlength);
  NhlRLSetFloat(srlist, "tmXBMinorLengthF",     xmlength);
  NhlRLSetFloat(srlist, "tmYLLabelFontHeightF", (xbfont+ylfont)/2.);
  NhlRLSetFloat(srlist, "tmYLMajorLengthF",     ylength);
  NhlRLSetFloat(srlist, "tmYLMinorLengthF",     ymlength);
  NhlSetValues(plot,srlist);

  return;
}

/*
 * This function coerces a void array to a float array (for routines
 * like NhlDataPolygon that expect floats).
 */
float *coerce_to_float(void *x, const char *type_x, int len)
{
  int i;
  float *xf;

/*
 * If it's already float, just return a pointer to it.
 */
  if(!strcmp(type_x,"float")) {
    xf = (float*)x;
  }

/*
 * Otherwise, it must be a double or integer.
 */
  else {
/*
 * Allocate space for float array.
 */
    xf = (float *)malloc(len*sizeof(float));
    if(xf == NULL) {
      NhlPError(NhlWARNING,NhlEUNKNOWN,"Not enough memory to coerce input array to float");
      return(NULL);
    }
/*
 * Do the conversion the old-fashioned way.
 */
    if(!strcmp(type_x,"double")) {
      for(i = 0; i < len; i++ ) {
        xf[i] = (float)((double*)x)[i];
      }
    }
    else if(!strcmp(type_x,"integer")) {
      for(i = 0; i < len; i++ ) {
        xf[i] = (float)((int*)x)[i];
      }
    }
    else {
      NhlPError(NhlWARNING,NhlEUNKNOWN,"Unrecognized type: input array must be integer, float, or double");
      return(NULL);
    }
  }
  return(xf);
}


/*
 * This function returns the indices of all the non-missing pairs of 
 * points in two float arrays.
 */
int *get_non_missing_pairs(float *xf, float *yf, int is_missing_x,
                           int is_missing_y, float *xmsg, float *ymsg,
                           int len, int *nlines)
{
  int i, *indices, *new_indices, ibeg, iend, is_missing_any;

  indices = calloc(len*2,sizeof(int));

  *nlines = 0;
  ibeg = -1;
  for(i = 0; i < len; i++ ) {
    if((!is_missing_x || (is_missing_x && xf[i] != *xmsg)) && 
       (!is_missing_y || (is_missing_y && yf[i] != *ymsg))) {
/*
 * ibeg < 0 ==> on the first point of the line
 */
      if(ibeg < 0) {
        ibeg = i;
        iend = i;
      }
      else {
        iend = i;
      }
      is_missing_any = 0;
    }
    else {
      is_missing_any = 1;
    }
    if(ibeg >= 0 && (is_missing_any || iend == (len-1))) {
      indices[*nlines*2]   = ibeg;
      indices[*nlines*2+1] = iend;
/*
 * Reinitialize
 */
      ibeg = -1;
      (*nlines)++;
    }
  }
  new_indices = malloc(*nlines*2*sizeof(int));
  memcpy((void *)new_indices,(const void *)indices,*nlines*2*sizeof(int));
  free(indices);
  return(new_indices);
}


/*
 * This function removes all the missing values from a float array.
 */
void collapse_nomsg(float *xf, float *xmsg, int len, int *newlen)
{
  int i;

  *newlen = 0;
  for(i = 0; i < len; i++ ) {
    if(xf[i] != *xmsg) {
      xf[*newlen] = xf[i];
      *newlen++;
    }
  }
}


/*
 * This function removes all the missing values from a pair of
 * float arrays. If they are missing in one or both arrays in the same
 * location, then this location is not stored in the new array.
 */
void collapse_nomsg_xy(float *xf, float *yf, int len, int is_missing_x, 
                       int is_missing_y, float *xmsg, float *ymsg,
                       int *newlen)
{
  int i;

/*
 * Special case of no missing values.
 */
  if(!is_missing_x && !is_missing_y) {
    *newlen = len;
  }
  else {
    *newlen = 0;
    for(i = 0; i < len; i++ ) {
      if((!is_missing_x || xmsg == NULL ||
          (is_missing_x && xf[i] != *xmsg)) && 
         (!is_missing_y || ymsg == NULL || 
          (is_missing_y && yf[i] != *ymsg))) {
        xf[*newlen] = xf[i];
        yf[*newlen] = yf[i];
        *newlen++;
      }
    }
  }
}

/*
 * This procedure sets a resource, given its name, and type and size
 * of its data. 
 */

void set_resource(char *resname, int rlist, void *x, 
                  const char *type_x, int ndims_x, int *dsizes_x)
{

/*
 * Check if scalar or multi-dimensional array. (A 1D array is treated
 * as a multi-d array.)
 */

  if(ndims_x == 1 && dsizes_x[0] == 1) {
    if(!strcmp(type_x,"double")) {
      NhlRLSetDouble  (rlist, resname, ((double*)x)[0]);
    }
    else if(!strcmp(type_x,"float")) {
      NhlRLSetFloat   (rlist, resname, ((float*)x)[0]);
    }
    else if(!strcmp(type_x,"integer")) {
      NhlRLSetInteger (rlist, resname, ((int*)x)[0]);
    }
  }
  else {
    if(!strcmp(type_x,"double")) {
      NhlRLSetMDDoubleArray  (rlist, resname, (double*)x, ndims_x, dsizes_x);
    }
    else if(!strcmp(type_x,"float")) {
      NhlRLSetMDFloatArray   (rlist, resname, (float*)x , ndims_x, dsizes_x);
    }
    else if(!strcmp(type_x,"integer")) {
      NhlRLSetMDIntegerArray (rlist, resname, (int*)x   , ndims_x, dsizes_x);
    }
  }
}

/*
 * Create a graphic style object so we can draw primitives on it.
 * We could have retrieved the one that is created when you create
 * a workstation object, but then if you draw a bunch of primitives, 
 * the resources that were previously set will still apply, even if
 * you create a new resource list.
 *
 * Creating a brand new graphic style object for each primitive
 * seems like the way to go.
 */

int create_graphicstyle_object(int wks)
{
  int srlist, gsid;

  srlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(srlist);
  NhlCreate(&gsid,"GraphicStyle",NhlgraphicStyleClass,wks,srlist);

  return(gsid);
}

/*
 * This function maximizes and draws the plot, and advances the frame.
 */

void draw_and_frame(int wks, int plot, gsnRes *special_res)
{
  if(special_res->gsnMaximize) maximize_plot(wks, plot, special_res);
  if(special_res->gsnDraw)  NhlDraw(plot);
  if(special_res->gsnFrame) NhlFrame(wks);
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
  int app, field, length1[1], length2[2];

/*
 * Retrieve application id.
 */
  app = NhlAppGetDefaultParentId();

/*
 * Create a scalar field object that will be used as the
 * dataset for the contour object. Check for missing values
 * here as well.
 */

  length2[0] = ylen;
  length2[1] = xlen;
  length1[0] = 1;

  set_resource("sfDataArray", sf_rlist, data, type_data, 2, length2 );

  if(is_missing_data) {
    set_resource("sfMissingValueV", sf_rlist, FillValue_data, type_data, 1, 
                 length1);
  }

/*
 * Check for coordinate arrays.
 */
 
  if(is_ycoord) {
    set_resource("sfYArray", sf_rlist, ycoord, type_ycoord, 1, &length2[0] );
  }

  if(is_ycoord) {
    set_resource("sfXArray", sf_rlist, xcoord, type_xcoord, 1, &length2[1] );
  }

/*
 * Create the object.
 */
   NhlCreate(&field,"field",NhlscalarFieldClass,app,sf_rlist);
   
   return(field);
}

/*
 * This function creates a coord arrays object that will get
 * used with the XY object. Note that X and or Y can be 1 or
 * 2-dimensional, but they must match in the rightmost dimension.
 */

int coord_array(void *x, void *y, const char *type_x, const char *type_y, 
                int ndims_x, int *dsizes_x, int ndims_y, int *dsizes_y, 
                int is_missing_x, int is_missing_y,
                void *FillValue_x, void *FillValue_y, int ca_rlist)
{
  int app, carray, length[1];

/*
 * Retrieve application id.
 */
  app = NhlAppGetDefaultParentId();

/*
 * Create a coord arrays object that will be used as the
 * dataset for an XY object.
 */

  if(x != NULL) {
    set_resource("caXArray", ca_rlist, x, type_x, ndims_x, dsizes_x );
 
    if(is_missing_x) {
      length[1] = 1;
      set_resource("caXMissingV", ca_rlist, FillValue_x, type_x, 1, 
                   &length[0]);
    }
  }

  set_resource("caYArray", ca_rlist, y, type_y, ndims_y, dsizes_y );
  
  if(is_missing_y) {
    length[1] = 1;
    set_resource("caYMissingV", ca_rlist, FillValue_y, type_y, 1, &length[0]);
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
  int app, field, length1[1], length2[2];

/*
 * Retrieve application id.
 */
  app = NhlAppGetDefaultParentId();

/*
 * Create a vector field object that will be used as the
 * dataset for the vector or streamline object.
 */

  length2[0] = ylen;
  length2[1] = xlen;

  set_resource("vfUDataArray", vf_rlist, u, type_u, 2, length2 );
  set_resource("vfVDataArray", vf_rlist, v, type_v, 2, length2 );

  length1[0] = 1;
  if(is_missing_u) {
    set_resource("vfMissingUValueV", vf_rlist, FillValue_u, type_u, 1, 
                 &length1[0] );
  }

  if(is_missing_v) {
    set_resource("vfMissingVValueV", vf_rlist, FillValue_v, type_v, 1, 
                 &length1[0] );
  }

/*
 * Check for coordinate arrays.
 */

  if(is_ycoord) {
    set_resource("vfYArray", vf_rlist, ycoord, type_ycoord, 1, &length2[0] );
  }

  if(is_xcoord) {
    set_resource("vfXArray", vf_rlist, xcoord, type_xcoord, 1, &length2[1] );
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
    filename = (char *)calloc(len+5,sizeof(char));

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
                     int sf_rlist, int cn_rlist, gsnRes *special_res)
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
 * Assign the data object.
 */

  NhlRLSetInteger(cn_rlist,"cnScalarFieldData",field);

/*
 * Create plot.
 */

  NhlCreate(&contour,"contour",NhlcontourPlotClass,wks,cn_rlist);

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->gsnScale) scale_plot(contour);

/*
 * Draw contour plot and advance frame.
 */

  draw_and_frame(wks, contour, special_res);

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
                int ca_rlist, int xy_rlist, int xyd_rlist,
                gsnRes *special_res)
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
 * Assign the data object.
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
  NhlRLGetIntegerArray(grlist,"xyCoordDataSpec",&xyds,&num_dspec);
  NhlGetValues(xy,grlist);
/*
 * Now apply the data spec resources.
 */
  NhlSetValues(*xyds,xyd_rlist);
  NhlFree(xyds);
/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->gsnScale) scale_plot(xy);

/*
 * Draw xy plot and advance frame.
 */

  draw_and_frame(wks, xy, special_res);

/*
 * Return.
 */

  return(xy);
}


/*
 * This function uses the HLUs to create an XY plot.
 */

int gsn_y_wrap(int wks, void *y, const char *type_y, int ndims_y, 
               int *dsizes_y, int is_missing_y, void *FillValue_y,
               int ca_rlist, int xy_rlist, int xyd_rlist,
               gsnRes *special_res)
{
  int xy;

/*
 * Call gsn_xy_wrap, only using NULLs for the X values.
 */
  xy = gsn_xy_wrap(wks, NULL, y, NULL, type_y, 0, NULL,
                   ndims_y, &dsizes_y[0], 0, is_missing_y, NULL, 
                   FillValue_y, ca_rlist, xy_rlist, xyd_rlist, special_res);

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
                    int vf_rlist, int vc_rlist, gsnRes *special_res)
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
 * Assign the data object.
 */

  NhlRLSetInteger(vc_rlist,"vcVectorFieldData",field);

/*
 * Create plot.
 */

  NhlCreate(&vector,"vector",NhlvectorPlotClass,wks,vc_rlist);

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->gsnScale) scale_plot(vector);

/*
 * Draw vector plot and advance frame.
 */

  draw_and_frame(wks, vector, special_res);

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
                        int vf_rlist, int st_rlist, gsnRes *special_res)
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
 * Assign the data object.
 */

  NhlRLSetInteger(st_rlist,"stVectorFieldData",field);

/*
 * Create plot.
 */

  NhlCreate(&streamline,"streamline",NhlstreamlinePlotClass,wks,st_rlist);

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->gsnScale) scale_plot(streamline);

/*
 * Draw streamline plot and advance frame.
 */

  draw_and_frame(wks, streamline, special_res);

/*
 * Return.
 */

  return(streamline);
}

/*
 * This function uses the HLUs to create a map plot.
 */

int gsn_map_wrap(int wks, int mp_rlist, gsnRes *special_res)
{
  int map;

/*
 * Create plot.
 */

  NhlCreate(&map,"map",NhlmapPlotClass,wks,mp_rlist);

/*
 * Draw map plot and advance frame.
 */

  draw_and_frame(wks, map, special_res);

/*
 * Return.
 */

  return(map);
}



/*
 * This function uses the HLUs to overlay contours on a map.
 * It is the user's responsibility to set the sfXArray/sfYArray
 * (or equivalent) coordinate arrays to the approprate lat/lon values.
 */

int gsn_contour_map_wrap(int wks, void *data, const char *type, 
                         int ylen, int xlen,
                         int is_ycoord, void *ycoord, const char *ycoord_type,
                         int is_xcoord, void *xcoord, const char *xcoord_type,
                         int is_missing, void *FillValue, 
                         int sf_rlist, int cn_rlist, int mp_rlist,
                         gsnRes *special_res)
{
  int contour, map;
  gsnRes special_res2;

/*
 * Create contour plot.
 */

  special_res2.gsnDraw     = 0;
  special_res2.gsnFrame    = 0;
  special_res2.gsnMaximize = 0;
  special_res2.gsnDebug    = special_res->gsnDebug;

  contour = gsn_contour_wrap(wks, data, type, ylen, xlen,
                             is_ycoord, ycoord, ycoord_type,
                             is_xcoord, xcoord, xcoord_type,
                             is_missing, FillValue, sf_rlist, cn_rlist,
                             &special_res2);

/*
 * Create map plot.
 */
  map = gsn_map_wrap(wks, mp_rlist, &special_res2);

/*
 * Overlay contour plot on map plot.
 */
  NhlAddOverlay(map,contour,-1);

/*
 * Make tickmarks and axis labels the same size.
 */
  if(special_res->gsnScale) scale_plot(contour);

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, map, special_res);

/*
 * Return.
 */

  return(map);
}

/*
 * This function uses the HLUs to create a vector plot over a map.
 */

int gsn_vector_map_wrap(int wks, void *u, void *v, const char *type_u,
                        const char *type_v, int ylen, int xlen, 
                        int is_ycoord, void *ycoord, const char *type_ycoord,
                        int is_xcoord, void *xcoord, const char *type_xcoord,
                        int is_missing_u, int is_missing_v, 
                        void *FillValue_u, void *FillValue_v,
                        int vf_rlist, int vc_rlist, int mp_rlist,
                        gsnRes *special_res)
{
  int vector, map;
  gsnRes special_res2;

/*
 * Create vector plot.
 */

  special_res2.gsnDraw     = 0;
  special_res2.gsnFrame    = 0;
  special_res2.gsnMaximize = 0;
  special_res2.gsnDebug    = special_res->gsnDebug;

  vector = gsn_vector_wrap(wks, u, v, type_u, type_v, ylen, xlen, is_ycoord,
                           ycoord, type_ycoord, is_xcoord, xcoord, 
                           type_xcoord, is_missing_u, is_missing_v, 
                           FillValue_u, FillValue_v, vf_rlist, vc_rlist,
                           &special_res2);

/*
 * Create map plot.
 */
  map = gsn_map_wrap(wks, mp_rlist, &special_res2);

/*
 * Overlay vector plot on map plot.
 */
  NhlAddOverlay(map,vector,-1);

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->gsnScale) scale_plot(vector);

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, map, special_res);

/*
 * Return.
 */

  return(map);
}

/*
 * This function uses the HLUs to create a streamline plot over a map.
 */

int gsn_streamline_map_wrap(int wks, void *u, void *v, const char *type_u,
                            const char *type_v, int ylen, int xlen, 
                            int is_ycoord, void *ycoord, 
                            const char *type_ycoord, int is_xcoord, 
                            void *xcoord, const char *type_xcoord,
                            int is_missing_u, int is_missing_v, 
                            void *FillValue_u, void *FillValue_v,
                            int vf_rlist, int vc_rlist, int mp_rlist,
                            gsnRes *special_res)
{
  int streamline, map;
  gsnRes special_res2;

/*
 * Create streamline plot.
 */

  special_res2.gsnDraw     = 0;
  special_res2.gsnFrame    = 0;
  special_res2.gsnMaximize = 0;
  special_res2.gsnDebug    = special_res->gsnDebug;

  streamline = gsn_streamline_wrap(wks, u, v, type_u, type_v, ylen, xlen, 
                                   is_ycoord, ycoord, type_ycoord, 
                                   is_xcoord, xcoord, type_xcoord, 
                                   is_missing_u, is_missing_v, FillValue_u, 
                                   FillValue_v, vf_rlist, vc_rlist,
                                   &special_res2);

/*
 * Create map plot.
 */
  map = gsn_map_wrap(wks, mp_rlist, &special_res2);

/*
 * Overlay streamline plot on map plot.
 */
  NhlAddOverlay(map,streamline,-1);

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->gsnScale) scale_plot(streamline);

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, map, special_res);

/*
 * Return.
 */

  return(map);
}


/*
 * This function uses the HLUs to create a vector plot colored by
 * a scalar field.
 */

int gsn_vector_scalar_wrap(int wks, void *u, void *v, void *t, 
                           const char *type_u, const char *type_v, 
                           const char *type_t, int ylen, int xlen, 
                           int is_ycoord, void *ycoord, 
                           const char *type_ycoord, int is_xcoord, 
                           void *xcoord, const char *type_xcoord, 
                           int is_missing_u, int is_missing_v, 
                           int is_missing_t, void *FillValue_u, 
                           void *FillValue_v, void *FillValue_t,
                           int vf_rlist, int sf_rlist, int vc_rlist, 
                           gsnRes *special_res)
{
  int vfield, sfield, vector;

/*
 * Create vector and scalar field objects that will be used as the
 * datasets for the vector object.
 */

  vfield = vector_field(u, v, type_u, type_v, ylen, xlen, 
                        is_ycoord, ycoord, type_ycoord, 
                        is_xcoord, xcoord, type_xcoord, 
                        is_missing_u, is_missing_v, 
                        FillValue_u, FillValue_v, vf_rlist);

  sfield = scalar_field(t, type_t, ylen, xlen, is_ycoord, ycoord, 
                        type_ycoord, is_xcoord, xcoord, type_xcoord, 
                        is_missing_t, FillValue_t, sf_rlist);

/*
 * Assign the data objects and create vector object.
 */

  NhlRLSetInteger(vc_rlist, "vcVectorFieldData",    vfield);
  NhlRLSetInteger(vc_rlist, "vcScalarFieldData",    sfield);
  NhlRLSetString (vc_rlist, "vcUseScalarArray",     "True");
  NhlRLSetString (vc_rlist, "vcMonoLineArrowColor", "False");

  NhlCreate(&vector,"vector",NhlvectorPlotClass,wks,vc_rlist);

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->gsnScale) scale_plot(vector);

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, vector, special_res);

/*
 * Return.
 */

  return(vector);
}

/*
 * This function uses the HLUs to create a vector plot colored by
 * a scalar field overlaid on a map.
 */

int gsn_vector_scalar_map_wrap(int wks, void *u, void *v, void *t, 
                               const char *type_u, const char *type_v, 
                               const char *type_t, int ylen, int xlen, 
                               int is_ycoord, void *ycoord, 
                               const char *type_ycoord, int is_xcoord, 
                               void *xcoord, const char *type_xcoord, 
                               int is_missing_u, int is_missing_v, 
                               int is_missing_t, void *FillValue_u, 
                               void *FillValue_v, void *FillValue_t,
                               int vf_rlist, int sf_rlist, int vc_rlist, 
                               int mp_rlist, gsnRes *special_res)
{
  int vector, map;
  gsnRes special_res2;

/*
 * Create vector plot.
 */

  special_res2.gsnDraw     = 0;
  special_res2.gsnFrame    = 0;
  special_res2.gsnMaximize = 0;
  special_res2.gsnDebug    = special_res->gsnDebug;

  vector = gsn_vector_scalar_wrap(wks, u, v, t, type_u, type_v, type_t,
                                  ylen, xlen, is_ycoord, ycoord, 
                                  type_ycoord, is_xcoord, xcoord, 
                                  type_xcoord, is_missing_u, is_missing_v, 
                                  is_missing_t, FillValue_u, FillValue_v, 
                                  FillValue_t, vf_rlist, sf_rlist, vc_rlist, 
                                  &special_res2);

/*
 * Create map plot.
 */
  map = gsn_map_wrap(wks, mp_rlist, &special_res2);

/*
 * Overlay vector plot on map plot.
 */
  NhlAddOverlay(map,vector,-1);

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->gsnScale) scale_plot(vector);

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, map, special_res);

/*
 * Return.
 */

  return(map);
}

int gsn_text_ndc_wrap(int wks, char* string, void *x, void *y,
                      const char *type_x, const char *type_y,
                      int tx_rlist, gsnRes *special_res)
{
  int text, length[1];

  length[1] = 1;

  set_resource("txPosXF", tx_rlist, x, type_x, 1, &length[1] );
  set_resource("txPosYF", tx_rlist, y, type_y, 1, &length[1] );


  NhlRLSetString(tx_rlist, "txString", string);
  NhlCreate(&text,"text",NhltextItemClass,wks,tx_rlist);
/*
 * Draw text.
 */

  draw_and_frame(wks, text, special_res);

/*
 * Return.
 */

  return(text);
}


int gsn_text_wrap(int wks, int plot, char* string, void *x, void *y, 
                  const char *type_x, const char *type_y, int tx_rlist, 
                  gsnRes *special_res)
{
  float *xf, *yf, xndc, yndc, oor = 0.;
  int text, status;

/*
 * Convert x and y to float, since NhlDatatoNDC routine only accepts 
 * floats.
 */
  xf = coerce_to_float(x,type_x,1);
  yf = coerce_to_float(y,type_y,1);

/*
 * Convert from plot's data space to NDC space.
 */

  (void)NhlDataToNDC(plot,xf,yf,1,&xndc,&yndc,NULL,NULL,&status,&oor);

  if(special_res->gsnDebug) {
    printf("gsn_text: string = %s x = %g y = %g xndc = %g yndc = %g\n", 
           string, *xf, *yf, xndc, yndc);
  }

  text = gsn_text_ndc_wrap(wks, string, &xndc, &yndc, "float", "float",
                           tx_rlist, special_res);

/*
 * Return.
 */

  return(text);
}

/*
 * Routine for drawing any kind of primitive in NDC or data space.
 */
void gsn_poly_wrap(int wks, int plot, void *x, void *y, const char *type_x,
                   const char *type_y, int len, int is_missing_x, 
                   int is_missing_y, void *FillValue_x, 
                   void *FillValue_y, NhlPolyType polytype, int is_ndc,
                   int gs_rlist, gsnRes *special_res)
{
  int gsid, newlen;
  float *xf, *yf, *xfmsg, *yfmsg;

/*
 * Determine if we need to convert x and/or y (and their missing values)
 * to float.
 */
  xf  = coerce_to_float(x,type_x,len);
  yf  = coerce_to_float(y,type_y,len);
  if(is_missing_x) xfmsg = coerce_to_float(FillValue_x,type_x,1);
  if(is_missing_y) yfmsg = coerce_to_float(FillValue_y,type_y,1);

/*
 * Remove missing values, if any.
 */
  collapse_nomsg_xy(xf,yf,len,is_missing_x,is_missing_y,xfmsg,yfmsg,&newlen);

/*
 * Create graphic style object on which to draw primitives.
 */
  gsid = create_graphicstyle_object(wks);

/*
 * Set the graphic style resources, if any.
 */

  NhlSetValues(gsid,gs_rlist);

/*
 * Draw the appropriate primitive.
 */
  if(special_res->gsnDraw) {
    if(is_ndc) {
      switch(polytype) {

      case NhlPOLYLINE:
        NhlNDCPolyline(wks,gsid,xf,yf,newlen);
        break;

      case NhlPOLYMARKER:
        NhlNDCPolymarker(wks,gsid,xf,yf,newlen);
        break;

      case NhlPOLYGON:
        NhlNDCPolygon(wks,gsid,xf,yf,newlen);
        break;
      }
    }
    else {
      switch(polytype) {

      case NhlPOLYLINE:
        NhlDataPolyline(plot,gsid,xf,yf,newlen);
        break;

      case NhlPOLYMARKER:
        NhlDataPolymarker(plot,gsid,xf,yf,newlen);
        break;

      case NhlPOLYGON:
        NhlDataPolygon(plot,gsid,xf,yf,newlen);
        break;
      }
    }
  }
    
  if(special_res->gsnFrame) NhlFrame(wks);

}


/*
 * Routine for adding any kind of primitive (in data space only).
 * The difference between adding a primitive, and just drawing a 
 * primitive, is that when you add a primitive to a plot, then it will
 * only get drawn when you draw the plot, and it will also be scaled
 * if you scale the plot. 
 */
int gsn_add_poly_wrap(int wks, int plot, void *x, void *y,
                      const char *type_x, const char *type_y, int len, 
                      int is_missing_x, int is_missing_y, void *FillValue_x, 
                      void *FillValue_y, NhlPolyType polytype, 
                      int gs_rlist, gsnRes *special_res)
{
  int *primitive_object, gsid, pr_rlist, grlist;
  int i, newlen, *indices, nlines, ibeg, iend, npts;
  float *xf, *yf, *xfmsg, *yfmsg;
  char *astring;

/*
 * Create resource list for primitive object.
 */

  pr_rlist = NhlRLCreate(NhlSETRL);

/*
 * Create graphic style object on which to draw primitives.
 */

  gsid = create_graphicstyle_object(wks);

/*
 * Set the graphic style (primitive) resources, if any.
 */

  NhlSetValues(gsid,gs_rlist);

/*
 * Convert x and/or y (and their missing values) to float, if necessary.
 */
  xf  = coerce_to_float(x,type_x,len);
  yf  = coerce_to_float(y,type_y,len);
  if(is_missing_x) xfmsg = coerce_to_float(FillValue_x,type_x,1);
  if(is_missing_y) yfmsg = coerce_to_float(FillValue_y,type_y,1);

/*
 * If the poly type is polymarkers or polygons, then remove all
 * missing values, and plot.
 */

  if(polytype != NhlPOLYLINE) {
/*
 * Remove missing values, if any.
 */
    collapse_nomsg_xy(xf, yf, len, is_missing_x, is_missing_y, 
                      xfmsg, yfmsg, &newlen);
/*
 * Set some primitive object resources.  Namely, the location of
 * the X/Y points, and the type of primitive (polymarker or polygon
 * in this case).
 */
    NhlRLSetFloatArray(pr_rlist,"prXArray",       xf, newlen);
    NhlRLSetFloatArray(pr_rlist,"prYArray",       yf, newlen);
    NhlRLSetInteger   (pr_rlist,"prPolyType",     polytype);
    NhlRLSetInteger   (pr_rlist,"prGraphicStyle", gsid);

/*
 * Allocate a variable to hold the primitive object, and create it.
 */

    primitive_object = (int*)calloc(1,sizeof(int));
    NhlCreate(primitive_object,"Primitive",NhlprimitiveClass,wks,pr_rlist);

/*
 * Attach primitive object to the plot.
 */

    NhlAddPrimitive(plot,*primitive_object,-1);
  }
  else {
/*
 * If the primitive is a polyline, then retrieve the indices of the 
 * non-missing points, and plot them individually.  This may result
 * in several primitive objects being created.  If there's only one
 * point in a section, then plot a marker.
 */
    indices = get_non_missing_pairs(xf, yf, is_missing_x, is_missing_y,
                                    xfmsg, yfmsg, len, &nlines);

    if(nlines > 0) {
      primitive_object = (int*)calloc(nlines,sizeof(int));
      astring = calloc(strlen(polylinestr)+8,sizeof(char));
      for(i = 0; i < nlines; i++) {
/*
 * Get the begin and end indices of the non-missing section of points.
 */
        ibeg = indices[i*2];
        iend = indices[i*2+1];
/*
 * Create a unique string to name this polyline. 
 */
        sprintf(astring,"%s%d", polylinestr, i);

/*
 * If iend=ibeg, then this means we only have one point, and thus
 * we need to create a marker.
 */
        if(iend == ibeg) {
/*
 * Create primitive object.
 */
          NhlRLSetFloat  (pr_rlist,"prXArray",       xf[ibeg]);
          NhlRLSetFloat  (pr_rlist,"prYArray",       yf[ibeg]);
          NhlRLSetInteger(pr_rlist,"prPolyType",     NhlPOLYMARKER);
          NhlRLSetInteger(pr_rlist,"prGraphicStyle", gsid);
          NhlCreate(&primitive_object[i],astring,NhlprimitiveClass,wks,
                    pr_rlist);
        }
        else {
          npts = iend - ibeg + 1;
/*
 * We have more than one point, so create a polyline.
 */

          NhlRLSetFloatArray(pr_rlist,"prXArray",       xf, npts);
          NhlRLSetFloatArray(pr_rlist,"prYArray",       yf, npts);
          NhlRLSetInteger   (pr_rlist,"prPolyType",     polytype);
          NhlRLSetInteger   (pr_rlist,"prGraphicStyle", gsid);
        }

/*
 * Create the polyline or marker, and  attach it to the plot.
 */
        NhlCreate(&primitive_object[i],"Primitive",NhlprimitiveClass,
                  wks,pr_rlist);

        NhlAddPrimitive(plot,primitive_object[i],-1);
      }
    }
    else {
/*
 * Create a NULL primitive object.
 */
      primitive_object = (int*)calloc(1,sizeof(int));
    }     
  }

  return(*primitive_object);
}

/*
 * Routine for drawing markers in NDC space.
 */
void gsn_polymarker_ndc_wrap(int wks, void *x, void *y, const char *type_x, 
                             const char *type_y,  int len,
                             int is_missing_x, int is_missing_y, 
                             void *FillValue_x, void *FillValue_y, 
                             int gs_rlist, gsnRes *special_res)
{
  gsn_poly_wrap(wks, 0, x, y, type_x, type_y, len, is_missing_x, 
                is_missing_y, FillValue_x, FillValue_y, NhlPOLYMARKER, 1,
                gs_rlist,special_res);
}


/*
 * Routine for drawing lines in NDC space.
 */
void gsn_polyline_ndc_wrap(int wks, void *x, void *y, const char *type_x,
                           const char *type_y, int len,
                           int is_missing_x, int is_missing_y, 
                           void *FillValue_x, void *FillValue_y, 
                           int gs_rlist, gsnRes *special_res)
{
  gsn_poly_wrap(wks, 0, x, y, type_x, type_y, len, is_missing_x, 
                is_missing_y, FillValue_x, FillValue_y, NhlPOLYLINE, 1, 
                gs_rlist, special_res);
}


/*
 * Routine for drawing polygons in NDC space.
 */
void gsn_polygon_ndc_wrap(int wks, void *x, void *y, const char *type_x, 
                          const char *type_y, int len,
                          int is_missing_x, int is_missing_y, 
                          void *FillValue_x, void *FillValue_y, 
                          int gs_rlist, gsnRes *special_res)
{
  gsn_poly_wrap(wks, 0, x, y, type_x, type_y, len, is_missing_x,
                is_missing_y, FillValue_x, FillValue_y, NhlPOLYGON, 1,
                gs_rlist, special_res);
}

/*
 * Routine for drawing markers in data space.
 */
void gsn_polymarker_wrap(int wks, int plot, void *x, void *y, 
                         const char *type_x, const char *type_y, int len,
                         int is_missing_x, int is_missing_y, 
                         void *FillValue_x, void *FillValue_y, 
                         int gs_rlist, gsnRes *special_res)
{
  gsn_poly_wrap(wks,plot,x,y,type_x,type_y,len,is_missing_x,is_missing_y,
                FillValue_x,FillValue_y,NhlPOLYMARKER,0,gs_rlist,special_res);
}


/*
 * Routine for drawing lines in data space.
 */
void gsn_polyline_wrap(int wks, int plot, void *x, void *y, 
                       const char *type_x, const char *type_y, int len, 
                       int is_missing_x, int is_missing_y, 
                       void *FillValue_x, void *FillValue_y, 
                       int gs_rlist, gsnRes *special_res)
{
  gsn_poly_wrap(wks,plot,x,y,type_x,type_y,len,is_missing_x,is_missing_y,
                FillValue_x,FillValue_y,NhlPOLYLINE,0,gs_rlist,special_res);
}

/*
 * Routine for drawing polygons in data space.
 */
void gsn_polygon_wrap(int wks, int plot, void *x, void *y, 
                      const char *type_x, const char *type_y, int len, 
                      int is_missing_x, int is_missing_y, 
                      void *FillValue_x, void *FillValue_y, 
                      int gs_rlist, gsnRes *special_res)
{
  gsn_poly_wrap(wks, plot, x, y, type_x, type_y, len, is_missing_x, 
                is_missing_y, FillValue_x, FillValue_y, NhlPOLYGON, 0,
                gs_rlist, special_res);
}

/*
 * Routine for adding polylines to a plot (in the plot's data space).
 */
int gsn_add_polyline_wrap(int wks, int plot, void *x, void *y, 
                          const char *type_x, const char *type_y, int len, 
                          int is_missing_x, int is_missing_y, 
                          void *FillValue_x, void *FillValue_y, 
                          int gs_rlist, gsnRes *special_res)
{
  int ipoly;

  ipoly = gsn_add_poly_wrap(wks, plot, x, y, type_x, type_y, len, 
                            is_missing_x, is_missing_y, FillValue_x, 
                            FillValue_y, NhlPOLYLINE, gs_rlist, 
                            special_res);
  return(ipoly);

}


/*
 * Routine for adding polymarkers to a plot (in the plot's data space).
 */
int gsn_add_polymarker_wrap(int wks, int plot, void *x, void *y, 
                          const char *type_x, const char *type_y, int len, 
                          int is_missing_x, int is_missing_y, 
                          void *FillValue_x, void *FillValue_y, 
                          int gs_rlist, gsnRes *special_res)
{
  int ipoly;

  ipoly = gsn_add_poly_wrap(wks, plot, x, y, type_x, type_y, len, 
                            is_missing_x, is_missing_y, FillValue_x, 
                            FillValue_y, NhlPOLYMARKER, gs_rlist, 
                            special_res);
  return(ipoly);

}


/*
 * Routine for adding polygons to a plot (in the plot's data space).
 */
int gsn_add_polygon_wrap(int wks, int plot, void *x, void *y, 
                          const char *type_x, const char *type_y, int len, 
                          int is_missing_x, int is_missing_y, 
                          void *FillValue_x, void *FillValue_y, 
                          int gs_rlist, gsnRes *special_res)
{
  int ipoly;

  ipoly = gsn_add_poly_wrap(wks, plot, x, y, type_x, type_y, len, 
                            is_missing_x, is_missing_y, FillValue_x, 
                            FillValue_y, NhlPOLYGON, gs_rlist, 
                            special_res);
  return(ipoly);

}


/*
 * Routine for adding text (in data space only) to a plot.
 *
 * The difference between adding text and just drawing text
 * is that when you add a text string to a plot, it will
 * only get drawn when you draw the plot. It will also be scaled
 * appropriately if you scale the plot. 
 */
int gsn_add_text_wrap(int wks, int plot, char *string, void *x, void *y,
                      const char *type_x, const char *type_y,
                      int tx_rlist, int am_rlist, gsnRes *special_res)
{
/*
 * We need to make this "100" number be the maximum number of annotations
 * allowed attached to a plot.
 */
  int i, srlist, grlist, text, just;
  int *anno_views, *anno_mgrs, *new_anno_views, num_annos;

  float *xf, *yf;

/*
 * First create the text object with the given string.
 */

  NhlRLSetString(tx_rlist, "txString", string);
  NhlCreate(&text,"text",NhltextItemClass,wks,tx_rlist);

/*
 * Get current list of annotations already attached to the plot.
 */

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  NhlRLGetIntegerArray(grlist,"pmAnnoViews",&anno_views,&num_annos);
  NhlGetValues(plot,grlist);

/*
 * Make sure the new text string is first in the list of annotations,
 * if any.
 */
  if(num_annos > 0) {
    new_anno_views = (int *)calloc(1+num_annos,sizeof(int));
    new_anno_views[0] = text;
    for( i = 1; i <= num_annos; i++ ) {
      new_anno_views[i] = anno_views[i-1];
    }
    num_annos++;
  }
  else {
    new_anno_views  = (int *)calloc(1,sizeof(int));
    *new_anno_views = text;
    num_annos = 1;
  }

/*
 * Set the old and new annotations, with the new text being first.
 */

  srlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(srlist);
  NhlRLSetIntegerArray(srlist,"pmAnnoViews",new_anno_views,num_annos);
  NhlSetValues(plot,srlist);

/*
 * Retrieve the ids of the AnnoManager objects created by the
 * PlotManager.
 */

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  NhlRLGetIntegerArray(grlist,"pmAnnoManagers",&anno_mgrs,&num_annos);
  NhlGetValues(plot,grlist);

/*
 * Get the text justification and use it for the anno justification.
 */

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  NhlRLGetInteger(grlist,"txJust",&just);
  NhlGetValues(text,grlist);

/*
 * Convert x and y to float.
 */
  xf = coerce_to_float(x,type_x,1);
  yf = coerce_to_float(y,type_y,1);

/*
 * Set the X/Y location and the justification of the new annotation.
 */

  NhlRLSetFloat  (am_rlist,"amDataXF",       *xf);
  NhlRLSetFloat  (am_rlist,"amDataYF",       *yf);
  NhlRLSetString (am_rlist,"amResizeNotify", "True");
  NhlRLSetString (am_rlist,"amTrackData",    "True");
  NhlRLSetInteger(am_rlist,"amJust",         just);
  NhlSetValues(anno_mgrs[0],am_rlist);

/*
 * Return.
 */

  return(anno_mgrs[0]);

}

