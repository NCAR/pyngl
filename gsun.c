#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gsun.h"

/*
 * Define some global strings.
 */
const char *polylinestr = "polyline";
  
/*
 *  This function calculates the maximum value of a 1D int array.
 */
int imax_array(int *array, int npts)
{
  int i, imax;

  imax = array[0];
  for(i = 1; i < npts; i++) imax = max(array[i],imax);
  return(imax);
}

/*
 *  This function calculates the maximum value of a 1D float array.
 */
float xmax_array(float *array, int npts)
{
  int i;
  float xmax;

  xmax = array[0];
  for(i = 1; i < npts; i++) xmax = max(array[i],xmax);
  return(xmax);
}

/*
 *  This function calculates the minimum value of a 1D float array.
 */
float xmin_array(float *array, int npts)
{
  int i;
  float xmin;

  xmin = array[0];
  for(i = 1; i < npts; i++) xmin = min(array[i],xmin);
  return(xmin);
}

/*
 * This function creates npts values from start to end.
 */
float *fspan(float start, float end, int npts)
{
  int i;
  float spacing, *ret_val;
  
  ret_val = (float *)malloc(npts*sizeof(float));
  spacing = (end - start) / (npts - 1);
  for (i = 0; i < npts; i++) {
    ret_val[i] = start + (i * spacing);
  }
/*
 * Make sure first and last points are exactly start and end.
 */
  ret_val[0]      = start;
  ret_val[npts-1] = end;

  return(ret_val);
}

/*
 * This function creates values from start to end with skip.
 */
int *ispan(int start, int end, int nskip)
{
  int i, npts, *ret_val;

  if(end > start) {
    npts = ((end-start)/nskip) + 1;

    ret_val = (int *)malloc(npts*sizeof(int));
    for (i = 0; i < npts; i++) {
      ret_val[i] = start + (i * nskip);
    }
  }    
  else {
    npts = ((start-end)/nskip) + 1;

    ret_val = (int *)malloc(npts*sizeof(int));
    for (i = 0; i < npts; i++) {
      ret_val[i] = start - (i * nskip);
    }
  }

  return(ret_val);
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
 * "nglPaperOrientation" - orientation of paper. Can be "landscape",
 *                         "portrait", or "auto". Default is "auto".
 *
 *       "nglPaperWidth"  - width of paper (in inches, default is 8.5)
 *       "nglPaperHeight" - height of paper (in inches, default is 11.0)
 *       "nglPaperMargin" - margin to leave around plots (in inches,
 *                        default is 0.5)
 *
 */
void compute_ps_device_coords(int wks, nglPlotId *plots, int nplots, 
                              nglRes *special_res)
{
  NhlBoundingBox *box;
  float top, bot, lft, rgt, dpi_pw, dpi_ph, dpi_margin;
  float paper_width, paper_height, paper_margin;
  float pw, ph, lx, ly, ux, uy, dw, dh, ndc2du;
  int i, srlist, is_debug, dpi, coords[4];
  int lft_pnl, rgt_pnl, top_pnl, bot_pnl;
  int lft_inv_pnl, rgt_inv_pnl, top_inv_pnl, bot_inv_pnl;
  NhlWorkOrientation paper_orient; 

/*
 * These four paper* resources are for PDF/PS output.
 */
  paper_orient = special_res->nglPaperOrientation;
  paper_height = special_res->nglPaperHeight;
  paper_width  = special_res->nglPaperWidth;
  paper_margin = special_res->nglPaperMargin;
  is_debug     = special_res->nglDebug;

/*
 * Check to see if any panel resources have been set. They are only 
 * used if they have been explicitly set by the user.
 */
  if(special_res->nglPanelLeft != 0.) {
    lft_pnl = 1;
  }
  else {
    lft_pnl = 0;
  }
  if(special_res->nglPanelRight != 1.) {
    rgt_pnl = 1;
  }
  else {
    rgt_pnl = 0;
  }
  if(special_res->nglPanelBottom != 0.) {
    bot_pnl = 1;
  }
  else {
    bot_pnl = 0;
  }
  if(special_res->nglPanelTop != 1.) {
    top_pnl = 1;
  }
  else {
    top_pnl = 0;
  }

  if(special_res->nglPanelInvsblLeft == -999) {
    lft_inv_pnl = 0;
  }
  else {
    lft_inv_pnl = 1;
  }
  if(special_res->nglPanelInvsblRight == -999) {
    rgt_inv_pnl = 0;
  }
  else {
    rgt_inv_pnl = 1;
  }
  if(special_res->nglPanelInvsblBottom == -999) {
    bot_inv_pnl = 0;
  }
  else {
    bot_inv_pnl = 1;
  }
  if(special_res->nglPanelInvsblTop == -999) {
    top_inv_pnl = 0;
  }
  else {
    top_inv_pnl = 1;
  }

/*
 * Get the bounding box that encompasses the plot(s). Note that even
 * though the bounding box coordinates should be positive, it is
 * possible for them to be negative, and we need to keep these
 * negative values in our calculations later to preserve the 
 * aspect ratio.
 */
  box = (NhlBoundingBox *)malloc(nplots*sizeof(NhlBoundingBox));
  NhlGetBB(*(plots[0].base),&box[0]);
  top = box[0].t;
  bot = box[0].b;
  lft = box[0].l;
  rgt = box[0].r;

/*
 * Get largest bounding box that encompasses all non-missing graphical
 * objects.
 */
  for( i = 1; i < nplots; i++ ) { 
    NhlGetBB(*(plots[i].base),&box[i]);
    top = max(top,box[i].t);
    bot = min(bot,box[i].b);
    lft = min(lft,box[i].l);
    rgt = max(rgt,box[i].r);
  }
  if(top_inv_pnl) {
    top = max(special_res->nglPanelInvsblTop,top);
  }
  else if(top_pnl) {
    top = max(1.,top);
  }
  if(bot_inv_pnl) {
    bot = min(special_res->nglPanelInvsblBottom,bot);
  }
  else if(bot_pnl) {
    bot = min(0.,bot);
  }
  if(lft_inv_pnl) {
    lft = min(special_res->nglPanelInvsblLeft,lft);
  }
  else if(lft_pnl) {
    lft = min(0.,lft);
  }
  if(rgt_inv_pnl) {
    rgt = max(special_res->nglPanelInvsblRight,rgt);
  }
  else if(rgt_pnl) {
    rgt = max(1.,rgt);
  }

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
 
  if( (paper_orient == NhlPORTRAIT) || ((paper_orient == -1) &&
     (ph / pw) >= 1.0)) {
/*
 * If plot is higher than it is wide, then default to portrait if
 * orientation is not specified.
 */
    paper_orient = NhlPORTRAIT;

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
    paper_orient = NhlLANDSCAPE;

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
    if(paper_orient == NhlLANDSCAPE) {
      printf("    wkOrientation  = landscape\n");
    }     
    else if(paper_orient == NhlPORTRAIT) {
      printf("    wkOrientation  = portrait\n");
    }
    else {
      printf("    wkOrientation  = unknown\n");
    }
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
  NhlRLSetInteger(srlist,"wkOrientation",  paper_orient);
  (void)NhlSetValues(wks, srlist);

/*
 * Free up memory.
 */
  free(box);
}


/*
 * This function maximizes the size of the plot in the viewport.
 */

void maximize_plot(int wks, nglPlotId *plot, int nplots, int ispanel, 
                   nglRes *special_res)
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
  NhlRLClear(grlist);

/*
 * If dealing with paneled plots, then this means that the
 * viewport coordinates have already been optimized, and thus
 * we don't need to calculate them again.
 */

  if(!ispanel) {
/*
 * Get bounding box of plot.
 */
    NhlGetBB(*(plot[0].base),&box);

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
    (void)NhlGetValues(*(plot[0].base), grlist);

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
 * Set new coordinates 
 */

    NhlRLClear(srlist);
    NhlRLSetFloat(srlist,"vpXF",      new_vpx);
    NhlRLSetFloat(srlist,"vpYF",      new_vpy);
    NhlRLSetFloat(srlist,"vpWidthF",  new_vpw);
    NhlRLSetFloat(srlist,"vpHeightF", new_vph);
    (void)NhlSetValues(*(plot[0].base), srlist);

    if(special_res->nglDebug) {
      printf("vpXF/vpYF/vpWidthF/vpHeightF = %g / %g / %g / %g\n", new_vpx, new_vpy, new_vpw, new_vph);
    }
  }

  if(!strcmp(NhlClassName(wks),"psWorkstationClass") ||
     !strcmp(NhlClassName(wks),"pdfWorkstationClass")) {
/*
 * Compute and set device coordinates that will make plot fill the 
 * whole page.
 */
    compute_ps_device_coords(wks, plot, nplots, special_res);
  }
}

/*
 * Function : spread_colors
 *
 * By default, all of the plotting routines use the first n colors from
 * a color map, where "n" is the number of contour or vector levels
 * If "nglSpreadColors" is set to  True, then the colors are spanned
 * across the whole color map. The min_index and max_index values are
 * used for the start and end colors.  If either min_index or max_index
 * is < 0 (but not both), then this indicates to use ncol-i, where "i
 * is equal to the negative value
 *
 * If after adjusting for negative index color(s), and
 * max_index < min_index, then the colors are reversed
 */
void spread_colors(int wks, int plot, int min_index, int max_index, 
                   char *get_resname, char *set_resname, int debug)
{
  int i, ncols, lcount, *icols, minix, maxix, reverse, grlist, srlist, itmp;
  float fmin, fmax, *fcols;

  srlist = NhlRLCreate(NhlSETRL);
  grlist = NhlRLCreate(NhlGETRL);

/*
 * Get number of colors in color map so we know how much
 * to spread the colors.
 */
  NhlRLClear(grlist);
  NhlRLGetInteger(grlist,"wkColorMapLen", &ncols);
  NhlGetValues(wks,grlist);

/*
 * Retrieve the appropriate resource that will be used to
 * determine how many fill colors we need.
 */
  NhlRLClear(grlist);
  NhlRLGetInteger(grlist,get_resname,&lcount);
  NhlGetValues(plot,grlist);

/*
 * -1 indicates that min/max_index should be set equal to ncols - 1
 * -2 indicates that min/max_index should be set equal to ncols - 2, etc.
 *
 * If after adjusting for negative indices, and maxix < minix, then 
 * this implies that the user wants to reverse the colors.
 */

  if (min_index < 0) {
    minix = ncols + min_index;
  }
  else {
    minix = min_index;
  }

  if (max_index < 0) {
    maxix = ncols + max_index;
  }
  else {
    maxix = max_index;
  }

/*
 * Make sure indices fall within range of the color map.
 */
  minix = min(ncols-1,max(0,minix));
  maxix = min(ncols-1,max(0,maxix));

/*
 * If maxix < minix, then colors are to be reversed.
 */
  reverse = 0;
  if(maxix < minix) {
    reverse = 1;
    itmp    = maxix;
    maxix   = minix;
    minix   = itmp;
  }

  fmin = minix;
  fmax = maxix;
  fcols = fspan(fmin,fmax,lcount+1);
  icols = (int*)malloc((lcount+1)*sizeof(int));
  if(!reverse) {
    for(i = 0; i <= lcount; i++) icols[i] = (int)(fcols[i] + 0.5);
  }
  else {
    for(i = lcount; i >= 0; i--) icols[i] = (int)(fcols[i] + 0.5);
  }

  if(debug) {
    printf("Original min_index           = %d\n", min_index);
    printf("Original max_index           = %d\n", max_index);
    printf("Number of colors in colormap = %d\n", ncols);
    printf("Number of levels to fill     = %d\n", lcount);
    printf("Minimum color index          = %d\n", minix);
    printf("Maximum color index          = %d\n", maxix);
    printf("\n");
  }

/*
 * Set the appropriate resource that uses the newly calculated
 * list of color indices.
 */
  NhlRLClear(srlist);
  NhlRLSetIntegerArray(srlist, set_resname, icols, lcount+1);
  NhlSetValues(plot,srlist);

  free(icols);
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
 * This function removes all the missing values from a pair of
 * float arrays. If they are missing in one or both arrays in the same
 * location, then this location is not stored in the new array.
 */
void collapse_nomsg_xy(float *xf, float *yf, float **xfnew, float **yfnew,
                       int len, int is_missing_x, int is_missing_y, 
                       float *xmsg, float *ymsg, int *newlen)
{
  int i;
  float *xtmp, *ytmp;

/*
 * Special case of no missing values.
 */
  if(!is_missing_x && !is_missing_y) {
    *newlen = len;
    *xfnew = (float*)malloc(len*sizeof(float));
    *yfnew = (float*)malloc(len*sizeof(float));
    memcpy((void *)(*xfnew),(const void *)xf,len*sizeof(float));
    memcpy((void *)(*yfnew),(const void *)yf,len*sizeof(float));
  }
  else {
    *newlen = 0;
    xtmp = (float*)malloc(len*sizeof(float));
    ytmp = (float*)malloc(len*sizeof(float));
    for(i = 0; i < len; i++ ) {
      if((!is_missing_x || xmsg == NULL ||
          (is_missing_x && xf[i] != *xmsg)) && 
         (!is_missing_y || ymsg == NULL || 
          (is_missing_y && yf[i] != *ymsg))) {
        xtmp[*newlen] = xf[i];
        ytmp[*newlen] = yf[i];
        (*newlen)++;
      }
    }
    *xfnew = (float*)malloc(*newlen*sizeof(float));
    *yfnew = (float*)malloc(*newlen*sizeof(float));
    memcpy((void *)(*xfnew),(const void *)&xtmp[0],*newlen*sizeof(float));
    memcpy((void *)(*yfnew),(const void *)&ytmp[0],*newlen*sizeof(float));
    free(xtmp);
    free(ytmp);
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
 * This function sets all HLU objects ids to -1.
 */
void initialize_ids(nglPlotId *plot)
{
  plot->base       = NULL;
  plot->contour    = NULL;
  plot->vector     = NULL;
  plot->streamline = NULL;
  plot->map        = NULL;
  plot->xy         = NULL;
  plot->xydspec    = NULL;
  plot->text       = NULL;
  plot->primitive  = NULL;
  plot->cafield    = NULL;
  plot->sffield    = NULL;
  plot->vffield    = NULL;

  plot->nbase       = 0;
  plot->ncontour    = 0;
  plot->nvector     = 0;
  plot->nstreamline = 0;
  plot->nmap        = 0;
  plot->nxy         = 0;
  plot->nxydspec    = 0;
  plot->ntext       = 0;
  plot->nprimitive  = 0;
  plot->ncafield    = 0;
  plot->nsffield    = 0;
  plot->nvffield    = 0;
}

/*
 * This function initializes all of the special resources.
 */
void initialize_resources(nglRes *res, int list_type)
{
/*
 * We need to determine if this resource list is for the plotting
 * routines (ngl_contour, ngl_map, ngl_vector, etc) or for the
 * primitive and text routines (ngl_polymarker, ngl_text, ngl_add text, 
 * etc).  This is because for primitives and text, we don't want 
 * Maximize or Frame to be on by default.
 *
 * list_type = 0 ==> generic resource list (for data plotting routines)
 * list_type = 1 ==> primitive resource list
 */

  switch(list_type) {
  case nglPlot:
    res->nglMaximize = 1;
    res->nglFrame    = 1;
    break;

  case nglPrimitive:
    res->nglMaximize = 0;
    res->nglFrame    = 0;
    break;

  default:
    res->nglMaximize = 1;
    res->nglFrame    = 1;
  }

  res->nglDraw  = 1;
  res->nglDebug = 0;
  res->nglScale = 0;

/*
 * Special resources for spanning a full color map when
 * filling vectors and/or contours.
 */
  res->nglSpreadColors     =  1;
  res->nglSpreadColorStart =  2;
  res->nglSpreadColorEnd   = -1;

/*
 * Special resources for PS/PDF output.
 */
  res->nglPaperOrientation = -1;
  res->nglPaperWidth       =  8.5;
  res->nglPaperHeight      = 11.0;
  res->nglPaperMargin      =  0.5;

/*
 * Special resources for paneling.
 */
  res->nglPanelSave               = 0;
  res->nglPanelCenter             = 0;
  res->nglPanelRowSpec            = 0;
  res->nglPanelXWhiteSpacePercent = 1.;
  res->nglPanelYWhiteSpacePercent = 1.;
  res->nglPanelBoxes              = 0;
  res->nglPanelLeft               = 0.;
  res->nglPanelRight              = 1.;
  res->nglPanelBottom             = 0.;
  res->nglPanelTop                = 1.;
  res->nglPanelInvsblTop          = -999.;
  res->nglPanelInvsblLeft         = -999.;
  res->nglPanelInvsblRight        = -999.;
  res->nglPanelInvsblBottom       = -999.;

/*
 * Special resources for a panel labelbar.
 */
  res->nglPanelLabelBar               = 0;  
  res->nglPanelLabelBarXF             = -999.;
  res->nglPanelLabelBarYF             = -999.;
  res->nglPanelLabelBarWidthF         = -999.;
  res->nglPanelLabelBarHeightF        = -999.;
  res->nglPanelLabelBarOrientation    = NhlHORIZONTAL;
  res->nglPanelLabelBarPerimOn        = 0;
  res->nglPanelLabelBarAlignment      = NhlINTERIOREDGES;
  res->nglPanelLabelBarFontHeightF    = -999.;
  res->nglPanelLabelBarOrthogonalPosF = -999.;
  res->nglPanelLabelBarParallelPosF   = -999.;
}

/*
 * This function maximizes and draws the plot, and advances the frame.
 */

void draw_and_frame(int wks, nglPlotId *plots, int nplots, int ispanel, 
                    nglRes *special_res)
{
  int i;

  if(special_res->nglMaximize) maximize_plot(wks, plots, nplots, 
                                             ispanel, special_res);
  if(special_res->nglDraw)  {
    for( i = 0; i < nplots; i++ ) {
      NhlDraw(*(plots[i].base));
    }
  }
  if(special_res->nglFrame) NhlFrame(wks);
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

int ngl_open_wks_wrap(const char *type, const char *name, int wk_rlist)
{
  int wks, len, tlen;
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
  else if(!strcmp(type,"ps")   || !strcmp(type,"PS") || 
          !strcmp(type,"eps")  || !strcmp(type,"EPS") || 
          !strcmp(type,"epsi") || !strcmp(type,"EPSI")) {
/*
 * Generate PS file name.
 */

    len      = strlen(name);
    tlen     = strlen(type);
    filename = (char *)calloc(len+tlen+2,sizeof(char));

    strncpy(filename,name,len);
    strncat(filename,".",1);
    strcat(filename,type);

/*
 * Create a PS workstation.
 */

    NhlRLSetString(wk_rlist,"wkPSFileName",filename);
    NhlCreate(&wks,type,NhlpsWorkstationClass,NhlDEFAULT_APP,wk_rlist);
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
    NhlPError(NhlWARNING,NhlEUNKNOWN,"spread_colors: Invalid workstation type, must be 'x11', 'ncgm', 'ps', or 'pdf'\n");
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

nglPlotId ngl_contour_wrap(int wks, void *data, const char *type, int ylen,
                           int xlen, int is_ycoord, void *ycoord, 
                           const char *ycoord_type,int is_xcoord, 
                           void *xcoord, const char *xcoord_type,
                           int is_missing, void *FillValue, int sf_rlist,
                           int cn_rlist, nglRes *special_res)
{
  nglPlotId plot;
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
 * If the contours are being filled, span the full colormap for the colors.
 */
  if(special_res->nglSpreadColors)  {
    spread_colors(wks, contour, special_res->nglSpreadColorStart,
                  special_res->nglSpreadColorEnd, "cnLevelCount", 
                  "cnFillColors",special_res->nglDebug);
  }

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->nglScale) scale_plot(contour);

/*
 * Initialize plot id structure.
 */
  initialize_ids(&plot);
  plot.sffield    = (int *)malloc(sizeof(int));
  plot.contour    = (int *)malloc(sizeof(int));
  plot.base       = plot.contour;
  *(plot.sffield) = field;
  *(plot.contour) = contour;
  plot.nsffield   = 1;
  plot.ncontour   = 1;
  plot.nbase      = plot.ncontour;

/*
 * Draw contour plot and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Return.
 */
  return(plot);
}


/*
 * This function uses the HLUs to create an XY plot.
 */

nglPlotId ngl_xy_wrap(int wks, void *x, void *y, const char *type_x,
                      const char *type_y, int ndims_x, int *dsizes_x,
                      int ndims_y, int *dsizes_y, 
                      int is_missing_x, int is_missing_y, 
                      void *FillValue_x, void *FillValue_y,
                      int ca_rlist, int xy_rlist, int xyd_rlist,
                      nglRes *special_res)
{
  int cafield, xy, grlist, num_dspec, *xyds;
  nglPlotId plot;


/*
 * Create a coord arrays object that will be used as the
 * dataset for the xy object.
 */

  cafield = coord_array(x, y, type_x, type_y, ndims_x, dsizes_x, 
                        ndims_y, dsizes_y, is_missing_x, is_missing_y, 
                        FillValue_x, FillValue_y, ca_rlist);
 
/*
 * Assign the data object.
 */

  NhlRLSetInteger(xy_rlist,"xyCoordData",cafield);

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

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->nglScale) scale_plot(xy);

/*
 * Initialize plot ids.
 */
  initialize_ids(&plot);
  plot.cafield    = (int *)malloc(sizeof(int));
  plot.xy         = (int *)malloc(sizeof(int));
  plot.base       = plot.xy;
  *(plot.cafield) = cafield;
  *(plot.xy)      = xy;
  plot.ncafield   = 1;
  plot.nxy        = 1;
  plot.nbase      = plot.nxy;

/*
 * Draw xy plot and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Return.
 */
  return(plot);
}


/*
 * This function uses the HLUs to create an XY plot.
 */

nglPlotId ngl_y_wrap(int wks, void *y, const char *type_y, int ndims_y, 
                     int *dsizes_y, int is_missing_y, void *FillValue_y,
                     int ca_rlist, int xy_rlist, int xyd_rlist,
                     nglRes *special_res)
{
  nglPlotId xy;

/*
 * Call ngl_xy_wrap, only using NULLs for the X values.
 */
  xy = ngl_xy_wrap(wks, NULL, y, NULL, type_y, 0, NULL,
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

nglPlotId ngl_vector_wrap(int wks, void *u, void *v, const char *type_u,
                          const char *type_v, int ylen, int xlen, 
                          int is_ycoord, void *ycoord, 
                          const char *type_ycoord, int is_xcoord, 
                          void *xcoord, const char *type_xcoord, 
                          int is_missing_u, int is_missing_v, 
                          void *FillValue_u, void *FillValue_v,
                          int vf_rlist, int vc_rlist, nglRes *special_res)
{
  int field, vector;
  nglPlotId plot;

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
 * If the vectors are being filled, span the full colormap for the colors.
 */
  if(special_res->nglSpreadColors)  {
    spread_colors(wks, vector, special_res->nglSpreadColorStart,
                  special_res->nglSpreadColorEnd, "vcLevelCount", 
                  "vcLevelColors",special_res->nglDebug);
  }

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->nglScale) scale_plot(vector);

/*
 * Initialize plot ids.
 */
  initialize_ids(&plot);
  plot.vffield    = (int *)malloc(sizeof(int));
  plot.vector     = (int *)malloc(sizeof(int));
  plot.base       = plot.vector;
  *(plot.vffield) = field;
  *(plot.vector)  = vector;
  plot.nvffield   = 1;
  plot.nvector    = 1;
  plot.nbase      = plot.nvector;

/*
 * Draw vector plot and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Return.
 */
  return(plot);

}


/*
 * This function uses the HLUs to create a streamline plot.
 */

nglPlotId ngl_streamline_wrap(int wks, void *u, void *v, const char *type_u,
                              const char *type_v, int ylen, int xlen, 
                              int is_ycoord, void *ycoord, 
                              const char *type_ycoord, int is_xcoord, 
                              void *xcoord, const char *type_xcoord, 
                              int is_missing_u, int is_missing_v, 
                              void *FillValue_u, void *FillValue_v, 
                              int vf_rlist, int st_rlist, 
                              nglRes *special_res)
{
  int field, streamline;
  nglPlotId plot;

/*
 * Create a vector field object that will be used as the
 * dataset for the streamline object.
 */

  field = vector_field(u, v, type_u, type_v, ylen, xlen, is_ycoord,
                       ycoord, type_ycoord, is_xcoord, xcoord,
                       type_xcoord, is_missing_u, is_missing_v,
                       FillValue_u, FillValue_v, vf_rlist);
 
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

  if(special_res->nglScale) scale_plot(streamline);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield       = (int *)malloc(sizeof(int));
  plot.streamline    = (int *)malloc(sizeof(int));
  plot.base          = plot.streamline;
  *(plot.vffield)    = field;
  *(plot.streamline) = streamline;
  plot.nvffield      = 1;
  plot.nstreamline   = 1;
  plot.nbase         = plot.nstreamline;

/*
 * Draw streamline plot and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Return.
 */
  return(plot);
}


/*
 * This function uses the HLUs to create a map plot.
 */

nglPlotId ngl_map_wrap(int wks, int mp_rlist, nglRes *special_res)
{
  int map;
  nglPlotId plot;

/*
 * Create plot.
 */

  NhlCreate(&map,"map",NhlmapPlotClass,wks,mp_rlist);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.map    = (int *)malloc(sizeof(int));
  plot.base   = plot.map;
  *(plot.map) = map;
  plot.nmap   = 1;
  plot.nbase  = plot.nmap;

/*
 * Draw map plot and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Return.
 */
  return(plot);
}



/*
 * This function uses the HLUs to overlay contours on a map.
 * It is the user's responsibility to set the sfXArray/sfYArray
 * (or equivalent) coordinate arrays to the approprate lat/lon values.
 */

nglPlotId ngl_contour_map_wrap(int wks, void *data, const char *type, 
                               int ylen, int xlen, int is_ycoord, 
                               void *ycoord, const char *ycoord_type,
                               int is_xcoord, void *xcoord, 
                               const char *xcoord_type, int is_missing, 
                               void *FillValue, int sf_rlist, int cn_rlist,
                               int mp_rlist, nglRes *special_res)
                         
{
  nglRes special_res2;
  nglPlotId contour, map, plot;

/*
 * Create contour plot. Be sure to copy over special resources.
 */
 
  special_res2                     = *special_res;
  special_res2.nglDraw             = 0;
  special_res2.nglFrame            = 0;
  special_res2.nglMaximize         = 0;

  contour = ngl_contour_wrap(wks, data, type, ylen, xlen,
                             is_ycoord, ycoord, ycoord_type,
                             is_xcoord, xcoord, xcoord_type,
                             is_missing, FillValue, sf_rlist, cn_rlist,
                             &special_res2);


/*
 * Create map plot.
 */
  map = ngl_map_wrap(wks, mp_rlist, &special_res2);

/*
 * Overlay contour plot on map plot.
 */
  NhlAddOverlay(*(map.base),*(contour.base),-1);

/*
 * Make tickmarks and axis labels the same size.
 */
  if(special_res->nglScale) scale_plot(*(contour.base));

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.sffield    = (int *)malloc(sizeof(int));
  plot.contour    = (int *)malloc(sizeof(int));
  plot.map        = (int *)malloc(sizeof(int));
  plot.base       = plot.map;
  *(plot.sffield) = *(contour.sffield);
  *(plot.contour) = *(contour.base);
  *(plot.map)     = *(map.base);
  plot.nsffield   = 1;
  plot.ncontour   = 1;
  plot.nmap       = 1;
  plot.nbase      = plot.nmap;

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Free up memory we don't need.
 */
  free(map.map);
  free(contour.contour);

/*
 * Return.
 */
  return(plot);
}

/*
 * This function uses the HLUs to create a vector plot over a map.
 */

nglPlotId ngl_vector_map_wrap(int wks, void *u, void *v, const char *type_u,
                              const char *type_v, int ylen, int xlen, 
                              int is_ycoord, void *ycoord, 
                              const char *type_ycoord, int is_xcoord, 
                              void *xcoord, const char *type_xcoord, 
                              int is_missing_u, int is_missing_v, 
                              void *FillValue_u, void *FillValue_v,
                              int vf_rlist, int vc_rlist, int mp_rlist,
                              nglRes *special_res)
{
  nglRes special_res2;
  nglPlotId vector, map, plot;

/*
 * Create vector plot. Be sure to copy over special resources.
 */

  special_res2                     = *special_res;
  special_res2.nglDraw             = 0;
  special_res2.nglFrame            = 0;
  special_res2.nglMaximize         = 0;

  vector = ngl_vector_wrap(wks, u, v, type_u, type_v, ylen, xlen, is_ycoord,
                           ycoord, type_ycoord, is_xcoord, xcoord, 
                           type_xcoord, is_missing_u, is_missing_v, 
                           FillValue_u, FillValue_v, vf_rlist, vc_rlist,
                           &special_res2);

/*
 * Create map plot.
 */
  map = ngl_map_wrap(wks, mp_rlist, &special_res2);

/*
 * Overlay vector plot on map plot.
 */
  NhlAddOverlay(*(map.base),*(vector.base),-1);

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->nglScale) scale_plot(*(vector.base));

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield    = (int *)malloc(sizeof(int));
  plot.vector     = (int *)malloc(sizeof(int));
  plot.map        = (int *)malloc(sizeof(int));
  plot.base       = plot.map;
  *(plot.vffield) = *(vector.vffield);
  *(plot.vector)  = *(vector.base);
  *(plot.map)     = *(map.base);
  plot.nvffield   = 1;
  plot.nvector    = 1;
  plot.nmap       = 1;
  plot.nbase      = plot.nmap;

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Free up memory we don't need.
 */
  free(map.map);
  free(vector.vector);

/*
 * Return.
 */
  return(plot);
}

/*
 * This function uses the HLUs to create a streamline plot over a map.
 */

nglPlotId ngl_streamline_map_wrap(int wks, void *u, void *v, 
                                  const char *type_u, const char *type_v, 
                                  int ylen, int xlen, int is_ycoord, 
                                  void *ycoord, const char *type_ycoord, 
                                  int is_xcoord, void *xcoord, 
                                  const char *type_xcoord, int is_missing_u,
                                  int is_missing_v, void *FillValue_u, 
                                  void *FillValue_v, int vf_rlist, 
                                  int vc_rlist, int mp_rlist,
                                  nglRes *special_res)
{
  nglRes special_res2;
  nglPlotId streamline, map, plot;

/*
 * Create streamline plot.
 */

  special_res2             = *special_res;
  special_res2.nglDraw     = 0;
  special_res2.nglFrame    = 0;
  special_res2.nglMaximize = 0;

  streamline = ngl_streamline_wrap(wks, u, v, type_u, type_v, ylen, xlen, 
                                   is_ycoord, ycoord, type_ycoord, 
                                   is_xcoord, xcoord, type_xcoord, 
                                   is_missing_u, is_missing_v, FillValue_u, 
                                   FillValue_v, vf_rlist, vc_rlist,
                                   &special_res2);

/*
 * Create map plot.
 */
  map = ngl_map_wrap(wks, mp_rlist, &special_res2);

/*
 * Overlay streamline plot on map plot.
 */
  NhlAddOverlay(*(map.base),*(streamline.base),-1);

/*
 * Make tickmarks and axis labels the same size.
 */
  if(special_res->nglScale) scale_plot(*(streamline.base));

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield       = (int *)malloc(sizeof(int));
  plot.streamline    = (int *)malloc(sizeof(int));
  plot.map           = (int *)malloc(sizeof(int));
  plot.base          = plot.map;
  *(plot.vffield)    = *(streamline.vffield);
  *(plot.streamline) = *(streamline.base);
  *(plot.map)        = *(map.base);

  plot.nvffield    = 1;
  plot.nstreamline = 1;
  plot.nmap        = 1;
  plot.nbase       = plot.nmap;

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Free up memory we don't need.
 */
  free(streamline.streamline);
  free(map.map);

/*
 * Return.
 */
  return(plot);
}


/*
 * This function uses the HLUs to create a vector plot colored by
 * a scalar field.
 */

nglPlotId ngl_vector_scalar_wrap(int wks, void *u, void *v, void *t, 
                                 const char *type_u, const char *type_v, 
                                 const char *type_t, int ylen, int xlen, 
                                 int is_ycoord, void *ycoord, 
                                 const char *type_ycoord, int is_xcoord, 
                                 void *xcoord, const char *type_xcoord, 
                                 int is_missing_u, int is_missing_v, 
                                 int is_missing_t, void *FillValue_u, 
                                 void *FillValue_v, void *FillValue_t,
                                 int vf_rlist, int sf_rlist, int vc_rlist, 
                                 nglRes *special_res)
{
  int vffield, sffield, vector;
  nglPlotId plot;

/*
 * Create vector and scalar field objects that will be used as the
 * datasets for the vector object.
 */

  vffield = vector_field(u, v, type_u, type_v, ylen, xlen, 
                         is_ycoord, ycoord, type_ycoord, 
                         is_xcoord, xcoord, type_xcoord, 
                         is_missing_u, is_missing_v, 
                         FillValue_u, FillValue_v, vf_rlist);
  
  sffield = scalar_field(t, type_t, ylen, xlen, is_ycoord, ycoord, 
                         type_ycoord, is_xcoord, xcoord, type_xcoord, 
                         is_missing_t, FillValue_t, sf_rlist);

/*
 * Assign the data objects and create vector object.
 */

  NhlRLSetInteger(vc_rlist, "vcVectorFieldData",    vffield);
  NhlRLSetInteger(vc_rlist, "vcScalarFieldData",    sffield);
  NhlRLSetString (vc_rlist, "vcUseScalarArray",     "True");
  NhlRLSetString (vc_rlist, "vcMonoLineArrowColor", "False");

  NhlCreate(&vector,"vector",NhlvectorPlotClass,wks,vc_rlist);

/*
 * If the vectors are being filled, span the full colormap for the colors.
 */
  if(special_res->nglSpreadColors)  {
    spread_colors(wks, vector, special_res->nglSpreadColorStart,
                  special_res->nglSpreadColorEnd, "vcLevelCount", 
                  "vcLevelColors",special_res->nglDebug);
  }

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->nglScale) scale_plot(vector);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield    = (int *)malloc(sizeof(int));
  plot.sffield    = (int *)malloc(sizeof(int));
  plot.vector     = (int *)malloc(sizeof(int));
  plot.base       = plot.vector;
  *(plot.vffield) = vffield;
  *(plot.sffield) = sffield;
  *(plot.vector)  = vector;
  plot.nvffield = 1;
  plot.nsffield = 1;
  plot.nvector  = 1;
  plot.nbase    = plot.nvector;

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Return.
 */
  return(plot);
}

/*
 * This function uses the HLUs to create a vector plot colored by
 * a scalar field overlaid on a map.
 */

nglPlotId ngl_vector_scalar_map_wrap(int wks, void *u, void *v, void *t, 
                                     const char *type_u, const char *type_v, 
                                     const char *type_t, int ylen, int xlen, 
                                     int is_ycoord, void *ycoord, 
                                     const char *type_ycoord, int is_xcoord, 
                                     void *xcoord, const char *type_xcoord, 
                                     int is_missing_u, int is_missing_v, 
                                     int is_missing_t, void *FillValue_u, 
                                     void *FillValue_v, void *FillValue_t,
                                     int vf_rlist, int sf_rlist,
                                     int vc_rlist, int mp_rlist,
                                     nglRes *special_res)
{
  nglRes special_res2;
  nglPlotId plot, vector, map;

/*
 * Create vector plot.
 */

  special_res2             = *special_res;
  special_res2.nglDraw     = 0;
  special_res2.nglFrame    = 0;
  special_res2.nglMaximize = 0;

  vector = ngl_vector_scalar_wrap(wks, u, v, t, type_u, type_v, type_t,
                                  ylen, xlen, is_ycoord, ycoord, 
                                  type_ycoord, is_xcoord, xcoord, 
                                  type_xcoord, is_missing_u, is_missing_v, 
                                  is_missing_t, FillValue_u, FillValue_v, 
                                  FillValue_t, vf_rlist, sf_rlist, vc_rlist, 
                                  &special_res2);

/*
 * Create map plot.
 */
  map = ngl_map_wrap(wks, mp_rlist, &special_res2);

/*
 * Overlay vector plot on map plot.
 */
  NhlAddOverlay(*(map.base),*(vector.base),-1);

/*
 * Make tickmarks and axis labels the same size.
 */

  if(special_res->nglScale) scale_plot(*(vector.base));

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield    = (int *)malloc(sizeof(int));
  plot.sffield    = (int *)malloc(sizeof(int));
  plot.vector     = (int *)malloc(sizeof(int));
  plot.map        = (int *)malloc(sizeof(int));
  plot.base       = plot.map;
  *(plot.vffield) = *(vector.vffield);
  *(plot.sffield) = *(vector.sffield);
  *(plot.vector)  = *(vector.base);
  *(plot.map)     = *(map.base);

  plot.nvffield = 1;
  plot.nsffield = 1;
  plot.nvector  = 1;
  plot.nmap     = 1;
  plot.nbase    = plot.nmap;

/*
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Free up memory we don't need.
 */
  free(vector.vector);
  free(map.map);

/*
 * Return.
 */
  return(plot);
}


nglPlotId ngl_text_ndc_wrap(int wks, char* string, void *x, void *y,
                            const char *type_x, const char *type_y,
                            int tx_rlist, nglRes *special_res)
{
  int text, length[1];
  nglPlotId plot;

  length[0] = 1;

  set_resource("txPosXF", tx_rlist, x, type_x, 1, &length[0] );
  set_resource("txPosYF", tx_rlist, y, type_y, 1, &length[0] );


  NhlRLSetString(tx_rlist, "txString", string);
  NhlCreate(&text,"text",NhltextItemClass,wks,tx_rlist);

/*
 * Initialize plot ids.
 */
  initialize_ids(&plot);
  plot.text    = (int *)malloc(sizeof(int));
  plot.base    = plot.text;
  *(plot.text) = text;
  plot.ntext   = 1;
  plot.nbase   = 1;

/*
 * Draw text.
 */
  draw_and_frame(wks, &plot, 1, 0, special_res);

/*
 * Return.
 */
  return(plot);
}


nglPlotId ngl_text_wrap(int wks, nglPlotId *plot, char* string, void *x, 
                        void *y, const char *type_x, const char *type_y,
                        int tx_rlist, nglRes *special_res)
{
  float *xf, *yf, xndc, yndc, oor = 0.;
  int status;
  nglPlotId text;

/*
 * Convert x and y to float, since NhlDatatoNDC routine only accepts 
 * floats.
 */
  xf = coerce_to_float(x,type_x,1);
  yf = coerce_to_float(y,type_y,1);

/*
 * Convert from plot's data space to NDC space.
 */

  (void)NhlDataToNDC(*(plot->base),xf,yf,1,&xndc,&yndc,NULL,NULL,&status,&oor);

  if(special_res->nglDebug) {
    printf("ngl_text: string = %s x = %g y = %g xndc = %g yndc = %g\n", 
           string, *xf, *yf, xndc, yndc);
  }

  text = ngl_text_ndc_wrap(wks, string, &xndc, &yndc, "float", "float",
                           tx_rlist, special_res);

/*
 * Return.
 */
  return(text);
}

/*
 * Routine for drawing any kind of primitive in NDC or data space.
 */
void ngl_poly_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                   const char *type_x, const char *type_y, int len,
                   int is_missing_x, int is_missing_y, void *FillValue_x, 
                   void *FillValue_y, NhlPolyType polytype, int gs_rlist,
                   nglRes *special_res)
{
  int i, gsid, newlen, *indices, ibeg, iend, nlines, color;
  int srlist, grlist;
  float *xf, *yf, *xfnew, *yfnew, *xfmsg, *yfmsg;

/*
 * Determine if we need to convert x and/or y (and their missing values)
 * to float.
 */
  xf  = coerce_to_float(x,type_x,len);
  yf  = coerce_to_float(y,type_y,len);
  if(is_missing_x) xfmsg = coerce_to_float(FillValue_x,type_x,1);
  if(is_missing_y) yfmsg = coerce_to_float(FillValue_y,type_y,1);

/*
 * Create graphic style object on which to draw primitives.
 */
  gsid = create_graphicstyle_object(wks);

/*
 * Set the graphic style resources, if any.
 */

  NhlSetValues(gsid,gs_rlist);

/*
 * Remove missing values, if any. Don't do this for polylines, because for
 * polylines, we need to keep track of location of missing values.
 */
  if(polytype == NhlPOLYGON || polytype == NhlPOLYMARKER) {
    collapse_nomsg_xy(xf,yf,&xfnew,&yfnew,len,is_missing_x,is_missing_y,
                      xfmsg,yfmsg,&newlen);
  }

/*
 * Draw the appropriate primitive.
 */
  if(special_res->nglDraw) {
    switch(polytype) {
    case NhlPOLYMARKER:
      if(plot->base == NULL) {
        NhlNDCPolymarker(wks,gsid,&xfnew[0],&yfnew[0],newlen);
      }
      else {
        NhlDataPolymarker(*(plot->base),gsid,&xfnew[0],&yfnew[0],newlen);
      }
      break;

    case NhlPOLYGON:
      if(plot->base == NULL) {
        NhlNDCPolygon(wks,gsid,&xfnew[0],&yfnew[0],newlen);
      }
      else {
        NhlDataPolygon(*(plot->base),gsid,&xfnew[0],&yfnew[0],newlen);
      }
      break;

    case NhlPOLYLINE:
      indices = get_non_missing_pairs(xf, yf, is_missing_x, is_missing_y,
                                      xfmsg, yfmsg, len, &nlines);
      for(i = 0; i < nlines; i++) {
/*
 * Get the begin and end indices of the non-missing section of points.
 */
        ibeg = indices[i*2];
        iend = indices[i*2+1];
/*
 * If ibeg = iend, then this means we just have one point, and so
 * we want to put down a marker instead of a line.
 */
        if(iend == ibeg) {
          srlist = NhlRLCreate(NhlSETRL);
          grlist = NhlRLCreate(NhlGETRL);
          NhlRLClear(grlist);
          NhlRLGetInteger(grlist,"gsLineColor",&color);
          NhlGetValues(gsid,grlist);
          
          NhlRLClear(srlist);
          NhlRLSetInteger(srlist,"gsMarkerColor",color);
          NhlSetValues(gsid,srlist);
          
          if(plot->base == NULL) {
            NhlNDCPolymarker(wks,gsid,&xf[ibeg],&yf[ibeg],1);
          }
          else {
            NhlDataPolymarker(*(plot->base),gsid,&xf[ibeg],&yf[ibeg],1);
          }
        }
        else {
          newlen = iend - ibeg + 1;
/*
 * We have more than one point, so create a polyline.
 */
          if(plot->base == NULL) {
            NhlNDCPolyline(wks,gsid,&xf[ibeg],&yf[ibeg],newlen);
          } 
          else {
            NhlDataPolyline(*(plot->base),gsid,&xf[ibeg],&yf[ibeg],newlen);
          }
        }
      }
      free(indices);
      break;
    }
  }

/*
 * Free up memory.
 */
  if(strcmp(type_x,"float")) free(xf);
  if(strcmp(type_y,"float")) free(yf);

  if(polytype == NhlPOLYGON || polytype == NhlPOLYMARKER) {
    free(xfnew);
    free(yfnew);
  }
    
  if(special_res->nglFrame) NhlFrame(wks);

}


/*
 * Routine for adding any kind of primitive (in data space only).
 * The difference between adding a primitive, and just drawing a 
 * primitive, is that when you add a primitive to a plot, then it will
 * only get drawn when you draw the plot, and it will also be scaled
 * if you scale the plot. 
 */
nglPlotId ngl_add_poly_wrap(int wks, nglPlotId *plot, void *x, void *y,
                            const char *type_x, const char *type_y, int len, 
                            int is_missing_x, int is_missing_y, 
                            void *FillValue_x, void *FillValue_y,
                            NhlPolyType polytype, int gs_rlist, 
                            nglRes *special_res)
{
  int *primitive_object, gsid, pr_rlist, grlist;
  int i, newlen, *indices, nlines, npoly, ibeg, iend, npts;
  float *xf, *yf, *xfnew, *yfnew, *xfmsg, *yfmsg;
  char *astring;
  nglPlotId poly;

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
 * missing values, and plot. Also, if markers or gons, then only
 * one primitive object is created. If lines, then one primitive
 * object is created for each section of non-missing points.
 */

  if(polytype != NhlPOLYLINE) {
    npoly = 1;
/*
 * Remove missing values, if any.
 */
    collapse_nomsg_xy(xf, yf, &xfnew, &yfnew, len, is_missing_x,
                      is_missing_y, xfmsg, yfmsg, &newlen);
/*
 * Set some primitive object resources.  Namely, the location of
 * the X/Y points, and the type of primitive (polymarker or polygon
 * in this case).
 */
    NhlRLSetFloatArray(pr_rlist,"prXArray",       &xfnew[0], newlen);
    NhlRLSetFloatArray(pr_rlist,"prYArray",       &yfnew[0], newlen);
    NhlRLSetInteger   (pr_rlist,"prPolyType",     polytype);
    NhlRLSetInteger   (pr_rlist,"prGraphicStyle", gsid);

/*
 * Allocate a variable to hold the primitive object, and create it.
 */

    primitive_object = (int*)malloc(sizeof(int));
    NhlCreate(primitive_object,"Primitive",NhlprimitiveClass,wks,pr_rlist);

/*
 * Attach primitive object to the plot.
 */

    NhlAddPrimitive(*(plot->base),*primitive_object,-1);
/*
 * Free up memory.
 */
    free(xfnew);
    free(yfnew);
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

    npoly = nlines;
    if(nlines > 0) {
      primitive_object = (int*)malloc(nlines*sizeof(int));
      astring          = (char*)malloc((strlen(polylinestr)+8)*sizeof(char));
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

        NhlAddPrimitive(*(plot->base),primitive_object[i],-1);
      }
    }
    else {
/*
 * Create a NULL primitive object.
 */
      primitive_object = (int*)malloc(sizeof(int));
    }     
    free(indices);
  }

/*
 * Free up memory.
 */
  if(strcmp(type_x,"float")) free(xf);
  if(strcmp(type_y,"float")) free(yf);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&poly);
  poly.primitive  = primitive_object;
  poly.base       = poly.primitive;
  poly.nprimitive = npoly;
  poly.nbase      = npoly;

/*
 * Return.
 */
  return(poly);
}

/*
 * Routine for drawing markers in NDC space.
 */
void ngl_polymarker_ndc_wrap(int wks, void *x, void *y, const char *type_x, 
                             const char *type_y,  int len,
                             int is_missing_x, int is_missing_y, 
                             void *FillValue_x, void *FillValue_y, 
                             int gs_rlist, nglRes *special_res)
{
  nglPlotId plot;

  initialize_ids(&plot);
  ngl_poly_wrap(wks, &plot, x, y, type_x, type_y, len, is_missing_x, 
                is_missing_y, FillValue_x, FillValue_y, NhlPOLYMARKER, 
                gs_rlist,special_res);
}


/*
 * Routine for drawing lines in NDC space.
 */
void ngl_polyline_ndc_wrap(int wks, void *x, void *y, const char *type_x,
                           const char *type_y, int len,
                           int is_missing_x, int is_missing_y, 
                           void *FillValue_x, void *FillValue_y, 
                           int gs_rlist, nglRes *special_res)
{
  nglPlotId plot;

  initialize_ids(&plot);
  ngl_poly_wrap(wks, &plot, x, y, type_x, type_y, len, is_missing_x, 
                is_missing_y, FillValue_x, FillValue_y, NhlPOLYLINE, 
                gs_rlist, special_res);
}


/*
 * Routine for drawing polygons in NDC space.
 */
void ngl_polygon_ndc_wrap(int wks, void *x, void *y, const char *type_x, 
                          const char *type_y, int len,
                          int is_missing_x, int is_missing_y, 
                          void *FillValue_x, void *FillValue_y, 
                          int gs_rlist, nglRes *special_res)
{
  nglPlotId plot;

  initialize_ids(&plot);
  ngl_poly_wrap(wks, &plot, x, y, type_x, type_y, len, is_missing_x,
                is_missing_y, FillValue_x, FillValue_y, NhlPOLYGON, 
                gs_rlist, special_res);
}

/*
 * Routine for drawing markers in data space.
 */
void ngl_polymarker_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                         const char *type_x, const char *type_y, int len,
                         int is_missing_x, int is_missing_y, 
                         void *FillValue_x, void *FillValue_y, 
                         int gs_rlist, nglRes *special_res)
{
  ngl_poly_wrap(wks,plot,x,y,type_x,type_y,len,is_missing_x,is_missing_y,
                FillValue_x,FillValue_y,NhlPOLYMARKER,gs_rlist,special_res);
}


/*
 * Routine for drawing lines in data space.
 */
void ngl_polyline_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                       const char *type_x, const char *type_y, int len, 
                       int is_missing_x, int is_missing_y, 
                       void *FillValue_x, void *FillValue_y, 
                       int gs_rlist, nglRes *special_res)
{
  ngl_poly_wrap(wks,plot,x,y,type_x,type_y,len,is_missing_x,is_missing_y,
                FillValue_x,FillValue_y,NhlPOLYLINE,gs_rlist,special_res);
}

/*
 * Routine for drawing polygons in data space.
 */
void ngl_polygon_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                      const char *type_x, const char *type_y, int len, 
                      int is_missing_x, int is_missing_y, 
                      void *FillValue_x, void *FillValue_y, 
                      int gs_rlist, nglRes *special_res)
{
  ngl_poly_wrap(wks, plot, x, y, type_x, type_y, len, is_missing_x, 
                is_missing_y, FillValue_x, FillValue_y, NhlPOLYGON, 
                gs_rlist, special_res);
}

/*
 * Routine for adding polylines to a plot (in the plot's data space).
 */
nglPlotId ngl_add_polyline_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                                const char *type_x, const char *type_y,
                                int len, int is_missing_x, int is_missing_y, 
                                void *FillValue_x, void *FillValue_y, 
                                int gs_rlist, nglRes *special_res)
{
  nglPlotId poly;

  poly = ngl_add_poly_wrap(wks, plot, x, y, type_x, type_y, len, 
                           is_missing_x, is_missing_y, FillValue_x, 
                           FillValue_y, NhlPOLYLINE, gs_rlist, 
                           special_res);
/*
 * Return.
 */
  return(poly);

}


/*
 * Routine for adding polymarkers to a plot (in the plot's data space).
 */
nglPlotId ngl_add_polymarker_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                                  const char *type_x, const char *type_y,
                                  int len, int is_missing_x,
                                  int is_missing_y, void *FillValue_x,
                                  void *FillValue_y, int gs_rlist, 
                                  nglRes *special_res)
{
  nglPlotId poly;

  poly = ngl_add_poly_wrap(wks, plot, x, y, type_x, type_y, len, 
                           is_missing_x, is_missing_y, FillValue_x, 
                           FillValue_y, NhlPOLYMARKER, gs_rlist, 
                           special_res);
/*
 * Return.
 */
  return(poly);

}


/*
 * Routine for adding polygons to a plot (in the plot's data space).
 */
nglPlotId ngl_add_polygon_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                               const char *type_x, const char *type_y, 
                               int len, int is_missing_x, int is_missing_y, 
                               void *FillValue_x, void *FillValue_y, 
                               int gs_rlist, nglRes *special_res)
{
  nglPlotId poly;

  poly = ngl_add_poly_wrap(wks, plot, x, y, type_x, type_y, len, 
                           is_missing_x, is_missing_y, FillValue_x, 
                           FillValue_y, NhlPOLYGON, gs_rlist, 
                           special_res);
/*
 * Return.
 */
  return(poly);

}


/*
 * Routine for adding text (in data space only) to a plot.
 *
 * The difference between adding text and just drawing text
 * is that when you add a text string to a plot, it will
 * only get drawn when you draw the plot. It will also be scaled
 * appropriately if you scale the plot. 
 */
nglPlotId ngl_add_text_wrap(int wks, nglPlotId *plot, char *string, void *x,
                            void *y, const char *type_x, const char *type_y,
                            int tx_rlist, int am_rlist, nglRes *special_res)
{
  int i, srlist, grlist, text, just;
  int *anno_views, *anno_mgrs, *new_anno_views, num_annos;
  float *xf, *yf;
  nglPlotId annos;

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
  NhlGetValues(*(plot->base),grlist);

/*
 * Make sure the new text string is first in the list of annotations,
 * if any.
 */
  if(num_annos > 0) {
    new_anno_views = (int *)malloc((1+num_annos)*sizeof(int));
    new_anno_views[0] = text;
    for( i = 1; i <= num_annos; i++ ) {
      new_anno_views[i] = anno_views[i-1];
    }
    num_annos++;
  }
  else {
    new_anno_views  = (int *)malloc(sizeof(int));
    *new_anno_views = text;
    num_annos = 1;
  }

/*
 * Set the old and new annotations, with the new text being first.
 */

  srlist = NhlRLCreate(NhlSETRL);
  NhlRLClear(srlist);
  NhlRLSetIntegerArray(srlist,"pmAnnoViews",new_anno_views,num_annos);
  NhlSetValues(*(plot->base),srlist);

/*
 * Don't need new_anno_views anymore.
 */
  free(new_anno_views);

/*
 * Retrieve the ids of the AnnoManager objects created by the
 * PlotManager.
 */

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  NhlRLGetIntegerArray(grlist,"pmAnnoManagers",&anno_mgrs,&num_annos);
  NhlGetValues(*(plot->base),grlist);

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
 * Set up plot id structure to return.
 */
  initialize_ids(&annos);
  annos.text    = (int *)malloc(sizeof(int));
  *(annos.text) = anno_mgrs[0];
  annos.base    = annos.text;
  annos.ntext   = 1;
  annos.nbase   = annos.ntext;

/*
 * Return.
 */
  return(annos);
}


/*
 * Routine for drawing the current color map. This is mostly for
 * debugging purposes.
 */
void ngl_draw_colormap_wrap(int wks)
{
  int i, j, k, ii, jj, ibox, nrows, ncols, ncolors, maxcols, ntotal, offset;
  int grlist, srlist, txrlist, lnrlist, gnrlist;
  int reset_colormap, cmap_ndims, *cmap_dimsizes;
  float width, height, *xpos, *ypos, xbox[5], ybox[4], xbox2[5], ybox2[5];
  float txpos, typos;
  float *cmap, *cmapnew, font_height, font_space;
  char tmpstr[5];
  nglRes special_res2;

/*
 * Initialize special_res2.
 */
  special_res2.nglDraw     = 1;
  special_res2.nglFrame    = 0;
  special_res2.nglMaximize = 0;
  special_res2.nglDebug    = 0;

  nrows   = 16;                   /* # of rows of colors per page. */
  maxcols = 256;                  /* max # of colors per color table. */

/*
 * Set up generic lists for retrieving and setting resources.
 */
  grlist = NhlRLCreate(NhlGETRL);
  srlist = NhlRLCreate(NhlSETRL);

/*
 * Get # of colors in color map.
 */

  NhlRLClear(grlist);
  NhlRLGetInteger(grlist,"wkColorMapLen",&ncolors);
  NhlGetValues(wks,grlist);

/*
 * Figure out ncols such that the columns will span across the page.
 * Or, just set ncols to 16, which is big enough to cover the largest
 * possible color map.
 */
  ncols = ncolors/nrows;
  if((ncols*nrows) < ncolors) {
    ncols++;
  }

  ntotal = nrows * ncols;        /* # of colors per page. */

/*
 * If the number of colors in our color map is less than the allowed
 * maximum, then this gives us room to add a white background and/or a
 * black foreground.
 */
  reset_colormap = 0;
  if(ncolors < maxcols) {
    reset_colormap = 1;
/*
 * Get current color map.
 */
    NhlRLClear(grlist);
    NhlRLGetMDFloatArray(grlist,"wkColorMap",&cmap,&cmap_ndims,
                         &cmap_dimsizes);
    NhlGetValues(wks,grlist);

/*
 * If we haven't used the full colormap, then we can add 1 or 2 more
 * colors to 1) force the background to be white (it looks better this
 * way), and to 2) force the foreground to be black (for text). 
 */
    if(ncolors < (maxcols-1)) {
      offset = 2;
      cmapnew = (float *)malloc(3*(ncolors+2)*sizeof(float));
      cmapnew[0] = cmapnew[1] = cmapnew[2] = 1.;   /* white bkgrnd */
      cmapnew[3] = cmapnew[4] = cmapnew[5] = 0.;   /* black frgrnd */
      cmap_dimsizes[0] = ncolors+2;
      memcpy((void *)&cmapnew[6],(const void *)cmap,ncolors*3*sizeof(float));
    }
    else {
      offset = 1;
      cmapnew = (float *)malloc(3*(ncolors+1)*sizeof(float));
      cmapnew[0] = cmapnew[1] = cmapnew[2] = 1.;   /* white bkgrnd */
      cmap_dimsizes[0] = ncolors+1;
      memcpy((void *)&cmapnew[3],(const void *)cmap,ncolors*3*sizeof(float));
    }

/*
 * Set new color map in the workstation.
 */
    NhlRLClear(srlist);
    NhlRLSetMDFloatArray(srlist,"wkColorMap",&cmapnew[0],2,cmap_dimsizes);
    NhlSetValues(wks,srlist);
    free(cmapnew);
  }
  else {
/* 
 * The full 256 colors are being used, so we can't add anymore colors. 
 */
    offset = 0;
  }

/*
 * Calculate width/height of each color box, and the X and Y positions
 * of text and box in the view port.
 */
  width  = 1./ncols;
  height = 1./nrows;
  xpos   = fspan(0.,1.-width,ncols);
  ypos   = fspan(1.-height,0.,nrows);

/*
 * Box coordinates.
 */
  xbox[0] = xbox[3] = xbox[4] = 0.;
  xbox[1] = xbox[2] = width;
  ybox[0] = ybox[1] = ybox[4] = 0.;
  ybox[2] = ybox[3] = height;

/*
 * Values for text placement and size. 
 */
  font_height = 0.015;
  font_space  = font_height/2.;

/*
 * Initialize some resource lists for text, polygons, and polylines.
 */
  txrlist = NhlRLCreate(NhlSETRL);
  lnrlist = NhlRLCreate(NhlSETRL);
  gnrlist = NhlRLCreate(NhlSETRL);

/*
 * Set some text resources. 
 */
  NhlRLClear(txrlist);
  NhlRLSetFloat (txrlist, "txFontHeightF",         font_height);
  NhlRLSetString(txrlist, "txFont",                "helvetica-bold");
  NhlRLSetString(txrlist, "txJust",                "BottomLeft");
  NhlRLSetString(txrlist, "txPerimOn",             "True");
  NhlRLSetString(txrlist, "txPerimColor",          "black");
  NhlRLSetString(txrlist, "txFontColor",           "black");
  NhlRLSetString(txrlist, "txBackgroundFillColor", "white");

/*
 * Set some line resources for outlining the color boxes.
 **/
  NhlRLClear(lnrlist);
  NhlRLSetString(lnrlist,"gsLineColor","black");

/*
 * Clear the resource lines for the polygon.  The resources will
 * be set inside the loop below.
 */
  NhlRLClear(gnrlist);

/*
 * ntotal colors per page.
 */
  for(k = 1; k <= ncolors; k += ntotal) {

/*
 * Loop through rows.
 */
    jj = 0;
    for(j = k; j <= min(k+ntotal-1,ncolors); j += nrows) {

/*
 * Initialize row values for text and box positions.
 */
      txpos = font_space + xpos[jj];
      for( ibox = 0; ibox <= 4; ibox++) {
        xbox2[ibox] = xbox[ibox] + xpos[jj];
      }

/*
 * Loop through columns.
 */
      ii = 0;
      for(i = j; i <= min(j+nrows-1,ncolors); i++) {

/*
 * Initialize row values for text and box positions.
 */
        typos = font_space + ypos[ii];
        for( ibox = 0; ibox <= 4; ibox++) {
          ybox2[ibox] = ybox[ibox] + ypos[ii];
        }

/*
 * Set the color for the box and draw box it.
 */
        NhlRLSetInteger(gnrlist,"gsFillColor", offset + (i-1));
                                                                    
        ngl_polygon_ndc_wrap(wks, xbox2, ybox2, "float","float", 5,
                             0, 0, NULL, NULL, gnrlist, &special_res2);
/*
 * Outline box in black.
 */
        ngl_polyline_ndc_wrap(wks, xbox2, ybox2, "float","float", 5,
                              0, 0, NULL, NULL, lnrlist, &special_res2);
/*
 * Draw text string indicating color index value.
 */
        sprintf(tmpstr,"%d",i-1);
        ngl_text_ndc_wrap(wks, tmpstr, &txpos, &typos, "float", "float",
                          txrlist, &special_res2);
        ii++;
      }
      jj++; 
    }
/*
 * Advance the frame.
 */
    NhlFrame(wks);
  }

/*
 * Put the original color map back.
 */
  if(reset_colormap) {
    cmap_dimsizes[0] = ncolors;
    NhlRLClear(srlist);
    NhlRLSetMDFloatArray(srlist,"wkColorMap",&cmap[0],2,cmap_dimsizes);
    NhlSetValues(wks,srlist);
    free(cmap);
  }
}

/*
 * Routine for paneling same-sized plots.
 */
void ngl_panel_wrap(int wks, nglPlotId *plots, int nplots_orig, int *dims, 
                    int ndims, int lb_rlist, nglRes *special_res)
{
  int i, nplots, npanels, is_row_spec, nrows, ncols, draw_boxes = 0;
  int num_plots_left, nplot, nplot4, nr, nc, new_ncols, nnewplots;
  nglPlotId *newplots, pplot;
  int *row_spec, all_ismissing, first_time;
  int nvalid_plot, nvalid_plots, valid_plot;
  int panel_save, panel_debug, panel_center;
  int panel_labelbar, main_string_on, is_figure_strings;
  int *colors, *patterns;
  float *scales, *levels;
  int fill_on, glyph_style, fill_arrows_on, mono_line_color, mono_fill_arrow;
  int ncolors, nlevels, npatterns, nscales;
  int mono_fill_pat, mono_fill_scl, mono_fill_col;
  int lb_orient, lb_perim_on, lb_alignment, labelbar_object;
  float labelbar_width, labelbar_height;
  float lb_fh, lb_x, lb_y, lb_w, lb_h, tmp_range;
  char plot_type[10];
  int maxbb, calldraw, callframe;
  int lft_pnl, rgt_pnl, bot_pnl, top_pnl;
  float font_height;
  float x_lft, x_rgt, y_bot, y_top;
  float xlft, xrgt, xbot, xtop;
  float xsp, ysp, xwsp_perc, ywsp_perc, xwsp, ywsp;
  float vpx, vpy, vpw, vph, dxl, dxr, dyt, dyb;
  float *old_vp, *xpos, *ypos, max_rgt, max_top, max_lft, max_bot;
  float top, bottom, left, right;
  float newtop, newbot, newrgt, newlft;
  float plot_width, plot_height, total_width, total_height;
  float new_plot_width, new_plot_height, new_total_width, new_total_height;
  float min_xpos, max_xpos, min_ypos, max_ypos;
  float scaled_width, scaled_height, xrange, yrange;
  float scale, row_scale, col_scale, max_width, max_height;
  int grlist, srlist;
  NhlBoundingBox bb, *newbb;

/*
 * Resource lists for getting and retrieving resources.
 */

  grlist = NhlRLCreate(NhlGETRL);
  srlist = NhlRLCreate(NhlSETRL);

/*
 * First check if paneling is to be specified by (#rows x #columns) or
 * by #columns per row.  The default is rows x columns, unless 
 * resource nglPanelRowSpec is set to True
 */
  is_row_spec = special_res->nglPanelRowSpec;

  if(is_row_spec) {
    row_spec = dims;
    npanels  = 0;
    nrows    = ndims;
    ncols    = imax_array(row_spec,ndims);
    
    for(i = 0; i <= nrows-1; i++) {
      if(row_spec[i] < 0) {
        NhlPError(NhlFATAL,NhlEUNKNOWN,"ngl_panel: you have specified a negative value for the number of plots in a row.\n");
        return;
      }
      npanels += row_spec[i];
    }
  }
  else {
    if(ndims != 2) {
      NhlPError(NhlFATAL,NhlEUNKNOWN,"ngl_panel: for the third argument of ngl_panel, you must either specify # rows by # columns or set the nglPanelRowSpec resource to True and set the number of plots per row.\n");
      return;
    }
    nrows    = dims[0];
    ncols    = dims[1];
    npanels  = nrows * ncols;
    row_spec = (int *)malloc(nrows * sizeof(int));
    for(i = 0; i <= nrows-1; i++) row_spec[i] = ncols;
  }
  
  if(nplots_orig > npanels) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"ngl_panel: you have more plots than you have panels.\nOnly %d plots will be drawn.\n", npanels);
    nplots = npanels;
  }
  else {
    nplots = nplots_orig;
  }

/*
 * Check for special resources.
 */ 
  panel_save     = special_res->nglPanelSave;
  panel_debug    = special_res->nglDebug;
  panel_center   = special_res->nglPanelCenter;
  main_string_on = is_figure_strings = 0;
  panel_labelbar = special_res->nglPanelLabelBar;
  calldraw       = special_res->nglDraw;
  callframe      = special_res->nglFrame;
  xwsp_perc      = special_res->nglPanelXWhiteSpacePercent;
  ywsp_perc      = special_res->nglPanelYWhiteSpacePercent;
  
/*
 * Check if these four have been changed from their default values.
 */
  x_lft = special_res->nglPanelLeft;
  x_rgt = special_res->nglPanelRight;
  y_bot = special_res->nglPanelBottom;
  y_top = special_res->nglPanelTop;

  if(x_lft != 0.) {
    lft_pnl = 1;
  }
  else {
    lft_pnl = 0;
  }

  if(x_rgt != 1.) {
    rgt_pnl = 1;
  }
  else {
    rgt_pnl = 0;
  }

  if(y_bot != 0.) {
    bot_pnl = 1;
  }
  else {
    bot_pnl = 0;
  }

  if(y_top != 1.) {
    top_pnl = 1;
  }
  else {
    top_pnl = 0;
  }

/*
 * We only need to set maxbb to True if the plots are being
 * drawn to a PostScript or PDF workstation, because the
 * bounding box is already maximized for an NCGM/X11 window.
 */ 
  maxbb = 1;
  if(special_res->nglMaximize) {
    if( (strcmp(NhlClassName(wks),"psWorkstationClass")) && 
        (strcmp(NhlClassName(wks),"pdfWorkstationClass"))) {
      maxbb = 0;
    }
  }

/*
 * Error check the values that the user has entered, to make sure
 * they are valid.
 */
  if(xwsp_perc < 0 || xwsp_perc >= 100.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"ngl_panel: attribute nglPanelXWhiteSpacePercent must be >= 0 and < 100.\nDefaulting to 1.\n");
    xwsp_perc = 1.;
  }

  if(ywsp_perc < 0 || ywsp_perc >= 100.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"ngl_panel: attribute nglPanelYWhiteSpacePercent must be >= 0 and < 100.\nDefaulting to 1.\n");
    ywsp_perc = 1.;
  }
  
  if(x_lft < 0. || x_lft >= 1.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"ngl_panel: attribute nglPanelLeft must be >= 0.0 and < 1.0\nDefaulting to 0.\n");
    x_lft = 0.0;
  }
  
  if(x_rgt <= 0. || x_rgt > 1.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"ngl_panel: attribute nglPanelRight must be > 0.0 and <= 1.0\nDefaulting to 1.\n");
    x_rgt = 1.0;
  }
  
  if(y_top <= 0. || y_top > 1.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"ngl_panel: attribute nglPanelTop must be > 0.0 and <= 1.0\nDefaulting to 1.\n");
    y_top = 1.0;
  }
  
  if(y_bot < 0. || y_bot >= 1.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"Warning: ngl_panel: attribute nglPanelBottom must be >= 0.0 and < 1.0\nDefaulting to 0.\n");
    y_bot = 0.0;
  }
  
  if(x_rgt <= x_lft) {
    NhlPError(NhlFATAL,NhlEUNKNOWN,"ngl_panel: nglPanelRight (%g",x_rgt,")\n must be greater than nglPanelLeft (%g",x_lft,").\n");
    return;
  }
  
  if(y_top <= y_bot) {
    NhlPError(NhlFATAL,NhlEUNKNOWN,"ngl_panel: nglPanelTop (%g",y_top,")\n must be greater than nglPanelBottom (%g",y_bot,").\n");
    return;
  }
  
/* 
 * We assume all plots are the same size, so if we get the size of
 * one of them, then this should represent the size of the rest
 * of them.  Also, count the number of non-missing plots for later.
 */
  valid_plot   = -1;
  nvalid_plots = 0;
  for(i = 0; i < nplots; i++) {
    if(plots[i].nbase > 0 && plots[i].base != NULL) {
      if(valid_plot < 0) {
        NhlGetBB(*(plots[i].base),&bb);
        top    = bb.t;
        bottom = bb.b;
        left   = bb.l;
        right  = bb.r;
        valid_plot = i;
      }
      nvalid_plots++;
    }
  }
  if(nvalid_plots == 0) {
    NhlPError(NhlFATAL,NhlEUNKNOWN,"ngl_panel: all of the plots passed to ngl_panel appear to be invalid\n");
    return;
  }  
  else {
    if(panel_debug) {
      printf("There are %d valid plots out of %d total plots\n",nvalid_plots, nplots);
    }
  }

/*
 * Get the type of plot we have (contour, vector, xy) so we can
 * retrieve font heights and/or labelbar information if needed.
 * Labelbar will only be generated if user has a contour and/or 
 * vector plot, and either the contours or the vectors are being 
 * filled in multiple colors.
 */
  strcpy(plot_type,"unknown");

  if(plots[valid_plot].ncontour > 0 && plots[valid_plot].contour != NULL) {
    strcpy(plot_type,"contour");
/*
 * Get information on how contour plot is filled, so we can recreate 
 * labelbar, if requested.
 */
    NhlRLClear(grlist);
    NhlRLGetFloat(grlist,   "cnInfoLabelFontHeightF", &font_height);
    NhlRLGetInteger(grlist, "cnFillOn",               &fill_on);
    (void)NhlGetValues(*(plots[valid_plot].contour), grlist);

    if(panel_labelbar) {
      if(fill_on) {
        NhlRLClear(grlist);
        NhlRLGetIntegerArray(grlist, "cnFillColors",      &colors, &ncolors);
        NhlRLGetIntegerArray(grlist, "cnFillPatterns",    &patterns,
                                                          &npatterns);
        NhlRLGetFloatArray(grlist,   "cnFillScales",      &scales, &nscales);
        NhlRLGetInteger(grlist,      "cnMonoFillPattern", &mono_fill_pat);
        NhlRLGetInteger(grlist,      "cnMonoFillScale",   &mono_fill_scl);
        NhlRLGetInteger(grlist,      "cnMonoFillColor",   &mono_fill_col);
        NhlRLGetFloatArray(grlist,   "cnLevels",          &levels, &nlevels);
        (void)NhlGetValues(*(plots[valid_plot].contour), grlist);
      }
      else {
        panel_labelbar = 0;
      }
    }
  }
  else if(plots[valid_plot].nvector > 0 && plots[valid_plot].vector != NULL) {
    strcpy(plot_type,"vector");
/*
 * There are many possible ways in which multiply filled arrows or mulitply
 * color line vectors can be on:
 *
 *  vcFillArrowsOn = True       && vcMonoFillArrowFillColor = False
 *  vcFillArrowsOn = False      && vcMonoLineArrowColor     = False
 *  vcGlyphStyle   = FillArrows && vcMonoFillArrowFillColor = False
 *  vcGlyphStyle   = (LineArrows || CurlyVector) 
 *                 && vcMonoLineArrowColor     = False
 */
    NhlRLClear(grlist);
    NhlRLGetFloat(grlist,  "vcRefAnnoFontHeightF",     &font_height);
    NhlRLGetInteger(grlist,"vcGlyphStyle",             &glyph_style);
    NhlRLGetInteger(grlist,"vcFillArrowsOn",           &fill_arrows_on);
    NhlRLGetInteger(grlist,"vcMonoLineArrowColor",     &mono_line_color);
    NhlRLGetInteger(grlist,"vcMonoFillArrowFillColor", &mono_fill_arrow);
    (void)NhlGetValues(*(plots[valid_plot].vector), grlist);

    if(panel_labelbar) {
      if((fill_arrows_on && !mono_fill_arrow) || 
        (!fill_arrows_on && !mono_line_color) || 
        (glyph_style == NhlFILLARROW && !mono_fill_arrow) ||
        ((glyph_style == NhlLINEARROW || glyph_style == NhlCURLYVECTOR)
                                                   && !mono_line_color)) {
/*
 * There are no fill patterns in VectorPlot, only solids.
 */
        mono_fill_pat = 1;
        mono_fill_scl = 1;
        mono_fill_col = 0;
        NhlRLClear(grlist);
        NhlRLGetFloatArray(grlist,   "vcLevels",      &levels, &nlevels);
        NhlRLGetIntegerArray(grlist, "vcLevelColors", &colors, &ncolors );
        (void)NhlGetValues(*(plots[valid_plot].vector), grlist);
      }
      else {
        panel_labelbar = 0;
      }
    }
  }
  else if(plots[valid_plot].nxy > 0 && plots[valid_plot].xy != NULL &&
         *(plots[valid_plot].base) == *(plots[valid_plot].xy)) {
    strcpy(plot_type,"xy");
/*
 * Retrieve this resource later, *after* the plots have been rescaled.
 */
    NhlRLClear(grlist);
    NhlRLGetFloat(grlist, "tiXAxisFontHeightF", &font_height);
    (void)NhlGetValues(*(plots[valid_plot].xy), grlist);
    font_height *= 0.6;
  }
  else {
    font_height = 0.01;
    NhlPError(NhlWARNING,NhlEUNKNOWN,"Warning: ngl_panel: unrecognized plot type, thus unable to get information for font height.\nDefaulting to %g", font_height);
  }

/*
 * Create array to save the valid plot objects, plus a labelbar and
 * a common title, if any.
 */

  nnewplots = nvalid_plots;
  if(panel_labelbar) nnewplots++;
  if(main_string_on) nnewplots++;

  newplots = (nglPlotId *)malloc(nnewplots*sizeof(nglPlotId));
  for(i = 0; i < nnewplots; i++ ) {
    newplots[i].base  = (int*)malloc(sizeof(int));
    newplots[i].nbase = 1;
  }
   
/*
 * plot_width  : total width of plot with all of its annotations
 * plot_height : total height of plot with all of its annotations
 * total_width : plot_width plus white space on both sides
 * total_height: plot_height plus white space on top and bottom
 * xwsp/ywsp   : white space  
 */
  plot_width  = right - left;
  plot_height = top - bottom;
  
  xwsp = xwsp_perc/100. * plot_width;
  ywsp = ywsp_perc/100. * plot_height;
  
  total_width  = 2.*xwsp + plot_width;
  total_height = 2.*ywsp + plot_height;

/*
 * If we are putting a common labelbar at the bottom (right), make 
 * it 2/10 the height (width) of the plot.
 */
  if(panel_labelbar) {
/*
 * Set some resource values. Some resources are calculated by this panel
 * routine later (like width, height, font height, etc) and some just
 * have a default, like PerimOn and Alignment.
 */
    lb_perim_on  = special_res->nglPanelLabelBarPerimOn;
    lb_alignment = special_res->nglPanelLabelBarAlignment;
    lb_orient    = special_res->nglPanelLabelBarOrientation;
    if(lb_orient == NhlVERTICAL) {
      labelbar_width = 0.20 * plot_width + 2.*xwsp;
/*
 * Adjust height depending on whether we have one row or multiple rows.
 */
      if(nplots > 1 && nrows > 1) {
        labelbar_height  = (nrows-1) * (2.*ywsp + plot_height);
      }
      else {
        labelbar_height  = plot_height;
      }
    }
    else {
      labelbar_height = 0.20 * plot_height + 2.*ywsp;
/*
 * Adjust width depending on whether we have one column or multiple 
 * columns.
 */
      if(nplots > 1 && ncols > 1) {
        labelbar_width  = (ncols-1) * (2.*xwsp + plot_width);
      }
      else {
        labelbar_width  = plot_width;
      }
    }
  }
  else {
    labelbar_height = 0.;
    labelbar_width  = 0.;
  }

/*
 * We want:
 *
 *   ncols * scale * total_width  <= x_rgt - x_lft (the viewport width)
 *   nrows * scale * total_height <= y_top - y_bot (the viewport height)
 *
 * By taking the minimum of these two, we get the scale
 * factor that we need to fit all plots on a page.
 *
 * Previously, we used to include xrange and yrange as part of the min
 * statement. This seemed to cause problems if you set one of
 * nglPanelTop/Bottom/Right/Left however, so I removed it.  Initial
 * testing on Sylvia's panel examples seems to indicate this is okay.
 *
 */
  xrange = x_rgt - x_lft;
  yrange = y_top - y_bot;
  
  if(lb_orient == NhlHORIZONTAL) {
    row_scale = yrange/(nrows*total_height+labelbar_height);
    col_scale = xrange/(ncols*total_width);
    scale     = min(col_scale,row_scale);
    yrange    = yrange - scale * labelbar_height;
  }
  else {
    row_scale = yrange/(nrows*total_height);
    col_scale = xrange/(ncols*total_width+labelbar_width);
    scale     = min(col_scale,row_scale);
    xrange    = xrange - scale * labelbar_width;
  }

/*
 * Calculate new width and height.
 */
  new_plot_width  = scale * plot_width;
  new_plot_height = scale * plot_height; 

/*
 * Calculate new white space.
 */
  xwsp = xwsp_perc/100. * new_plot_width;
  ywsp = ywsp_perc/100. * new_plot_height;

/*
 * Calculate new total width & height w/white space.
 */
  new_total_width  = 2.*xwsp + new_plot_width;  
  new_total_height = 2.*ywsp + new_plot_height; 

/*
 * Calculate total amt of white space left in both X and Y directions.
 */
  xsp = xrange - new_total_width*ncols;  
  ysp = yrange - new_total_height*nrows;

/*
 * Retrieve viewport size of plot. The size of all plots are supposed
 * to be the same, so you only need to do this for the first plot.
 * You might break some code if you change which plot you get the 
 * viewport sizes from, because some users knowingly make the first
 * plot larger, and go ahead and panel them anyway.
 */
  NhlRLClear(grlist);
  NhlRLGetFloat(grlist,"vpXF",     &vpx);
  NhlRLGetFloat(grlist,"vpYF",     &vpy);
  NhlRLGetFloat(grlist,"vpWidthF", &vpw);
  NhlRLGetFloat(grlist,"vpHeightF",&vph);
  (void)NhlGetValues(*(plots[valid_plot].base), grlist);
  
/*
 * Calculate distances from plot's left/right/top/bottom positions
 * to its leftmost/rightmost/topmost/bottommost annotation.
 */
  dxl = scale * (vpx-left); 
  dxr = scale * (right-(vpx+vpw));
  dyt = scale * (top-vpy);
  dyb = scale * ((vpy-vph)-bottom);
  
  ypos = (float *)malloc(nrows*sizeof(float));
  for(i = 0; i < nrows; i++) {
    ypos[i] = y_top - ywsp - dyt - (ysp/2.+new_total_height*i);
    if(i) {
      min_ypos = min(min_ypos,ypos[i]); 
      max_ypos = max(max_ypos,ypos[i]); 
    }
    else {
      min_ypos = ypos[i]; 
      max_ypos = ypos[i]; 
    }
  }

/*
 * Variable to store rightmost location of rightmost plot, and topmost
 * location of top plot.
 */
  max_rgt = 0.;
  max_top = 0.;

/*
 * Variable to hold original viewport coordinates, and annotations (if
 * they exist).
 */
  old_vp = (float *)malloc(4 * nplots * sizeof(float));

/*
 * Loop through each row and create each plot in the new scaled-down
 * size. We will draw plots later, outside the loop.
 */
  num_plots_left = nplots;
  nplot          = 0;
  nvalid_plot    = 0;
  nr             = 0;
  first_time     = 1;

  while(num_plots_left > 0) {
    new_ncols = min(num_plots_left,row_spec[nr]);
    
/*
 * "xsp" is space before plots.
 */
    if(panel_center) {
      xsp = xrange - new_total_width * new_ncols;
    }
    else {
      xsp = xrange - new_total_width * ncols;
    }
/*
 * Calculate new x positions.
 */
    xpos = (float *)malloc(new_ncols*sizeof(float));
    for(i = 0; i < new_ncols; i++) {
      xpos[i] = x_lft + xwsp + dxl +(xsp/2.+new_total_width*i);
    }

    for (nc = 0; nc < new_ncols; nc++) {
      if(plots[nplot].nbase > 0 && plots[nplot].base != NULL) {
        pplot.base = plots[nplot].base;
        nplot4 = nplot * 4;
/*
 * Get the size of the plot so we can rescale it. Technically, each
 * plot should be the exact same size, but we can't assume that.
 */
        NhlRLClear(grlist);
        NhlRLGetFloat(grlist,"vpXF",     &old_vp[nplot4]);
        NhlRLGetFloat(grlist,"vpYF",     &old_vp[nplot4+1]);
        NhlRLGetFloat(grlist,"vpWidthF", &old_vp[nplot4+2]);
        NhlRLGetFloat(grlist,"vpHeightF",&old_vp[nplot4+3]);
        (void)NhlGetValues(*(pplot.base), grlist);
        
        if(panel_debug) {
          printf("-------Panel viewport values for each plot-------\n");
          printf("    plot # %d\n", nplot);
          printf("    x,y     = %g,%g\n", xpos[nc], ypos[nr]);
          printf("orig wdt,hgt = %g,%g\n", old_vp[nplot4+2], 
                                           old_vp[nplot4+3]);
          printf("    wdt,hgt = %g,%g\n", scale*old_vp[nplot4+2], 
                                          scale*old_vp[nplot4+3]);
        }

        NhlRLClear(srlist);
        NhlRLSetFloat(srlist,"vpXF",     xpos[nc]);
        NhlRLSetFloat(srlist,"vpYF",     ypos[nr]);
        NhlRLSetFloat(srlist,"vpWidthF", scale*old_vp[nplot4+2]);
        NhlRLSetFloat(srlist,"vpHeightF",scale*old_vp[nplot4+3]);
        (void)NhlSetValues(*(pplot.base), srlist);

/*
 * Retain maximum width and height for later.
 */
        if(!first_time) {
          max_width  = max(max_width, old_vp[nplot4+2]);
          max_height = max(max_height,old_vp[nplot4+3]);
        }
        else {
          max_width  = old_vp[nplot4+2];
          max_height = old_vp[nplot4+3];
          first_time = 0;
        }

/*
 * Save this plot.
 */
        *(newplots[nvalid_plot].base) = *(pplot.base);
/*
 * Info for possible labelbar or main_string
 */
        if(main_string_on || panel_labelbar || draw_boxes) {
          NhlGetBB(*(pplot.base),&bb);
          top    = bb.t;
          bottom = bb.b;
          left   = bb.l;
          right  = bb.r;
          max_rgt = max(right,max_rgt);
          max_top = max(top,max_top);
        }
        nvalid_plot++;
      }
      else {
        if(panel_debug) {
          printf("    plot %d is missing\n", nplot);
        }
      }
      nplot++;
    }
/*
 * Retain the smallest and largest x positions, since xpos gets
 * recalculated every time inside the loop.
 */
    if(nr == 0) {
      max_xpos = xmax_array(xpos,new_ncols);
      min_xpos = xmin_array(xpos,new_ncols);
    }
    else {
      min_xpos = min(xmin_array(xpos,new_ncols), min_xpos);
      max_xpos = max(xmax_array(xpos,new_ncols), max_xpos);
    }
    num_plots_left = nplots - nplot;
    nr++;                                   /* increment rows */
    free(xpos);
  }

/*
 * Calculate the biggest rescaled widths and heights (technically, they
 * should all be the same).  These values will be used a few times 
 * throughout the rest of the code.
 */
  scaled_width  = scale * max_width;
  scaled_height = scale * max_height;

/*
 * Check if a labelbar is to be drawn at the bottom.
 */
  if(panel_labelbar)  {
    if(!strcmp(plot_type,"contour") || !strcmp(plot_type,"vector")) {
/*
 * Set labelbar height, width, and font height.
 */
      labelbar_height *= scale;
      labelbar_width  *= scale;
      if(special_res->nglPanelLabelBarFontHeightF > 0.) {
        lb_fh = special_res->nglPanelLabelBarFontHeightF;
      }
      else {
        lb_fh = font_height;
      }
/*
 * Set some labelbar resources if they haven't already been set by
 * user. 
 */
      if(special_res->nglPanelLabelBarWidthF > 0.) {
        lb_w = special_res->nglPanelLabelBarWidthF;
      }
      else {
        lb_w = labelbar_width;
      }
      
      if(special_res->nglPanelLabelBarHeightF > 0.) {
        lb_h = special_res->nglPanelLabelBarHeightF;
      }
      else {
        lb_h = labelbar_height;
      }
/*
 * Set position of labelbar depending on whether it's horizontal or
 * vertical.
 */
      if(lb_orient == NhlHORIZONTAL) {
        if(special_res->nglPanelLabelBarYF > 0.) {
          lb_y = special_res->nglPanelLabelBarYF;
        }
        else {
          lb_y = max(ywsp+labelbar_height,bottom-ywsp);
        }

        if(special_res->nglPanelLabelBarXF > 0.) {
          lb_x = special_res->nglPanelLabelBarXF;
        }
        else if(ncols == 1 && lb_w <= scaled_width) {
          lb_x = min_xpos + (scaled_width-lb_w)/2.;
        }
        else {
          tmp_range = x_rgt - x_lft;
          lb_x      = x_lft + (tmp_range - lb_w)/2.;
        }
        if(special_res->nglPanelLabelBarOrthogonalPosF > -1.) {
          lb_y += special_res->nglPanelLabelBarOrthogonalPosF;
        }
        if(special_res->nglPanelLabelBarParallelPosF > -1.) {
          lb_x += special_res->nglPanelLabelBarParallelPosF;
        }
      }
      else {
        if(special_res->nglPanelLabelBarXF > 0.) {
          lb_x = special_res->nglPanelLabelBarXF;
        }
        else {
          lb_x = min(1.-(xwsp+labelbar_width),max_rgt+xwsp);
        }
        if(special_res->nglPanelLabelBarYF > 0.) {
          lb_y = special_res->nglPanelLabelBarYF;
        }
        else if(nrows == 1 && lb_h <= scaled_height) {
          lb_y = ypos[0]-(scaled_height - lb_h)/2.;
        }
        else {
          tmp_range = y_top - y_bot;
          lb_y      = y_top-(tmp_range - lb_h)/2.;
        }
        if(special_res->nglPanelLabelBarOrthogonalPosF > -1.) {
          lb_x += special_res->nglPanelLabelBarOrthogonalPosF;
        }
        if(special_res->nglPanelLabelBarParallelPosF > -1.) {
          lb_y += special_res->nglPanelLabelBarParallelPosF;
        }
      }
/*
 * Now begin setting the labelbar resources.
 */
      NhlRLSetFloat       (lb_rlist,"vpXF",              lb_x);
      NhlRLSetFloat       (lb_rlist,"vpYF",              lb_y);
      NhlRLSetFloat       (lb_rlist,"vpWidthF",          lb_w);
      NhlRLSetFloat       (lb_rlist,"vpHeightF",         lb_h);
      NhlRLSetString      (lb_rlist,"lbAutoManage",      "False");
      NhlRLSetInteger     (lb_rlist,"lbOrientation",     lb_orient);
      NhlRLSetIntegerArray(lb_rlist,"lbFillColors",      colors, ncolors);
      NhlRLSetInteger     (lb_rlist,"lbBoxCount",        ncolors);
      NhlRLSetFloatArray  (lb_rlist,"lbLabelStrings",    levels, nlevels);
      NhlRLSetInteger     (lb_rlist,"lbPerimOn",         lb_perim_on);
      NhlRLSetFloat       (lb_rlist,"lbLabelFontHeightF",lb_fh);
      NhlRLSetInteger     (lb_rlist,"lbLabelAlignment",  lb_alignment);
      /*
       * Check if we want different fill patterns or fill scales.  If so, we
       * have to pass these on to the labelbar.
       */
      NhlRLSetInteger(lb_rlist,"lbMonoFillColor",   mono_fill_col);
      NhlRLSetInteger(lb_rlist,"lbMonoFillPattern", mono_fill_pat);
      if(!mono_fill_pat) {
        NhlRLSetIntegerArray(lb_rlist,"lbFillPatterns", patterns, npatterns);
      }
      NhlRLSetInteger(lb_rlist,"lbMonoFillScale", mono_fill_scl);
      if(!mono_fill_scl) {
        NhlRLSetFloatArray(lb_rlist,"lbFillScale", scales, nscales);
      }
/*
 * Create the labelbar and add to our list of plots.
 */
      NhlCreate(&labelbar_object,"labelbar",NhllabelBarClass,wks,lb_rlist);
      *(newplots[nvalid_plot].base) = labelbar_object;
/*
 * Increment plot counter.
 */
      nvalid_plot++;
/*
 * Free up memory.
 */
      free(levels);
      free(colors);
      if(!mono_fill_pat) free(patterns);
      if(!mono_fill_scl) free(scales);
    }
/*
 * The plot_type is not either vector or contour, so we have to bail on
 * creating a labelbar. 
 */
    else {
      NhlPError(NhlWARNING,NhlEUNKNOWN,"ngl_panel: unrecognized plot type for getting labelbar information. Ignoring labelbar request.");
    }
  }
/*
 * If some of the paneled plots are missing, we need to take these into
 * account so that the maximization will still work properly.  For
 * example, if we ask for a 2 x 2 configuration, but plots 1 and 3 (the
 * rightmost plots) are missing, then we need to set a new resource
 * called nglPanelInvsblRight to whatever approximate X value it 
 * would have been if those plots weren't missing.  Setting just 
 * nglPanelRight won't work in this case, because that resource is only
 * used to control where the plots are drawn in a 0 to 1 square, and
 * not to indicate the rightmost location of the rightmost graphic
 * (which could be a vertical labelbar.
 */
  newbb  = (NhlBoundingBox *)malloc(nnewplots*sizeof(NhlBoundingBox));
/*
 * Get largest bounding box that encompasses all non-missing graphical
 * objects.
 */
  for( i = 0; i < nnewplots; i++ ) { 
    NhlGetBB(*(newplots[i].base),&newbb[i]);

    if(i) {
      newtop = max(newtop,newbb[i].t);
      newbot = min(newbot,newbb[i].b);
      newlft = min(newlft,newbb[i].l);
      newrgt = max(newrgt,newbb[i].r);
    }
    else {
      newtop = newbb[i].t;
      newbot = newbb[i].b;
      newlft = newbb[i].l;
      newrgt = newbb[i].r;
/*
 * Get viewport coordinates of one of the scaled plots so
 * that we can calculate the distances between the viewport
 * coordinates and the edges of the bounding boxes.
 */
      NhlRLClear(grlist);
      NhlRLGetFloat(grlist,"vpXF",      &vpx);
      NhlRLGetFloat(grlist,"vpYF",      &vpy);
      NhlRLGetFloat(grlist,"vpWidthF",  &vpw);
      NhlRLGetFloat(grlist,"vpHeightF", &vph);
      (void)NhlGetValues(*(newplots[i].base), grlist);
      dxl = vpx-newbb[i].l;
      dxr = newbb[i].r-(vpx+vpw);
      dyt = (newbb[i].t-vpy);
      dyb = (vpy-vph)-newbb[i].b;
    }
  }

/*
 * This section makes sure that even though some plots may have been
 * missing, we still keep the original bounding box as if all of the
 * plots had been present. This is necessary so that the maximization
 * for PS/PDF output is done properly.
 *
 * We don't know the bounding box values for the possible missing
 * plots, so we have to estimate them based on viewport sizes and
 * bounding box values from existing plots. We can do this because 
 * all plots are supposed to be the same size.
 */
  xrgt = max_xpos + vpw + dxr;
  xlft = min_xpos - dxl;
  xtop = max_ypos + dyt;
  xbot = min_ypos - vph - dyb;

  if(!rgt_pnl && xrgt > newrgt) {
    special_res->nglPanelInvsblRight = xrgt;
    if(panel_debug) {
      printf("nglPanelInvsblRight = %g\n", special_res->nglPanelInvsblRight);
    }
  }

  if(!lft_pnl && xlft < newlft) {
    special_res->nglPanelInvsblLeft = xlft;
    if(panel_debug) {
      printf("nglPanelInvsblLeft = %g\n", special_res->nglPanelInvsblLeft);
    }
  }
    
  if(!top_pnl && xtop > newtop) {
    special_res->nglPanelInvsblTop = xtop;
    if(panel_debug) {
      printf("nglPanelInvsblTop = %g\n", special_res->nglPanelInvsblTop);
    }
  }
    
  if(!bot_pnl && xbot < newbot) {
    special_res->nglPanelInvsblBottom = xbot;
    if(panel_debug) {
      printf("nglPanelInvsblBottom = %g\n", special_res->nglPanelInvsblBottom);
    }
  }


/* 
 * Draw plots plus labelbar and main title (if they exist). This is
 * also where the plots will be maximized for PostScript output,
 * if so indicated.
 */
  draw_and_frame(wks, newplots, nnewplots, 1, special_res);

/*
 * Restore nglPanelInvsbl* resources because these should only
 * be set internally.
 */
  special_res->nglPanelInvsblRight  = -999.;
  special_res->nglPanelInvsblLeft   = -999.;
  special_res->nglPanelInvsblTop    = -999.;
  special_res->nglPanelInvsblBottom = -999.;

/*
 * Restore plots to original size.
 */
  if(!panel_save) {
    for(i = 0; i < nplots; i++) {
      if(plots[i].nbase > 0 && plots[i].base != NULL) {
        nplot4 = 4 * i;
        
        NhlRLClear(srlist);
        NhlRLSetFloat(srlist,"vpXF",     old_vp[nplot4]);
        NhlRLSetFloat(srlist,"vpYF",     old_vp[nplot4+1]);
        NhlRLSetFloat(srlist,"vpWidthF", old_vp[nplot4+2]);
        NhlRLSetFloat(srlist,"vpHeightF",old_vp[nplot4+3]);
        (void)NhlSetValues(*(plots[i].base), srlist);
      }
    }
  }
  if(!is_row_spec) free(row_spec);
  free(newplots);
  free(old_vp);
  free(ypos);
  free(newbb);
}
