#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <math.h>
#include "gsun.h"

#define NINT(x) ( (ceil((x))-(x)) > ( (x)-floor((x)))) ? floor((x)) : ceil((x))

/*
 * Define some global strings.
 */
const char *polylinestr = "polyline";
  
int global_wk_orientation = -1;
int nhl_initialized = 0;

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

ng_size_t *convert_dims(int *tmp_dimensions,int n_dimensions)
{
  ng_size_t i, *dimensions;

  dimensions = (ng_size_t *)malloc(sizeof(ng_size_t) * n_dimensions);
  for (i = 0; i < n_dimensions; i++) {
    ((ng_size_t *)dimensions)[i] = ((int*)tmp_dimensions)[i];
  }
  return(dimensions);
}

/*
 * This function checks a given resource list to see if a resource
 * has been set. If it has, then 1 is returned; otherwise 0 is returned.
 * This function is used to keep from setting resources internally
 * when they've been set by the user in a PyNGL script.
 */
int is_res_set(ResInfo *res_list, char *resname)
{
  int i;

  if(res_list->nstrings <= 0) {
    return(0);
  }

  for(i = 0; i < res_list->nstrings; i++) {
    if(!strcmp((res_list->strings)[i],resname)) {
      return(1);
    }
  }
  return(0);
}


/*
 * This is for the PyNGL function Ngl.get_bounding_box.
 */
void getbb (int pid, float *t, float *b, float *l, float *r)
{
  NhlBoundingBox *Box;
  Box = (NhlBoundingBox *) malloc(sizeof(NhlBoundingBox));
  NhlGetBB(pid,Box);
  *t = Box->t;
  *b = Box->b;
  *l = Box->l;
  *r = Box->r;
  free(Box);
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
 * The nglPaperHeight and nglPaperWidth should no longer be used, as
 *    we now have wkPaperWidthF and wkPaperHeightF. We will still 
 *    recognize them for now.
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
  int grlist;
  NhlWorkOrientation paper_orient; 

  is_debug     = special_res->nglDebug;
/*
 * This next section is for PS/PDF output.
 */
/*
 * First, make sure we use the correction orientation. If nglPaperOrientation
 * was set by the user, use this. Otherwise, if wkOrientation was set by
 * the user, use this. Otherwise, default to 3 (auto).
 */
  paper_orient = 3;                       /* Default to 3 (auto) */

  if(special_res->nglPaperOrientation == -1) {
    if(global_wk_orientation >= 0) {
      paper_orient = global_wk_orientation;
    }
  }
  else {
    if(special_res->nglPaperOrientation == NhlPORTRAIT || 
       special_res->nglPaperOrientation == NhlLANDSCAPE || 
       special_res->nglPaperOrientation == 3) { 
      paper_orient = special_res->nglPaperOrientation;
    }
    else {
      printf("Unrecognized value for nglPaperOrientation (%d).\nDefaulting to auto.\n",special_res->nglPaperOrientation);
    }
  }

/*
 * Get paper width and height.
 */
  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);
  NhlRLGetFloat(grlist,"wkPaperHeightF", &paper_height);
  NhlRLGetFloat(grlist,"wkPaperWidthF",  &paper_width);
  (void)NhlGetValues(wks, grlist);

/*
 * The nglPaperWidth/Height resources override the 
 * wkPaperWidthF/wkPaperHeightF resources.
 */
  if(special_res->nglPaperHeight != 11.) {
    paper_height = special_res->nglPaperHeight;
    printf("Warning: nglPaperWidth/nglPaperHeight are deprecated. Use wkPaperWidthF/wkPaperHeightF instead.\n");
  }
  if(special_res->nglPaperWidth != 8.5) {
    paper_width = special_res->nglPaperWidth;
    printf("Warning: nglPaperWidth/nglPaperHeight are deprecated. Use wkPaperWidthF/wkPaperHeightF instead.\n");
  }
  paper_margin = special_res->nglPaperMargin;

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
 
  if( (paper_orient == NhlPORTRAIT) || ((paper_orient == 3) &&
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
 * This function maximizes the given plots in the viewport.
 */

void maximize_plots(int wks, nglPlotId *plot, int nplots, int ispanel, 
                    nglRes *special_res)
{
  NhlBoundingBox box; 
  float top, bot, lft, rgt, uw, uh;
  float scale, vpx, vpy, vpw, vph, dx, dy, new_uw, new_uh, new_ux, new_uy;
  float new_vpx, new_vpy, new_vpw, new_vph, margin = 0.02;
  int srlist, grlist;
  const char *name;

/*
 * If wks < 0, this means we have a "bad" workstation.
 */
  if(wks < 0) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"maximize_plots: Invalid workstation. No plot maximization can take place.");
    return;
  }
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

    uw = rgt - lft;
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

  name = NhlClassName(wks);
  if(name != NULL && (!strcmp(name,"psWorkstationClass") ||
                     !strcmp(name,"pdfWorkstationClass") ||
		      !strcmp(name,"documentWorkstationClass"))) {
/*
 * Compute and set device coordinates that will make plot fill the 
 * whole page.
 */
    compute_ps_device_coords(wks, plot, nplots, special_res);
  }
}

/*
 * This function overlays the given plot object on an irregular object
 * so it can take an irregular axis and make it log or linear.
 * The axes types are determined by the resources nglXAxisType
 * and nglYAxisType.
 *
 * ir_res carries over any tm* and pmTick* resources that might have
 * been set in the original plot.
 *
 * plot_res is just the full list of resources set in the original
 * plot before it was overlaid. This is used to check for other
 * resources that might have been set (in this case, for scaling).
 */
void overlay_on_irregular(int wks, nglPlotId *plot, ResInfo *plot_res,
                          ResInfo *ir_res, nglRes *special_res)
{
  int xaxistype, yaxistype, overlay_plot, base_plot;
  float xmin, xmax, ymin, ymax, *xpts, *ypts;
  int xpts_ndims, ypts_ndims;
  int xreverse, yreverse;  
  int ir_rlist, grlist;
  ng_size_t *xpts_dimsizes, *ypts_dimsizes;

  overlay_plot = *(plot->base);

/*
 * Get the values of the special resources.
 */
  xaxistype = special_res->nglXAxisType;
  yaxistype = special_res->nglYAxisType;

/*
 * If both axes at this point are Irregular, then there is no point
 * in overlaying it on an irregular plot class.  Just return it as is.
 */

  if(xaxistype == NhlIRREGULARAXIS && yaxistype == NhlIRREGULARAXIS) {
    return;
  }

/*
 * Error checking.
 *
 * The values must be 0 (irregular), 1 (linear), or 2 (log).
 */

  if(xaxistype != NhlIRREGULARAXIS && xaxistype != NhlLINEARAXIS &&
     xaxistype != NhlLOGAXIS) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"Value of nglXAxisType is invalid.");
    NhlPError(NhlWARNING,NhlEUNKNOWN,"Defaulting to IrregularAxis.");
    xaxistype = NhlIRREGULARAXIS;
    return;
  }

  if(yaxistype != NhlIRREGULARAXIS && yaxistype != NhlLINEARAXIS &&
     yaxistype != NhlLOGAXIS) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"Value of nglYAxisType is invalid.");
    NhlPError(NhlWARNING,NhlEUNKNOWN,"Defaulting to IrregularAxis.");
    yaxistype = NhlIRREGULARAXIS;
    return;
  }

/*
 * Retrieve transformation information about existing plot so we can
 * use these values to create new overlay plot object.
 */

  grlist = NhlRLCreate(NhlGETRL);
  NhlRLClear(grlist);

  NhlRLGetFloat(grlist,"trXMinF",   &xmin);
  NhlRLGetFloat(grlist,"trXMaxF",   &xmax);
  NhlRLGetFloat(grlist,"trYMinF",   &ymin);
  NhlRLGetFloat(grlist,"trYMaxF",   &ymax);
  NhlRLGetInteger(grlist,"trYReverse", &yreverse);
  NhlRLGetInteger(grlist,"trXReverse", &xreverse);
  NhlGetValues(overlay_plot,grlist);

/*
 * If one of the axes is to remain irregular, we need to
 * retrieve the coordinate values so we can reset the
 * irregular axis later with trX/YCoordPoints.
 */
  if(!strcmp(NhlClassName(overlay_plot),"contourPlotClass")) {
    NhlRLClear(grlist);
    if(xaxistype == NhlIRREGULARAXIS && yaxistype != NhlIRREGULARAXIS) {
      NhlRLGetMDFloatArray(grlist,"sfXArray",&xpts,&xpts_ndims,
                           &xpts_dimsizes);
    }
    if(yaxistype == NhlIRREGULARAXIS && xaxistype != NhlIRREGULARAXIS) {
      NhlRLGetMDFloatArray(grlist,"sfYArray",&ypts,&ypts_ndims,
                           &ypts_dimsizes);
    }
    NhlGetValues(*(plot->sffield),grlist);
  }
  else {
    if(!strcmp(NhlClassName(overlay_plot),"vectorPlotClass") ||
       !strcmp(NhlClassName(overlay_plot),"streamlinePlotClass")) { 
      NhlRLClear(grlist);
      if(xaxistype == NhlIRREGULARAXIS && yaxistype != NhlIRREGULARAXIS) {
        NhlRLGetMDFloatArray(grlist,"vfXArray",&xpts,&xpts_ndims,
                             &xpts_dimsizes);
      }
      if(yaxistype == NhlIRREGULARAXIS &&  xaxistype != NhlIRREGULARAXIS) {
        NhlRLGetMDFloatArray(grlist,"vfYArray",&ypts,&ypts_ndims,
                             &ypts_dimsizes);
      }
      NhlGetValues(*(plot->vffield),grlist);
    }   
    else {
      NhlPError(NhlWARNING,NhlEUNKNOWN,"Invalid plot for overlay.");
      return;
    }
  }

/*
 * If x/yaxistype is irregular, then we must set trX/YCoordPoints
 * in order to retain the irregular axis.
 *
 * Also, if xpts or ypts are NULL, this means the corresponding
 * axis can't be irregular, and thus we default to linear.
 *
 */
  if(xpts == NULL && xaxistype == 0) {
    xaxistype = 1;
  }
  if(ypts == NULL && yaxistype == 0) {
    yaxistype = 1;
  }
/*
 * We have three possible cases that can exist at this point:
 *
 *   Case 1: Both X and Y axes are either linear or log.
 *   Case 2: X axis is irregular and Y axis is linear or log.
 *   Case 3: Y axis is irregular and X axis is linear or log.
 *
 * First, set resources common to all three cases. 
 */

  if(ir_res == NULL) {
    ir_rlist = NhlRLCreate(NhlSETRL);
    NhlRLClear(ir_rlist);
  }
  else {
    ir_rlist = ir_res->id;
  }
  NhlRLSetInteger(ir_rlist,"trXAxisType", xaxistype);
  NhlRLSetInteger(ir_rlist,"trYAxisType", yaxistype);
  NhlRLSetInteger(ir_rlist,"trYReverse",  yreverse);
  NhlRLSetInteger(ir_rlist,"trXReverse",  xreverse);
  NhlRLSetFloat(ir_rlist,  "trXMinF",     xmin);
  NhlRLSetFloat(ir_rlist,  "trXMaxF",     xmax);
  NhlRLSetFloat(ir_rlist,  "trYMinF",     ymin);
  NhlRLSetFloat(ir_rlist,  "trYMaxF",     ymax);
/*
 * A tickmark object won't appear if the original overlay plot
 * had it turned off, so force it to "Always" here, unless
 * the user has already set it.
 */
  if(!is_res_set(ir_res,"pmTickMarkDisplayMode")) { 
    NhlRLSetString(ir_rlist, "pmTickMarkDisplayMode", "Always");
  }

/*
 * Case 1: Both X and Y axes are either linear or log, nothing
 *         additional needed.
 *
 * Case 2: X axis is irregular and Y axis is linear or log.
 */
  if(xaxistype == NhlIRREGULARAXIS && yaxistype != NhlIRREGULARAXIS) {
    NhlRLSetMDFloatArray(ir_rlist,"trXCoordPoints",xpts,xpts_ndims,
                         xpts_dimsizes);
  }
/*
 * Case 3: Y axis is irregular and X axis is linear or log.
 */
  if(yaxistype == NhlIRREGULARAXIS && xaxistype != NhlIRREGULARAXIS) {
    NhlRLSetMDFloatArray(ir_rlist,"trYCoordPoints",ypts,ypts_ndims,
                         ypts_dimsizes);
  }

/*
 * Create the irregular plot object, and overlay the current plot
 * on this new object.
 */
  NhlCreate(&base_plot, "IrregularPlot",NhlirregularPlotClass,wks,ir_rlist);
  NhlAddOverlay(base_plot,overlay_plot,-1);

/*
 * Set the new base plot.
 */
  *(plot->base) = base_plot;

/*
 * We need to scale the plot again, because the old scales don't
 * apply after you do an overlay.
 */ 
  if(special_res->nglScale) scale_plot(*(plot->base),plot_res,0);

/*
 * Point tickmarks outward if requested specifically by user.
 */
  if(special_res->nglPointTickmarksOutward) {
    point_tickmarks_out(*(plot->base),plot_res);
  }

  return;
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
    for(i = lcount; i >= 0; i--) icols[lcount-i] = (int)(fcols[i] + 0.5);
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
  NhlRLSetIntegerArray(srlist, set_resname, icols, (ng_size_t)(lcount+1));
  NhlSetValues(plot,srlist);

  free(icols);
}

/*
 * This function "scales" the tickmarks and labels on the axes
 * so they are the same size/length.
 */

void scale_plot(int plot, ResInfo *res, int flag)
{
  int srlist, grlist, mode;
  float xfont, yfont;

  srlist = NhlRLCreate(NhlSETRL);
  grlist = NhlRLCreate(NhlGETRL);

/*
 * Originally, before we had the tmEqualizeXYSizes resource, we
 * had to retrieve all the tickmark info ourselves, and then
 * recalculate the sizes using averages.  Now, we just
 * set tmEqualizeXYSizes to True.  But, we only do this if
 * pmTickMarkDisplayMode is NhlCONDITIONAL or NhlALWAYS.
 * 
 * We still have to set the tiX/Y axis font sizes, though.
 */

  NhlRLClear(grlist);
  NhlRLGetFloat(grlist,"tiXAxisFontHeightF", &xfont);
  NhlRLGetFloat(grlist,"tiYAxisFontHeightF", &yfont);
  NhlRLGetInteger(grlist,"pmTickMarkDisplayMode", &mode);
  NhlGetValues(plot,grlist);

/*
 * Reset these resources, making the values the same for
 * both axes.
 */

  NhlRLClear(srlist);
  if(!is_res_set(res,"tiXAxisFontHeightF")) { 
    NhlRLSetFloat(srlist, "tiXAxisFontHeightF", (xfont+yfont)/2.);
  }
  if(!is_res_set(res,"tiYAxisFontHeightF")) {
    NhlRLSetFloat(srlist, "tiYAxisFontHeightF", (xfont+yfont)/2.);
  }
/*
 * This next resource takes care of making the tickmark lengths
 * and tickmark labels the same size.  Only set it if the tickmark
 * display mode is NhlCONDITIONAL or NhlALWAYS and flag is not 1.
 */
  if( flag != 1 && (mode == NhlCONDITIONAL || mode == NhlALWAYS) &&
     !is_res_set(res,"tmEqualizeXYSizes")) {
    NhlRLSetInteger(srlist, "tmEqualizeXYSizes", 1);
  }
  NhlSetValues(plot,srlist);
  return;
}

/*
 * Add a straight reference line to an XY plot. 0=x, 1=y.
 */
void add_ref_line(int wks, int plot, nglRes *special_res)
{
  int xgsid, ygsid, srlist, prlist, grlist;
  int *xprim_object, *yprim_object, color;
  float xmin, xmax, ymin, ymax, x[2], y[2], thickness;

  prlist = NhlRLCreate(NhlSETRL);
  srlist = NhlRLCreate(NhlSETRL);
  grlist = NhlRLCreate(NhlGETRL);

/*
 * Clear getvalues resource list and use it to retrieve axes limits.
 */
  NhlRLClear(grlist);

  if(special_res->nglYRefLine != -999.) {
    NhlRLGetFloat(grlist,"trXMinF", &xmin);
    NhlRLGetFloat(grlist,"trXMaxF", &xmax);
  }
  if(special_res->nglXRefLine != -999.) {
    NhlRLGetFloat(grlist,"trYMinF", &ymin);
    NhlRLGetFloat(grlist,"trYMaxF", &ymax);
  }
  NhlGetValues(plot,grlist);

  if(special_res->nglYRefLine != -999.) {
/*
 * Create graphic style object on which to draw primitives.
 */
    ygsid = create_graphicstyle_object(wks);

    NhlRLClear(prlist);
    NhlRLClear(srlist);

    x[0] = xmin;
    x[1] = xmax;
    y[0] = special_res->nglYRefLine;
    y[1] = special_res->nglYRefLine;

/*
 * Set some GraphicStyle resources. 
 */
    thickness = special_res->nglYRefLineThicknessF;
    color     = special_res->nglYRefLineColor;
    NhlRLSetInteger(srlist,"gsLineColor",color);
    NhlRLSetFloat(srlist,"gsLineThicknessF",thickness);
    NhlSetValues(ygsid,srlist);

/*
 * Set some Primitive resources. 
 */
    NhlRLSetFloatArray(prlist,"prXArray",   x, 2);
    NhlRLSetFloatArray(prlist,"prYArray",   y, 2);
    NhlRLSetInteger   (prlist,"prPolyType", NhlPOLYLINE);
    NhlRLSetInteger   (prlist,"prGraphicStyle", ygsid);

/*
 * Create the object and attach it to the XY plot.
 */
    yprim_object = (int*)malloc(sizeof(int));
    NhlCreate(yprim_object,"YRefPrimitive",NhlprimitiveClass,wks,prlist);
    NhlAddPrimitive(plot,*yprim_object,-1);
  }
  if(special_res->nglXRefLine != -999.) {
/*
 * Create graphic style object on which to draw primitives.
 */
    xgsid = create_graphicstyle_object(wks);

    NhlRLClear(prlist);
    NhlRLClear(srlist);

    x[0] = special_res->nglXRefLine;
    x[1] = special_res->nglXRefLine;
    y[0] = ymin;
    y[1] = ymax;

/*
 * Set some GraphicStyle resources. 
 */
    thickness = special_res->nglXRefLineThicknessF;
    color     = special_res->nglXRefLineColor;
    NhlRLSetInteger(srlist,"gsLineColor",color);
    NhlRLSetFloat(srlist,"gsLineThicknessF",thickness);
    NhlSetValues(xgsid,srlist);

/*
 * Set some Primitive resources. 
 */
    NhlRLSetFloatArray(prlist,"prXArray",   x, 2);
    NhlRLSetFloatArray(prlist,"prYArray",   y, 2);
    NhlRLSetInteger   (prlist,"prPolyType", NhlPOLYLINE);
    NhlRLSetInteger   (prlist,"prGraphicStyle", xgsid);

/*
 * Create the object and attach it to the XY plot.
 */
    xprim_object = (int*)malloc(sizeof(int));
    NhlCreate(xprim_object,"XRefPrimitive",NhlprimitiveClass,wks,prlist);
    NhlAddPrimitive(plot,*xprim_object,-1);
  }

}
/*
 * This function retrieves the current tickmark lengths and points
 * them outward.
 */

void point_tickmarks_out(int plot, ResInfo *res)
{
  int srlist, grlist;
  float xb_length, xt_length, yl_length, yr_length;
  float xb_mlength, xt_mlength, yl_mlength, yr_mlength;

  srlist = NhlRLCreate(NhlSETRL);
  grlist = NhlRLCreate(NhlGETRL);

  NhlRLClear(grlist);
  NhlRLGetFloat(grlist,"tmXBMajorLengthF", &xb_length);
  NhlRLGetFloat(grlist,"tmXTMajorLengthF", &xt_length);
  NhlRLGetFloat(grlist,"tmYLMajorLengthF", &yl_length);
  NhlRLGetFloat(grlist,"tmYRMajorLengthF", &yr_length);
  NhlRLGetFloat(grlist,"tmXBMinorLengthF", &xb_mlength);
  NhlRLGetFloat(grlist,"tmXTMinorLengthF", &xt_mlength);
  NhlRLGetFloat(grlist,"tmYLMinorLengthF", &yl_mlength);
  NhlRLGetFloat(grlist,"tmYRMinorLengthF", &yr_mlength);
  NhlGetValues(plot,grlist);

  NhlRLClear(srlist);

  NhlRLSetFloat(srlist, "tmXBMajorOutwardLengthF", xb_length);
  NhlRLSetFloat(srlist, "tmXTMajorOutwardLengthF", xt_length);
  NhlRLSetFloat(srlist, "tmYLMajorOutwardLengthF", yl_length);
  NhlRLSetFloat(srlist, "tmYRMajorOutwardLengthF", yr_length);
  NhlRLSetFloat(srlist, "tmXBMinorOutwardLengthF", xb_mlength);
  NhlRLSetFloat(srlist, "tmXTMinorOutwardLengthF", xt_mlength);
  NhlRLSetFloat(srlist, "tmYLMinorOutwardLengthF", yl_mlength);
  NhlRLSetFloat(srlist, "tmYRMinorOutwardLengthF", yr_mlength);

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
                  const char *type_x, int ndims_x, int *idsizes_x)
{
  ng_size_t *dsizes_x;

  dsizes_x = convert_dims(idsizes_x,ndims_x);

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
  free(dsizes_x);
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
  plot->labelbar   = NULL;
  plot->legend     = NULL;
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
  plot->nlabelbar   = 0;
  plot->nlegend     = 0;
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
 * routines (contour, map, vector, etc) or for the
 * primitive and text routines (polymarker, text, add text, 
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
  res->nglScale = 1;

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
  res->nglPanelCenter             = 1;
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
 * Special resources for panel figure strings.
 */
  res->nglPanelFigureStrings                    = NULL;
  res->nglPanelFigureStringsCount               = 0;
  res->nglPanelFigureStringsJust                = 8;   /* BottomRight */
  res->nglPanelFigureStringsOrthogonalPosF      = -999.;
  res->nglPanelFigureStringsParallelPosF        = -999.;
  res->nglPanelFigureStringsPerimOn             = 1;
  res->nglPanelFigureStringsBackgroundFillColor = 0;
  res->nglPanelFigureStringsFontHeightF         = -999.;

/*
 * Special resources for a panel labelbar.
 */
  res->nglPanelLabelBar                 = 0;  
  res->nglPanelLabelBarXF               = -999.;
  res->nglPanelLabelBarYF               = -999.;
  res->nglPanelLabelBarWidthF           = -999.;
  res->nglPanelLabelBarHeightF          = -999.;
  res->nglPanelLabelBarOrientation      = NhlHORIZONTAL;
  res->nglPanelLabelBarPerimOn          = 0;
  res->nglPanelLabelBarLabelAutoStride  = 1;
  res->nglPanelLabelBarAlignment        = NhlINTERIOREDGES;
  res->nglPanelLabelBarLabelFontHeightF = -999.;
  res->nglPanelLabelBarOrthogonalPosF   = -999.;
  res->nglPanelLabelBarParallelPosF     = -999.;
/*
 * Special resource to rename resource file.
 */
  res->nglAppResFileName                = NULL;

/*
 * Special resources for indicating the type of the axis. 
 * 0 = irregular, 1 = linear, 2 = log.
 */
  res->nglXAxisType =  0;
  res->nglYAxisType =  0;

/*
 * Special resources for drawing a vertical or horizontal X or Y
 * reference line.
 */
  res->nglXRefLine           = -999.;
  res->nglYRefLine           = -999.;
  res->nglXRefLineThicknessF = 1.;
  res->nglYRefLineThicknessF = 1.;
  res->nglXRefLineColor      = 1;
  res->nglYRefLineColor      = 1;
}

/*
 * This function maximizes and draws the plot, and advances the frame.
 */

void draw_and_frame(int wks, nglPlotId *plots, int nplots, int ispanel, 
                    nglRes *special_res)
{
  int i;

  if(special_res->nglMaximize) maximize_plots(wks, plots, nplots, 
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
  int app, field, rank, length[2];

/*
 * Retrieve application id.
 */
  app = NhlAppGetDefaultParentId();

/*
 * Create a scalar field object that will be used as the
 * dataset for the contour object.
 *
 * If xlen is -1, this means the data is 1D, and that (hopefully) the
 * user has also already set sfXArray and sfYArray to 1D arrays of 
 * the same length.
 */

  length[0] = ylen;
  if(xlen == -1) {
    rank      = 1;
    length[1] = ylen;
  }
  else {
    rank      = 2;
    length[1] = xlen;
  }
  set_resource("sfDataArray", sf_rlist, data, type_data, rank, length);

/*
 * Check for coordinate arrays.
 */
 
  if(is_ycoord) {
    set_resource("sfYArray", sf_rlist, ycoord, type_ycoord, 1, &length[0] );
  }

  if(is_xcoord) {
    set_resource("sfXArray", sf_rlist, xcoord, type_xcoord, 1, &length[1] );
  }

/*
 * Check for missing values.
 */
  if(is_missing_data) {
    length[0] = 1;
    set_resource("sfMissingValueV", sf_rlist, FillValue_data, type_data, 1, 
                 length);
  }

/*
 * Create the object.
 */
  if(rank == 2) {
    NhlCreate(&field,"field",NhlscalarFieldClass,app,sf_rlist);
  }
  else {
    NhlCreate(&field,"field",NhlmeshScalarFieldClass,app,sf_rlist);
  }
   
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
  int app, carray, length[2];

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

  set_resource("vfUDataArray", vf_rlist, u, type_u, 2, length );
  set_resource("vfVDataArray", vf_rlist, v, type_v, 2, length );

/*
 * Check for coordinate arrays.
 */

  if(is_ycoord) {
    set_resource("vfYArray", vf_rlist, ycoord, type_ycoord, 1, &length[0] );
  }

  if(is_xcoord) {
    set_resource("vfXArray", vf_rlist, xcoord, type_xcoord, 1, &length[1] );
  }

/*
 * Check for missing values.
 */
  length[0] = 1;
  if(is_missing_u) {
    set_resource("vfMissingUValueV", vf_rlist, FillValue_u, type_u, 1, 
                 &length[0] );
  }

  if(is_missing_v) {
    set_resource("vfMissingVValueV", vf_rlist, FillValue_v, type_v, 1, 
                 &length[0] );
  }

  NhlCreate(&field,"field",NhlvectorFieldClass,app,vf_rlist);

  return(field);
}


/*
 * This function uses the HLUs to create an Application object
 * and to open a workstation.
 */

int open_wks_wrap(const char *type, const char *name, ResInfo *wk_res,
                  ResInfo *ap_res, nglRes *special_res)
{
  int wks, len, tlen, wk_rlist, ap_rlist, grlist, check_orientation = 0;
  char *filename = (char *) NULL;
  int app;

  wk_rlist = wk_res->id;
  ap_rlist = ap_res->id;

/*
 * Initialize HLU library.
 */
  if(!nhl_initialized) {
    NhlInitialize();
  }

/*
 * Create Application object.
 */
  if(!is_res_set(ap_res,"appDefaultParent")) {
    NhlRLSetString(ap_rlist,"appDefaultParent","True");
  }
  if(!is_res_set(ap_res,"appUsrDir")) {
    NhlRLSetString(ap_rlist,"appUsrDir","./");
  }
  if(special_res->nglAppResFileName == NULL ||
     !strcmp(special_res->nglAppResFileName,"")) {
    NhlCreate(&app,name,NhlappClass,NhlDEFAULT_APP,ap_rlist);
  }
  else {
    NhlCreate(&app,special_res->nglAppResFileName,NhlappClass,
              NhlDEFAULT_APP,ap_rlist);
  }

/*
 * Load color maps. This is necessary for access to the color maps in
 * $NCARG_ROOT/lib/ncarg/colormaps, not for the 9 built-in color maps.
 */
  if(!nhl_initialized) {
    NhlPalLoadColormapFiles(NhlworkstationClass,True);
  }

/*
 * Start workstation code.
 */

  if(!strcmp(type,"x11") || !strcmp(type,"X11")) {
/*
 * Create an XWorkstation object.
 */
    if(!is_res_set(wk_res,"wkPause")) {
      NhlRLSetInteger(wk_rlist,"wkPause",True);
    }
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

    if(!is_res_set(wk_res,"wkMetaName")) {
      NhlRLSetString(wk_rlist,"wkMetaName",filename);
    }
    NhlCreate(&wks,"ncgm",NhlncgmWorkstationClass,NhlDEFAULT_APP,wk_rlist);
  }
  else if(!strcmp(type,"ps")   || !strcmp(type,"PS") || 
          !strcmp(type,"eps")  || !strcmp(type,"EPS") || 
          !strcmp(type,"epsi") || !strcmp(type,"EPSI")) {

    NhlRLSetString(wk_rlist,"wkPSFormat",(char*)type);
/*
 * Flag for whether we need to check for wkOrientation later.
 */
    check_orientation = 1;
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

    if(!is_res_set(wk_res,"wkPSFileName")) {
      NhlRLSetString(wk_rlist,"wkPSFileName",filename);
    }
    NhlCreate(&wks,type,NhlpsWorkstationClass,NhlDEFAULT_APP,wk_rlist);
  }
  else if(!strcmp(type,"pdf") || !strcmp(type,"PDF")) {
/*
 * Flag for whether we need to check for wkOrientation later.
 */
    check_orientation = 1;
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

    if(!is_res_set(wk_res,"wkPDFFileName")) {
      NhlRLSetString(wk_rlist,"wkPDFFileName",filename);
    }
    NhlCreate(&wks,"pdf",NhlpdfWorkstationClass,NhlDEFAULT_APP,wk_rlist);
  }
  else if(!strcmp(type,"newps") || !strcmp(type,"newpdf") || 
          !strcmp(type,"NEWPS") || !strcmp(type,"NEWPDF")) {
    NhlRLSetString(wk_rlist,"wkFormat",(char*)type);
/*
 * Flag for whether we need to check for wkOrientation later.
 */
    check_orientation = 1;

    len      = strlen(name);
    filename = (char *)calloc(len+1,sizeof(char));
    strncpy(filename,name,len);
    filename[len] = '\0';

    if(!is_res_set(wk_res,"wkFileName")) {
      NhlRLSetString(wk_rlist,"wkFileName",filename);
    }
    NhlCreate(&wks,type,NhlcairoPSPDFWorkstationClass,NhlDEFAULT_APP,wk_rlist);
  }
/*
 * Generate PNG file name.
 * Not yet available.
 */
  else if(!strcmp(type,"png")    || !strcmp(type,"PNG") ||
          !strcmp(type,"newpng") || !strcmp(type,"NEWPNG")) {
    len      = strlen(name);
    filename = (char *)calloc(len+1,sizeof(char));
    strncpy(filename,name,len);
    filename[len] = '\0';
    if(!is_res_set(wk_res,"wkFileName")) {
      NhlRLSetString(wk_rlist,"wkFileName",filename);
    }
    if(!is_res_set(wk_res,"wkFormat")) {
      NhlRLSetString(wk_rlist,"wkFormat","png");
    }
    NhlCreate(&wks,"png",NhlcairoImageWorkstationClass,NhlDEFAULT_APP,wk_rlist);
  }
  else {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"spread_colors: Invalid workstation type, must be 'x11', 'ncgm', 'ps', 'png', or 'pdf'\n");
  }

/*
 * Check if wkOrientation is set. If so, set a flag so that later on,
 * when we are maximizing the size of plot in the workstation, we use
 * the one set by wkOrientation. Setting nglPaperOrientation
 * will override wkOrientation.
 */
  if(check_orientation) {
    if(is_res_set(wk_res,"wkOrientation")) {
      grlist = NhlRLCreate(NhlGETRL);
      NhlRLClear(grlist);
      NhlRLGetInteger(grlist,"wkOrientation",&global_wk_orientation);
      NhlGetValues(wks,grlist);
    }
  }


/*
 * Clean up and return.
 */

  nhl_initialized = 1;
  if(filename != NULL) free(filename);
  return(wks);
}

/*
 * This function uses the HLUs to create a blank plot.
 */

nglPlotId blank_plot_wrap(int wks, ResInfo *blank_res, nglRes *special_res)
{
  int loglin, blank_rlist;
  nglPlotId plot;

/*
 * Set resource ids.
 */
  blank_rlist  = blank_res->id;

/*
 * Create loglin plot.
 */
  NhlCreate(&loglin,"blank",NhllogLinPlotClass,wks,blank_rlist);

/*
 * Make sure LogLin object is valid before moving on.
 */
  if(loglin > 0) {
/*
 * Make tickmarks and axis labels the same size.
 */
    if(special_res->nglScale) scale_plot(loglin,blank_res,0);

/*
 * Point tickmarks outward if requested specifically by user.
 */
    if(special_res->nglPointTickmarksOutward) {
      point_tickmarks_out(loglin,blank_res);
    }
  }
  else {
    loglin = -1;
  }
/*
 * Initialize plot ids.
 */
  initialize_ids(&plot);
  plot.base = (int *)malloc(sizeof(int));
  if(loglin > 0) {
    *(plot.base) = loglin;
    plot.nbase   = 1;
/*
 * Draw blank plot and advance frame.
 */
    if(special_res->nglDraw)  NhlDraw(loglin);
    if(special_res->nglFrame) NhlFrame(wks);
  }
  else {
/*
 * We have an invalid LogLin object!
 */
    *(plot.base) = -1;
  }
/*
 * Return.
 */
  return(plot);
}
  
/*
 * This function uses the HLUs to create a contour plot.
 */

nglPlotId contour_wrap(int wks, void *data, const char *type, int ylen,
                       int xlen, int is_ycoord, void *ycoord, 
                       const char *ycoord_type,int is_xcoord, 
                       void *xcoord, const char *xcoord_type,
                       int is_missing, void *FillValue, ResInfo *sf_res,
                       ResInfo *cn_res, ResInfo *tm_res, 
                       nglRes *special_res)
{
  nglPlotId plot;
  int field, contour;
  int sf_rlist, cn_rlist;

/*
 * Set resource ids.
 */
  sf_rlist = sf_res->id;
  cn_rlist = cn_res->id;

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
  if(special_res->nglScale) scale_plot(contour,cn_res,0);

/*
 * Point tickmarks outward if requested specifically by user.
 */
  if(special_res->nglPointTickmarksOutward) {
    point_tickmarks_out(contour,cn_res);
  }

/*
 * Initialize plot id structure.
 */
  initialize_ids(&plot);
  plot.sffield    = (int *)malloc(sizeof(int));
  plot.contour    = (int *)malloc(sizeof(int));
  plot.base       = (int *)malloc(sizeof(int));
  *(plot.sffield) = field;
  *(plot.contour) = contour;
  *(plot.base)    = contour;
  plot.nsffield   = 1;
  plot.ncontour   = 1;
  plot.nbase      = plot.ncontour;

/*
 * Overlay on an irregular plot object if it is desired to log
 * or linearize either of the axes.
 */

  if(special_res->nglXAxisType > 0 || special_res->nglYAxisType > 0) {
    overlay_on_irregular(wks,&plot,cn_res,tm_res,special_res);
  }

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

nglPlotId xy_wrap(int wks, void *x, void *y, const char *type_x,
                  const char *type_y, int ndims_x, int *dsizes_x,
                  int ndims_y, int *dsizes_y, 
                  int is_missing_x, int is_missing_y, 
                  void *FillValue_x, void *FillValue_y,
                  ResInfo *ca_res, ResInfo *xy_res, ResInfo *xyd_res,
                  nglRes *special_res)
{
  int cafield, xy, grlist, *xyds;
  ng_size_t num_dspec;
  nglPlotId plot;
  int ca_rlist, xy_rlist, xyd_rlist;

/*
 * Set resource ids.
 */
  ca_rlist  = ca_res->id;
  xy_rlist  = xy_res->id;
  xyd_rlist = xyd_res->id;

/*
 * Create a coord arrays object that will be used as the
 * dataset for the xy object.
 */

  cafield = coord_array(x, y, type_x, type_y, ndims_x, dsizes_x, 
                        ndims_y, dsizes_y, is_missing_x, is_missing_y, 
                        FillValue_x, FillValue_y, ca_rlist);
/*
 * Make sure CoordArray object is valid before moving on.
 */
  if(cafield > 0) {
/*
 * Assign the data object.
 */
    NhlRLSetInteger(xy_rlist,"xyCoordData",cafield);
/*
 * Create plot.
 */

    NhlCreate(&xy,"xy",NhlxyPlotClass,wks,xy_rlist);

/*
 * Make sure XyPlot object is valid before moving on.
 */
    if(xy > 0) {
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
 * Now apply the data spec resources, if DataSpec is a valid object.
 */
      if(*xyds > 0) NhlSetValues(*xyds,xyd_rlist);

/*
 * Make tickmarks and axis labels the same size.
 */
      if(special_res->nglScale) scale_plot(xy,xy_res,0);

/*
 * Add an X and/or Y reference line if requested.
 */
      if(special_res->nglYRefLine != -999 || 
         special_res->nglXRefLine != -999) {
        add_ref_line(wks,xy,special_res);
      }

/*
 * Point tickmarks outward if requested specifically by user.
 */
      if(special_res->nglPointTickmarksOutward) {
        point_tickmarks_out(xy,xy_res);
      }
    }
  }
  else {
    xy = -1;
  }
/*
 * Initialize plot ids.
 */
  initialize_ids(&plot);
  plot.cafield    = (int *)malloc(sizeof(int));
  plot.xydspec    = (int *)malloc(sizeof(int));
  plot.xy         = (int *)malloc(sizeof(int));
  plot.base       = (int *)malloc(sizeof(int));
  if(cafield > 0 && xy > 0) {
    *(plot.cafield) = cafield;
    *(plot.xydspec) = *xyds;
    *(plot.xy)      = xy;
    *(plot.base)    = xy;
    plot.ncafield   = 1;
    plot.nxydspec   = 1;
    plot.nxy        = 1;
    plot.nbase      = plot.nxy;

/*
 * Draw xy plot and advance frame.
 */
    draw_and_frame(wks, &plot, 1, 0, special_res);
  }
  else {
/*
 * We have an invalid XyPlot object!
 */
    *(plot.cafield) = -1;
    *(plot.xydspec) = -1;
    *(plot.xy)      = -1;
    *(plot.base)    = -1;
  }
/*
 * Return.
 */
  return(plot);
}


/*
 * This function uses the HLUs to create an XY plot.
 */

nglPlotId y_wrap(int wks, void *y, const char *type_y, int ndims_y, 
                 int *dsizes_y, int is_missing_y, void *FillValue_y,
                 ResInfo *ca_res, ResInfo *xy_res, ResInfo *xyd_res,
                 nglRes *special_res)
{
  nglPlotId xy;

/*
 * Call xy_wrap, only using NULLs for the X values.
 */
  xy = xy_wrap(wks, NULL, y, NULL, type_y, 0, NULL,
               ndims_y, &dsizes_y[0], 0, is_missing_y, NULL, 
               FillValue_y, ca_res, xy_res, xyd_res, special_res);
/*
 * Return.
 */
  return(xy);
}

/*
 * This function uses the HLUs to create a vector plot.
 */

nglPlotId vector_wrap(int wks, void *u, void *v, const char *type_u,
                      const char *type_v, int ylen, int xlen, 
                      int is_ycoord, void *ycoord, 
                      const char *type_ycoord, int is_xcoord, 
                      void *xcoord, const char *type_xcoord, 
                      int is_missing_u, int is_missing_v, 
                      void *FillValue_u, void *FillValue_v,
                      ResInfo *vf_res, ResInfo *vc_res, ResInfo *tm_res,
                      nglRes *special_res)
{
  int field, vector;
  nglPlotId plot;
  int vf_rlist, vc_rlist;

/*
 * Set resource ids.
 */
  vf_rlist = vf_res->id;
  vc_rlist = vc_res->id;

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
  if(special_res->nglScale) scale_plot(vector,vc_res,0);

/*
 * Point tickmarks outward if requested specifically by user.
 */
  if(special_res->nglPointTickmarksOutward) {
    point_tickmarks_out(vector,vc_res);
  }

/*
 * Initialize plot ids.
 */
  initialize_ids(&plot);
  plot.vffield    = (int *)malloc(sizeof(int));
  plot.vector     = (int *)malloc(sizeof(int));
  plot.base       = (int *)malloc(sizeof(int));
  *(plot.vffield) = field;
  *(plot.vector)  = vector;
  *(plot.base)    = vector;
  plot.nvffield   = 1;
  plot.nvector    = 1;
  plot.nbase      = plot.nvector;

/*
 * Overlay on an irregular plot object if it is desired to log
 * or linearize either of the axes.
 */

  if(special_res->nglXAxisType > 0 || special_res->nglYAxisType > 0) {
    overlay_on_irregular(wks,&plot,vc_res,tm_res,special_res);
  }

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

nglPlotId streamline_wrap(int wks, void *u, void *v, const char *type_u,
                          const char *type_v, int ylen, int xlen, 
                          int is_ycoord, void *ycoord, 
                          const char *type_ycoord, int is_xcoord, 
                          void *xcoord, const char *type_xcoord, 
                          int is_missing_u, int is_missing_v, 
                          void *FillValue_u, void *FillValue_v, 
                          ResInfo *vf_res, ResInfo *st_res, 
                          ResInfo *tm_res, nglRes *special_res)
{
  int field, streamline;
  int vf_rlist, st_rlist;
  nglPlotId plot;

/*
 * Set resource ids.
 */
  vf_rlist = vf_res->id;
  st_rlist = st_res->id;

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
  if(special_res->nglScale) scale_plot(streamline,st_res,0);

/*
 * Point tickmarks outward if requested specifically by user.
 */
  if(special_res->nglPointTickmarksOutward) {
    point_tickmarks_out(streamline,st_res);
  }

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield       = (int *)malloc(sizeof(int));
  plot.streamline    = (int *)malloc(sizeof(int));
  plot.base          = (int *)malloc(sizeof(int));
  *(plot.vffield)    = field;
  *(plot.streamline) = streamline;
  *(plot.base)       = streamline;
  plot.nvffield      = 1;
  plot.nstreamline   = 1;
  plot.nbase         = plot.nstreamline;

/*
 * Overlay on an irregular plot object if it is desired to log
 * or linearize either of the axes.
 */

  if(special_res->nglXAxisType > 0 || special_res->nglYAxisType > 0) {
    overlay_on_irregular(wks,&plot,st_res,tm_res,special_res);
  }

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

nglPlotId map_wrap(int wks, ResInfo *mp_res, nglRes *special_res)
{
  int map;
  nglPlotId plot;
  int mp_rlist;

/*
 * Set resource ids.
 */
  mp_rlist = mp_res->id;

/*
 * Create plot.
 */

  NhlCreate(&map,"map",NhlmapPlotClass,wks,mp_rlist);

/*
 * Scale the map only to get the axes titles the same size. The
 * tickmarks should already be okay.
 */ 
  if(special_res->nglScale) scale_plot(map,mp_res,1);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.map    = (int *)malloc(sizeof(int));
  plot.base   = (int *)malloc(sizeof(int));
  *(plot.map) = map;
  *(plot.base)= map;
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

nglPlotId contour_map_wrap(int wks, void *data, const char *type, 
                           int ylen, int xlen, int is_ycoord, 
                           void *ycoord, const char *ycoord_type,
                           int is_xcoord, void *xcoord, 
                           const char *xcoord_type, int is_missing, 
                           void *FillValue, ResInfo *sf_res, 
                           ResInfo *cn_res, ResInfo *mp_res,
                           nglRes *special_res)
{
  nglRes special_res2;
  nglPlotId contour, map, plot;
  int old_scale;
/*
 * Create contour plot. Be sure to copy over special resources, and
 * change some of them if necessary.
 *
 * Note that nglScale is set to 0 here, because we'll scale the
 * tickmarks, their labels, and the axis labels when we create 
 * the map.
 *
 * Also, XAxisType and YAxisType are set to 0 (IrregularAxis) to
 * ensure that one doesn't try to linearize or logize an axis
 * system that's about to be overlaid on a map.
 */
  special_res2              = *special_res;
  old_scale                 = special_res2.nglScale;  /* Save this value */
 
  special_res2.nglDraw      = 0;
  special_res2.nglFrame     = 0;
  special_res2.nglMaximize  = 0;
  special_res2.nglScale     = 0;
  special_res2.nglXAxisType = 0;
  special_res2.nglYAxisType = 0;

  contour = contour_wrap(wks, data, type, ylen, xlen,
                         is_ycoord, ycoord, ycoord_type,
                         is_xcoord, xcoord, xcoord_type,
                         is_missing, FillValue, sf_res, cn_res,
                         NULL, &special_res2);

/*
 * Create map plot.
 */
  special_res2.nglScale = old_scale;

  map = map_wrap(wks, mp_res, &special_res2);

/*
 * Overlay contour plot on map plot.
 */
  NhlAddOverlay(*(map.base),*(contour.base),-1);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.sffield    = (int *)malloc(sizeof(int));
  plot.contour    = (int *)malloc(sizeof(int));
  plot.map        = (int *)malloc(sizeof(int));
  plot.base       = (int *)malloc(sizeof(int));
  *(plot.sffield) = *(contour.sffield);
  *(plot.contour) = *(contour.base);
  *(plot.map)     = *(map.base);
  *(plot.base)    = *(map.base);
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

nglPlotId vector_map_wrap(int wks, void *u, void *v, const char *type_u,
                          const char *type_v, int ylen, int xlen, 
                          int is_ycoord, void *ycoord, 
                          const char *type_ycoord, int is_xcoord, 
                          void *xcoord, const char *type_xcoord, 
                          int is_missing_u, int is_missing_v, 
                          void *FillValue_u, void *FillValue_v,
                          ResInfo *vf_res, ResInfo *vc_res,
                          ResInfo *mp_res, nglRes *special_res)
{
  nglRes special_res2;
  nglPlotId vector, map, plot;
  int old_scale;

/*
 * Create vector plot. Be sure to copy over special resources.
 *
 * Note that nglScale is set to 0 here, because we'll scale the
 * tickmarks, their labels, and the axis labels when we create 
 * the map.
 *
 * Also, XAxisType and YAxisType are set to 0 (IrregularAxis) to
 * ensure that one doesn't try to linearize or logize an axis
 * system that's about to be overlaid on a map.
 */
  special_res2              = *special_res;
  old_scale                 = special_res2.nglScale;  /* Save this value */

  special_res2.nglDraw      = 0;
  special_res2.nglFrame     = 0;
  special_res2.nglMaximize  = 0;
  special_res2.nglScale     = 0;
  special_res2.nglXAxisType = 0;
  special_res2.nglYAxisType = 0;

  vector = vector_wrap(wks, u, v, type_u, type_v, ylen, xlen, is_ycoord,
                       ycoord, type_ycoord, is_xcoord, xcoord, 
                       type_xcoord, is_missing_u, is_missing_v, 
                       FillValue_u, FillValue_v, vf_res, vc_res,
                       NULL, &special_res2);

/*
 * Create map plot.
 */
  special_res2.nglScale = old_scale;

  map = map_wrap(wks, mp_res, &special_res2);

/*
 * Overlay vector plot on map plot.
 */
  NhlAddOverlay(*(map.base),*(vector.base),-1);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield    = (int *)malloc(sizeof(int));
  plot.vector     = (int *)malloc(sizeof(int));
  plot.map        = (int *)malloc(sizeof(int));
  plot.base       = (int *)malloc(sizeof(int));
  *(plot.vffield) = *(vector.vffield);
  *(plot.vector)  = *(vector.base);
  *(plot.map)     = *(map.base);
  *(plot.base)    = *(map.base);
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

nglPlotId streamline_map_wrap(int wks, void *u, void *v, 
                              const char *type_u, const char *type_v, 
                              int ylen, int xlen, int is_ycoord, 
                              void *ycoord, const char *type_ycoord, 
                              int is_xcoord, void *xcoord, 
                              const char *type_xcoord, int is_missing_u,
                              int is_missing_v, void *FillValue_u, 
                              void *FillValue_v, ResInfo *vf_res, 
                              ResInfo *st_res, ResInfo *mp_res,
                              nglRes *special_res)
{
  nglRes special_res2;
  nglPlotId streamline, map, plot;
  int old_scale;

/*
 * Create streamline plot.
 *
 * Note that nglScale is set to 0 here, because we'll scale the
 * tickmarks, their labels, and the axis labels when we create 
 * the map.
 */
  special_res2              = *special_res;
  old_scale                 = special_res2.nglScale;  /* Save this value */

  special_res2.nglDraw      = 0;
  special_res2.nglFrame     = 0;
  special_res2.nglMaximize  = 0;
  special_res2.nglScale     = 0;
  special_res2.nglXAxisType = 0;
  special_res2.nglYAxisType = 0;

  streamline = streamline_wrap(wks, u, v, type_u, type_v, ylen, xlen, 
                               is_ycoord, ycoord, type_ycoord, 
                               is_xcoord, xcoord, type_xcoord, 
                               is_missing_u, is_missing_v, FillValue_u, 
                               FillValue_v, vf_res, st_res, NULL,
                               &special_res2);

/*
 * Create map plot.
 */
  special_res2.nglScale = old_scale;

  map = map_wrap(wks, mp_res, &special_res2);

/*
 * Overlay streamline plot on map plot.
 */
  NhlAddOverlay(*(map.base),*(streamline.base),-1);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield       = (int *)malloc(sizeof(int));
  plot.streamline    = (int *)malloc(sizeof(int));
  plot.map           = (int *)malloc(sizeof(int));
  plot.base          = (int *)malloc(sizeof(int));
  *(plot.vffield)    = *(streamline.vffield);
  *(plot.streamline) = *(streamline.base);
  *(plot.map)        = *(map.base);
  *(plot.base)       = *(map.base);

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
 * This function uses the HLUs to create a streamline plot colored by
 * a scalar field.
 */

nglPlotId streamline_scalar_wrap(int wks, void *u, void *v, void *t, 
                             const char *type_u, const char *type_v, 
                             const char *type_t, int ylen, int xlen, 
                             int is_ycoord, void *ycoord, 
                             const char *type_ycoord, int is_xcoord, 
                             void *xcoord, const char *type_xcoord, 
                             int is_missing_u, int is_missing_v, 
                             int is_missing_t, void *FillValue_u, 
                             void *FillValue_v, void *FillValue_t,
                             ResInfo *vf_res, ResInfo *sf_res,
                             ResInfo *st_res, ResInfo *tm_res, 
                             nglRes *special_res)
{
  int vffield, sffield, streamline;
  nglPlotId plot;
  int vf_rlist, sf_rlist, st_rlist; 

/*
 * Set resource ids.
 */
  vf_rlist  = vf_res->id;
  sf_rlist  = sf_res->id;
  st_rlist  = st_res->id;

/*
 * Create vector and scalar field objects that will be used as the
 * datasets for the streamline object.
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
 * Assign the data objects and create streamline object.
 */
  NhlRLSetInteger(st_rlist, "stVectorFieldData",    vffield);
  NhlRLSetInteger(st_rlist, "stScalarFieldData",    sffield);
  if(!is_res_set(st_res," stUseScalarArray")) {
    NhlRLSetString (st_rlist, "stUseScalarArray",     "True");
  }

  NhlCreate(&streamline,"streamline",NhlstreamlinePlotClass,wks,st_rlist);

/*
 * If the streamlines are being colored, span the full colormap for the colors.
 */
  if(special_res->nglSpreadColors)  {
    spread_colors(wks, streamline, special_res->nglSpreadColorStart,
                  special_res->nglSpreadColorEnd, "stLevelCount", 
                  "stLevelColors",special_res->nglDebug);
  }

/*
 * Point tickmarks outward if requested specifically by user.
 */
  if(special_res->nglPointTickmarksOutward) {
    point_tickmarks_out(streamline,st_res);
  }

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield       = (int *)malloc(sizeof(int));
  plot.sffield       = (int *)malloc(sizeof(int));
  plot.streamline    = (int *)malloc(sizeof(int));
  plot.base          = (int *)malloc(sizeof(int));
  *(plot.vffield)    = vffield;
  *(plot.sffield)    = sffield;
  *(plot.streamline) = streamline;
  *(plot.base)       = streamline;
  plot.nvffield      = 1;
  plot.nsffield      = 1;
  plot.nstreamline   = 1;
  plot.nbase         = plot.nstreamline;

/*
 * Overlay on an irregular plot object if it is desired to log
 * or linearize either of the axes.
 */

  if(special_res->nglXAxisType > 0 || special_res->nglYAxisType > 0) {
    overlay_on_irregular(wks,&plot,st_res,tm_res,special_res);
  }

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
 * This function uses the HLUs to create a streamline plot colored by
 * a scalar field overlaid on a map.
 */

nglPlotId streamline_scalar_map_wrap(int wks, void *u, void *v, void *t, 
                                 const char *type_u, const char *type_v, 
                                 const char *type_t, int ylen, int xlen, 
                                 int is_ycoord, void *ycoord, 
                                 const char *type_ycoord, int is_xcoord, 
                                 void *xcoord, const char *type_xcoord, 
                                 int is_missing_u, int is_missing_v, 
                                 int is_missing_t, void *FillValue_u, 
                                 void *FillValue_v, void *FillValue_t,
                                 ResInfo *vf_res, ResInfo *sf_res,
                                 ResInfo *st_res, ResInfo *mp_res,
                                 nglRes *special_res)
{
  nglRes special_res2;
  nglPlotId plot, streamline, map;
  int old_scale;

/*
 * Create streamline plot.
 *
 * Note that nglScale is set to 0 here, because we'll scale the
 * tickmarks, their labels, and the axis labels when we create 
 * the map.
 *
 * Also, XAxisType and YAxisType are set to 0 (IrregularAxis) to
 * ensure that one doesn't try to linearize or logize an axis
 * system that's about to be overlaid on a map.
 *
 * Also, XAxisType and YAxisType are set to 0 (IrregularAxis) to
 * ensure that one doesn't try to linearize or logize an axis
 * system that's about to be overlaid on a map.
 */

  special_res2              = *special_res;
  old_scale                 = special_res2.nglScale;  /* Save this value */

  special_res2.nglDraw      = 0;
  special_res2.nglFrame     = 0;
  special_res2.nglMaximize  = 0;
  special_res2.nglScale     = 0;
  special_res2.nglXAxisType = 0;
  special_res2.nglYAxisType = 0;

  streamline = streamline_scalar_wrap(wks, u, v, t, type_u, type_v, type_t,
                                      ylen, xlen, is_ycoord, ycoord, 
                                      type_ycoord, is_xcoord, xcoord, 
                                      type_xcoord, is_missing_u, is_missing_v, 
                                      is_missing_t, FillValue_u, FillValue_v, 
                                      FillValue_t, vf_res, sf_res, st_res, 
                                      NULL, &special_res2);

/*
 * Create map plot.
 */
  special_res2.nglScale = old_scale;

  map = map_wrap(wks, mp_res, &special_res2);

/*
 * Overlay streamline plot on map plot.
 */
  NhlAddOverlay(*(map.base),*(streamline.base),-1);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield       = (int *)malloc(sizeof(int));
  plot.sffield       = (int *)malloc(sizeof(int));
  plot.streamline    = (int *)malloc(sizeof(int));
  plot.map           = (int *)malloc(sizeof(int));
  plot.base          = (int *)malloc(sizeof(int));
  *(plot.vffield)    = *(streamline.vffield);
  *(plot.sffield)    = *(streamline.sffield);
  *(plot.streamline) = *(streamline.base);
  *(plot.map)        = *(map.base);
  *(plot.base)       = *(map.base);

  plot.nvffield    = 1;
  plot.nsffield    = 1;
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

nglPlotId vector_scalar_wrap(int wks, void *u, void *v, void *t, 
                             const char *type_u, const char *type_v, 
                             const char *type_t, int ylen, int xlen, 
                             int is_ycoord, void *ycoord, 
                             const char *type_ycoord, int is_xcoord, 
                             void *xcoord, const char *type_xcoord, 
                             int is_missing_u, int is_missing_v, 
                             int is_missing_t, void *FillValue_u, 
                             void *FillValue_v, void *FillValue_t,
                             ResInfo *vf_res, ResInfo *sf_res,
                             ResInfo *vc_res, ResInfo *tm_res, 
                             nglRes *special_res)
{
  int vffield, sffield, vector;
  nglPlotId plot;
  int vf_rlist, sf_rlist, vc_rlist; 

/*
 * Set resource ids.
 */
  vf_rlist  = vf_res->id;
  sf_rlist  = sf_res->id;
  vc_rlist  = vc_res->id;

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
  if(!is_res_set(vc_res,"vcUseScalarArray")) {
    NhlRLSetString (vc_rlist, "vcUseScalarArray",     "True");
  }
  if(!is_res_set(vc_res,"vcMonoLineArrowColor")) {
    NhlRLSetString (vc_rlist, "vcMonoLineArrowColor", "False");
  }
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
 * Point tickmarks outward if requested specifically by user.
 */
  if(special_res->nglPointTickmarksOutward) {
    point_tickmarks_out(vector,vc_res);
  }

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield    = (int *)malloc(sizeof(int));
  plot.sffield    = (int *)malloc(sizeof(int));
  plot.vector     = (int *)malloc(sizeof(int));
  plot.base       = (int *)malloc(sizeof(int));
  *(plot.vffield) = vffield;
  *(plot.sffield) = sffield;
  *(plot.vector)  = vector;
  *(plot.base)    = vector;
  plot.nvffield = 1;
  plot.nsffield = 1;
  plot.nvector  = 1;
  plot.nbase    = plot.nvector;

/*
 * Overlay on an irregular plot object if it is desired to log
 * or linearize either of the axes.
 */

  if(special_res->nglXAxisType > 0 || special_res->nglYAxisType > 0) {
    overlay_on_irregular(wks,&plot,vc_res,tm_res,special_res);
  }

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

nglPlotId vector_scalar_map_wrap(int wks, void *u, void *v, void *t, 
                                 const char *type_u, const char *type_v, 
                                 const char *type_t, int ylen, int xlen, 
                                 int is_ycoord, void *ycoord, 
                                 const char *type_ycoord, int is_xcoord, 
                                 void *xcoord, const char *type_xcoord, 
                                 int is_missing_u, int is_missing_v, 
                                 int is_missing_t, void *FillValue_u, 
                                 void *FillValue_v, void *FillValue_t,
                                 ResInfo *vf_res, ResInfo *sf_res,
                                 ResInfo *vc_res, ResInfo *mp_res,
                                 nglRes *special_res)
{
  nglRes special_res2;
  nglPlotId plot, vector, map;
  int old_scale;

/*
 * Create vector plot.
 *
 * Note that nglScale is set to 0 here, because we'll scale the
 * tickmarks, their labels, and the axis labels when we create 
 * the map.
 *
 * Also, XAxisType and YAxisType are set to 0 (IrregularAxis) to
 * ensure that one doesn't try to linearize or logize an axis
 * system that's about to be overlaid on a map.
 *
 * Also, XAxisType and YAxisType are set to 0 (IrregularAxis) to
 * ensure that one doesn't try to linearize or logize an axis
 * system that's about to be overlaid on a map.
 */

  special_res2              = *special_res;
  old_scale                 = special_res2.nglScale;  /* Save this value */

  special_res2.nglDraw      = 0;
  special_res2.nglFrame     = 0;
  special_res2.nglMaximize  = 0;
  special_res2.nglScale     = 0;
  special_res2.nglXAxisType = 0;
  special_res2.nglYAxisType = 0;

  vector = vector_scalar_wrap(wks, u, v, t, type_u, type_v, type_t,
                              ylen, xlen, is_ycoord, ycoord, 
                              type_ycoord, is_xcoord, xcoord, 
                              type_xcoord, is_missing_u, is_missing_v, 
                              is_missing_t, FillValue_u, FillValue_v, 
                              FillValue_t, vf_res, sf_res, vc_res, 
                              NULL, &special_res2);

/*
 * Create map plot.
 */
  special_res2.nglScale = old_scale;

  map = map_wrap(wks, mp_res, &special_res2);

/*
 * Overlay vector plot on map plot.
 */
  NhlAddOverlay(*(map.base),*(vector.base),-1);

/*
 * Set up plot id structure to return.
 */
  initialize_ids(&plot);
  plot.vffield    = (int *)malloc(sizeof(int));
  plot.sffield    = (int *)malloc(sizeof(int));
  plot.vector     = (int *)malloc(sizeof(int));
  plot.map        = (int *)malloc(sizeof(int));
  plot.base       = (int *)malloc(sizeof(int));
  *(plot.vffield) = *(vector.vffield);
  *(plot.sffield) = *(vector.sffield);
  *(plot.vector)  = *(vector.base);
  *(plot.map)     = *(map.base);
  *(plot.base)    = *(map.base);

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


nglPlotId text_ndc_wrap(int wks, char* string, void *x, void *y,
                        const char *type_x, const char *type_y,
                        ResInfo *txres, nglRes *special_res)
{
  int text, length[1];
  nglPlotId plot;
  int tx_rlist;

/*
 * Set resource ids.
 */
  tx_rlist = txres->id;

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


nglPlotId text_wrap(int wks, nglPlotId *plot, char* string, void *x, 
                    void *y, const char *type_x, const char *type_y,
                    ResInfo *txres, nglRes *special_res)
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
    printf("text: string = %s x = %g y = %g xndc = %g yndc = %g\n", 
           string, *xf, *yf, xndc, yndc);
  }

  text = text_ndc_wrap(wks, string, &xndc, &yndc, "float", "float",
                       txres, special_res);

/*
 * Return.
 */
  return(text);
}

/*
 * Routine for drawing a random labelbar in NDC space.
 */
nglPlotId labelbar_ndc_wrap(int wks, int nbox, NhlString *labels,
                            int nlabels, void *x, void *y,
                            const char *type_x, const char *type_y,
                            ResInfo *lb_res, nglRes *special_res)
{
  int lb_rlist, *labelbar_object;
  nglPlotId labelbar;
  float *xf, *yf;

/*
 * Set resource ids.
 */
  lb_rlist = lb_res->id;
  
/*
 * Determine if we need to convert x and/or y.
 */
  xf  = coerce_to_float(x,type_x,1);
  yf  = coerce_to_float(y,type_y,1);

/*
 * Allocate a variable to hold the labelbar object, and create it.
 */
  labelbar_object = (int*)malloc(sizeof(int));

  NhlRLSetFloat(lb_rlist,"vpXF",*xf);
  NhlRLSetFloat(lb_rlist,"vpYF",*yf);
  NhlRLSetStringArray(lb_rlist,"lbLabelStrings",labels,(ng_size_t)nlabels);
  NhlRLSetInteger(lb_rlist,"lbBoxCount",nbox);
  NhlCreate(labelbar_object,"Labelbar",NhllabelBarClass,wks,lb_rlist);

  if(special_res->nglDraw)  NhlDraw(*labelbar_object);
  if(special_res->nglFrame) NhlFrame(wks);

/*
 * Free up memory.
 */
  if(strcmp(type_x,"float")) free(xf);
  if(strcmp(type_y,"float")) free(yf);
    
/*
 * Set up plot id structure to return.
 */
  initialize_ids(&labelbar);
  labelbar.labelbar  = labelbar_object;
  labelbar.base      = labelbar.labelbar;
  labelbar.nlabelbar = 1;
  labelbar.nbase     = 1;

/*
 * Return.
 */
  return(labelbar);
}

/*
 * Routine for drawing a random legend in NDC space.
 */
nglPlotId legend_ndc_wrap(int wks, int nitems, NhlString *labels,
                          int nlabels, void *x, void *y,
                          const char *type_x, const char *type_y,
                          ResInfo *lg_res, nglRes *special_res)
{
  int lg_rlist, *legend_object;
  nglPlotId legend;
  float *xf, *yf;

/*
 * Set resource ids.
 */
  lg_rlist = lg_res->id;

/*
 * Determine if we need to convert x and/or y.
 */
  xf  = coerce_to_float(x,type_x,1);
  yf  = coerce_to_float(y,type_y,1);

/*
 * Allocate a variable to hold the legend object, and create it.
 */
  legend_object = (int*)malloc(sizeof(int));

  NhlRLSetFloat(lg_rlist,"vpXF",*xf);
  NhlRLSetFloat(lg_rlist,"vpYF",*yf);
  NhlRLSetStringArray(lg_rlist,"lgLabelStrings",labels,(ng_size_t)nlabels);
  NhlRLSetInteger(lg_rlist,"lgItemCount",nitems);
  NhlCreate(legend_object,"Legend",NhllegendClass,wks,lg_rlist);

  if(special_res->nglDraw)  NhlDraw(*legend_object);
  if(special_res->nglFrame) NhlFrame(wks);

/*
 * Free up memory.
 */
  if(strcmp(type_x,"float")) free(xf);
  if(strcmp(type_y,"float")) free(yf);
    
/*
 * Set up plot id structure to return.
 */
  initialize_ids(&legend);
  legend.legend  = legend_object;
  legend.base    = legend.legend;
  legend.nlegend = 1;
  legend.nbase   = 1;

/*
 * Return.
 */
  return(legend);
}



/*
 * Routine for drawing any kind of primitive in NDC or data space.
 */
void poly_wrap(int wks, nglPlotId *plot, void *x, void *y, 
               const char *type_x, const char *type_y, int len,
               int is_missing_x, int is_missing_y, void *FillValue_x, 
               void *FillValue_y, NhlPolyType polytype, ResInfo *gs_res,
               nglRes *special_res)
{
  int i, gsid, newlen, *indices, ibeg, iend, nlines, color;
  int srlist, grlist;
  float *xf, *yf, *xfnew, *yfnew, *xfmsg, *yfmsg, thickness;
  int gs_rlist;

/*
 * Set resource ids.
 */
  gs_rlist = gs_res->id;

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
          NhlRLGetFloat(grlist,"gsLineThicknessF",&thickness);
          NhlGetValues(gsid,grlist);
          
          NhlRLClear(srlist);
          NhlRLSetInteger(srlist,"gsMarkerColor",color);
          NhlRLSetFloat(srlist,"gsMarkerThicknessF",thickness);
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
nglPlotId add_poly_wrap(int wks, nglPlotId *plot, void *x, void *y,
                        const char *type_x, const char *type_y, int len, 
                        int is_missing_x, int is_missing_y,  int isndc,
                        void *FillValue_x, void *FillValue_y,
                        NhlPolyType polytype, ResInfo *gs_res, 
                        nglRes *special_res)
{
  int *primitive_object, gsid, pr_rlist;
  int i, newlen, *indices, nlines, npoly, ibeg, iend, npts;
  float *xf, *yf, *xfnew, *yfnew, *xfmsg, *yfmsg;
  char *astring;
  nglPlotId poly;
  int gs_rlist, srlist, grlist, canvas, color;
  float vpx, vpy, vpw, vph, thickness;

/*
 * Set resource ids.
 */
  gs_rlist = gs_res->id;

/*
 * Create resource list for primitive object.
 */

  pr_rlist = NhlRLCreate(NhlSETRL);

/*
 * If in NDC space, make sure X and Y values are in the range 0 to 1 AND
 * within the viewport of the polot.
 */
  if(isndc) {
    grlist = NhlRLCreate(NhlGETRL);
    NhlRLClear(grlist);
    NhlRLGetFloat(grlist,"vpHeightF",&vph);
    NhlRLGetFloat(grlist,"vpWidthF", &vpw);
    NhlRLGetFloat(grlist,"vpXF",     &vpx);
    NhlRLGetFloat(grlist,"vpYF",     &vpy);
    NhlGetValues(*(plot->base),grlist);

    srlist = NhlRLCreate(NhlSETRL);
    NhlRLClear(srlist);

    NhlRLSetInteger(srlist,"tfDoNDCOverlay",1);
    NhlRLSetFloat(srlist,"trXMinF",  vpx);
    NhlRLSetFloat(srlist,"trXMaxF",  vpx+vpw);
    NhlRLSetFloat(srlist,"trYMaxF",  vpy);
    NhlRLSetFloat(srlist,"trYMinF",  vpy-vph);
    NhlCreate(&canvas,"ndc_canvas",NhllogLinPlotClass,wks,srlist);
  }

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
    NhlRLSetFloatArray(pr_rlist,"prXArray",       &xfnew[0], (ng_size_t)newlen);
    NhlRLSetFloatArray(pr_rlist,"prYArray",       &yfnew[0], (ng_size_t)newlen);
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

    if(isndc) {
      NhlAddPrimitive(canvas,*primitive_object,-1);
      NhlAddOverlay(*(plot->base),canvas,-1);
    }
    else {
      NhlAddPrimitive(*(plot->base),*primitive_object,-1);
    }
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
          grlist = NhlRLCreate(NhlGETRL);

          NhlRLClear(grlist);
          NhlRLGetInteger(grlist,"gsLineColor",&color);
          NhlRLGetFloat(grlist,"gsLineThicknessF",&thickness);
          NhlGetValues(gsid,grlist);
/*
 * Create primitive object.
 */
          NhlRLSetFloat  (pr_rlist,"prXArray",       xf[ibeg]);
          NhlRLSetFloat  (pr_rlist,"prYArray",       yf[ibeg]);
          NhlRLSetInteger(pr_rlist,"prPolyType",     NhlPOLYMARKER);
          NhlRLSetInteger(pr_rlist,"prGraphicStyle", gsid);
          NhlCreate(&primitive_object[i],astring,NhlprimitiveClass,wks,
                    pr_rlist);

          srlist = NhlRLCreate(NhlSETRL);
          NhlRLClear(srlist);
          NhlRLSetInteger(srlist,"gsMarkerColor",color);
          NhlRLSetFloat(srlist,"gsMarkerThicknessF",thickness);
          NhlSetValues(gsid,srlist);
          
        }
        else {
          npts = iend - ibeg + 1;
/*
 * We have more than one point, so create a polyline.
 */

          NhlRLSetFloatArray(pr_rlist,"prXArray",       &xf[ibeg], npts);
          NhlRLSetFloatArray(pr_rlist,"prYArray",       &yf[ibeg], npts);
          NhlRLSetInteger   (pr_rlist,"prPolyType",     polytype);
          NhlRLSetInteger   (pr_rlist,"prGraphicStyle", gsid);
        }

/*
 * Create the polyline or marker, and  attach it to the plot.
 */
        NhlCreate(&primitive_object[i],"Primitive",NhlprimitiveClass,
                  wks,pr_rlist);

	if(isndc) {
	  NhlAddPrimitive(canvas,primitive_object[i],-1);
	}
	else {
	  NhlAddPrimitive(*(plot->base),primitive_object[i],-1);
	}
      }
    }
    else {
/*
 * Create a NULL primitive object.
 */
      if(isndc) {
	NhlAddOverlay(*(plot->base),canvas,-1);
      }
      else {
	primitive_object = (int*)malloc(sizeof(int));
      }
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
void polymarker_ndc_wrap(int wks, void *x, void *y, const char *type_x, 
                         const char *type_y,  int len,
                         int is_missing_x, int is_missing_y, 
                         void *FillValue_x, void *FillValue_y, 
                         ResInfo *gs_res, nglRes *special_res)
{
  nglPlotId plot;

  initialize_ids(&plot);
  poly_wrap(wks, &plot, x, y, type_x, type_y, len, is_missing_x, 
            is_missing_y, FillValue_x, FillValue_y, NhlPOLYMARKER, 
            gs_res,special_res);
}


/*
 * Routine for drawing lines in NDC space.
 */
void polyline_ndc_wrap(int wks, void *x, void *y, const char *type_x,
                       const char *type_y, int len,
                       int is_missing_x, int is_missing_y, 
                       void *FillValue_x, void *FillValue_y, 
                       ResInfo *gs_res, nglRes *special_res)
{
  nglPlotId plot;

  initialize_ids(&plot);
  poly_wrap(wks, &plot, x, y, type_x, type_y, len, is_missing_x, 
            is_missing_y, FillValue_x, FillValue_y, NhlPOLYLINE, 
            gs_res, special_res);
}


/*
 * Routine for drawing polygons in NDC space.
 */
void polygon_ndc_wrap(int wks, void *x, void *y, const char *type_x, 
                      const char *type_y, int len,
                      int is_missing_x, int is_missing_y, 
                      void *FillValue_x, void *FillValue_y, 
                      ResInfo *gs_res, nglRes *special_res)
{
  nglPlotId plot;

  initialize_ids(&plot);
  poly_wrap(wks, &plot, x, y, type_x, type_y, len, is_missing_x,
            is_missing_y, FillValue_x, FillValue_y, NhlPOLYGON, 
            gs_res, special_res);
}

/*
 * Routine for drawing markers in data space.
 */
void polymarker_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                     const char *type_x, const char *type_y, int len,
                     int is_missing_x, int is_missing_y, 
                     void *FillValue_x, void *FillValue_y, 
                     ResInfo *gs_res, nglRes *special_res)
{
  poly_wrap(wks,plot,x,y,type_x,type_y,len,is_missing_x,is_missing_y,
            FillValue_x,FillValue_y,NhlPOLYMARKER,gs_res,special_res);
}


/*
 * Routine for drawing lines in data space.
 */
void polyline_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                   const char *type_x, const char *type_y, int len, 
                   int is_missing_x, int is_missing_y, 
                   void *FillValue_x, void *FillValue_y, 
                   ResInfo *gs_res, nglRes *special_res)
{
  poly_wrap(wks,plot,x,y,type_x,type_y,len,is_missing_x,is_missing_y,
            FillValue_x,FillValue_y,NhlPOLYLINE,gs_res,special_res);
}

/*
 * Routine for drawing polygons in data space.
 */
void polygon_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                  const char *type_x, const char *type_y, int len, 
                  int is_missing_x, int is_missing_y, 
                  void *FillValue_x, void *FillValue_y, 
                  ResInfo *gs_res, nglRes *special_res)
{
  poly_wrap(wks, plot, x, y, type_x, type_y, len, is_missing_x, 
            is_missing_y, FillValue_x, FillValue_y, NhlPOLYGON, 
            gs_res, special_res);
}

/*
 * Routine for adding polylines to a plot (in the plot's data space).
 */
nglPlotId add_polyline_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                            const char *type_x, const char *type_y,
                            int len, int is_missing_x, int is_missing_y, 
                            void *FillValue_x, void *FillValue_y, 
                            ResInfo *gs_res, nglRes *special_res)
{
  nglPlotId poly;

  poly = add_poly_wrap(wks, plot, x, y, type_x, type_y, len, 
                       is_missing_x, is_missing_y, 0, FillValue_x, 
                       FillValue_y, NhlPOLYLINE, gs_res, 
                       special_res);
/*
 * Return.
 */
  return(poly);

}


/*
 * Routine for adding polymarkers to a plot (in the plot's data space).
 */
nglPlotId add_polymarker_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                              const char *type_x, const char *type_y,
                              int len, int is_missing_x,
                              int is_missing_y, void *FillValue_x,
                              void *FillValue_y, ResInfo *gs_res, 
                              nglRes *special_res)
{
  nglPlotId poly;

  poly = add_poly_wrap(wks, plot, x, y, type_x, type_y, len, 
                       is_missing_x, is_missing_y, 0, FillValue_x, 
                       FillValue_y, NhlPOLYMARKER, gs_res, 
                       special_res);
/*
 * Return.
 */
  return(poly);

}


/*
 * Routine for adding polygons to a plot (in the plot's data space).
 */
nglPlotId add_polygon_wrap(int wks, nglPlotId *plot, void *x, void *y, 
                           const char *type_x, const char *type_y, 
                           int len, int is_missing_x, int is_missing_y, 
                           void *FillValue_x, void *FillValue_y, 
                           ResInfo *gs_res, nglRes *special_res)
{
  nglPlotId poly;

  poly = add_poly_wrap(wks, plot, x, y, type_x, type_y, len, 
                       is_missing_x, is_missing_y, 0, FillValue_x, 
                       FillValue_y, NhlPOLYGON, gs_res, 
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
nglPlotId add_text_wrap(int wks, nglPlotId *plot, char *string, void *x,
                        void *y, const char *type_x, const char *type_y,
                        ResInfo *tx_res, ResInfo *am_res,
                        nglRes *special_res)
{
  int i, srlist, grlist, text, just;
  ng_size_t num_annos;
  int *anno_views, *anno_mgrs, *new_anno_views;
  float *xf, *yf;
  nglPlotId annos;
  int tx_rlist, am_rlist;

/*
 * Set resource ids.
 */
  tx_rlist = tx_res->id;
  am_rlist = am_res->id;

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
void draw_colormap_wrap(int wks)
{
  int i, j, k, ii, jj, ibox, nrows, ncols, ncolors, maxcols, ntotal, offset;
  int grlist, sr_list, tx_rlist, ln_rlist, gn_rlist;
  int reset_colormap, cmap_ndims;
  ng_size_t *cmap_dimsizes;
  float width, height, *xpos, *ypos, xbox[5], ybox[5], xbox2[5], ybox2[5];
  float txpos, typos;
  float *cmap, *cmapnew, font_height, font_space;
  char tmpstr[5];
  nglRes special_res2;
  ResInfo gn_res, ln_res, tx_res;

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
  sr_list = NhlRLCreate(NhlSETRL);

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
    NhlRLClear(sr_list);
    NhlRLSetMDFloatArray(sr_list,"wkColorMap",&cmapnew[0],2,cmap_dimsizes);
    NhlSetValues(wks,sr_list);
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
  tx_res.id = tx_rlist = NhlRLCreate(NhlSETRL);
  ln_res.id = ln_rlist = NhlRLCreate(NhlSETRL);
  gn_res.id = gn_rlist = NhlRLCreate(NhlSETRL);

/*
 * Set some text resources. 
 */
  NhlRLClear(tx_rlist);
  NhlRLSetFloat (tx_rlist, "txFontHeightF",         font_height);
  NhlRLSetString(tx_rlist, "txFont",                "helvetica-bold");
  NhlRLSetString(tx_rlist, "txJust",                "BottomLeft");
  NhlRLSetString(tx_rlist, "txPerimOn",             "True");
  NhlRLSetString(tx_rlist, "txPerimColor",          "black");
  NhlRLSetString(tx_rlist, "txFontColor",           "black");
  NhlRLSetString(tx_rlist, "txBackgroundFillColor", "white");

/*
 * Set some line resources for outlining the color boxes.
 */
  NhlRLClear(ln_rlist);
  NhlRLSetString(ln_rlist,"gsLineColor","black");

/*
 * Clear the resource lines for the polygon.  The resources will
 * be set inside the loop below.
 */
  NhlRLClear(gn_res.id);

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
        NhlRLSetInteger(gn_rlist,"gsFillColor", offset + (i-1));
                                                                    
        polygon_ndc_wrap(wks, xbox2, ybox2, "float","float", 5,
                             0, 0, NULL, NULL, &gn_res, &special_res2);
/*
 * Outline box in black.
 */
        polyline_ndc_wrap(wks, xbox2, ybox2, "float","float", 5,
                              0, 0, NULL, NULL, &ln_res, &special_res2);
/*
 * Draw text string indicating color index value.
 */
        sprintf(tmpstr,"%d",i-1);
        text_ndc_wrap(wks, tmpstr, &txpos, &typos, "float", "float",
                          &tx_res, &special_res2);
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
    NhlRLClear(sr_list);
    NhlRLSetMDFloatArray(sr_list,"wkColorMap",&cmap[0],2,cmap_dimsizes);
    NhlSetValues(wks,sr_list);
    free(cmap);
  }
}

/*
 * Routine for paneling same-sized plots.
 */
void panel_wrap(int wks, nglPlotId *plots, int nplots_orig, int *dims, 
                int ndims, ResInfo *lb_res, ResInfo *fs_res,
                nglRes *special_res)
{
  int i, nplots, npanels, is_row_spec, nrows, ncols, draw_boxes = 0;
  int num_plots_left, nplot, nplot4, nr, nc, new_ncols, nnewplots;
  nglPlotId *newplots, pplot;
  int *row_spec, first_time;
  int nvalid_plot, nvalid_plots, valid_plot;
  int panel_save, panel_debug, panel_center;
  int panel_labelbar, main_string_on, is_figure_strings, explicit;
  int fs_bkgrn, fs_perim_on, just;
  int *anno, fs_text, added_anno, am_rlist;
  float paras[9], orths[9], para, orth, len_pct, wsp_hpct, wsp_wpct;
  NhlString *panel_strings, *lstrings;
  int *colors, *patterns;
  float *scales, *levels;
  int fill_on, glyph_style, fill_arrows_on, mono_line_color, mono_fill_arrow;
  ng_size_t ncolors, nlevels, npatterns, nscales, nstrings;
  int mono_fill_pat, mono_fill_scl, mono_fill_col;
  int lb_orient, lb_perim_on, lb_auto_stride, lb_alignment, labelbar_object;
  float lb_width_set, lb_height_set;
  float lb_fh, lb_x, lb_y, lb_width, lb_height, tmp_range;
  char plot_type[10];
  int maxbb, calldraw, callframe;
  int lft_pnl, rgt_pnl, bot_pnl, top_pnl;
  float font_height, fs_font_height;
  float x_lft, x_rgt, y_bot, y_top;
  float xlft, xrgt, xbot, xtop;
  float xsp, ysp, xwsp_perc, ywsp_perc, xwsp, ywsp;
  float vpx, vpy, vpw, vph, dxl, dxr, dyt, dyb;
  float *old_vp, *xpos, *ypos, max_rgt, max_top;
  float top, bottom, left, right;
  float newtop, newbot, newrgt, newlft;
  float plot_width, plot_height, total_width, total_height;
  float new_plot_width, new_plot_height, new_total_width, new_total_height;
  float min_xpos, max_xpos, min_ypos, max_ypos;
  float scaled_width, scaled_height, xrange, yrange;
  float scale, row_scale, col_scale, max_width, max_height;
  int grlist, sr_list;
  NhlBoundingBox bb, *newbb;
  int lb_rlist, fs_rlist;

/*
 * Set resource ids.
 */
  lb_rlist = lb_res->id;
  fs_rlist = fs_res->id;

/*
 * Resource lists for getting and retrieving resources.
 */

  grlist = NhlRLCreate(NhlGETRL);
  sr_list = NhlRLCreate(NhlSETRL);

/*
 * First check if paneling is to be specified by (#rows x #columns) or
 * by #columns per row.  The default is rows x columns, unless 
 * resource nglPanelRowSpec is set to True.
 */
  is_row_spec = special_res->nglPanelRowSpec;

  if(is_row_spec) {
    row_spec = dims;
    npanels  = 0;
    nrows    = ndims;
    ncols    = imax_array(row_spec,ndims);
    
    for(i = 0; i <= nrows-1; i++) {
      if(row_spec[i] < 0) {
        NhlPError(NhlFATAL,NhlEUNKNOWN,"panel: you have specified a negative value for the number of plots in a row.\n");
        return;
      }
      npanels += row_spec[i];
    }
  }
  else {
    if(ndims != 2) {
      NhlPError(NhlFATAL,NhlEUNKNOWN,"panel: for the third argument of panel, you must either specify # rows by # columns or set the nglPanelRowSpec resource to True and set the number of plots per row.\n");
      return;
    }
    nrows    = dims[0];
    ncols    = dims[1];
    npanels  = nrows * ncols;
    row_spec = (int *)malloc(nrows * sizeof(int));
    for(i = 0; i <= nrows-1; i++) row_spec[i] = ncols;
  }
  
  if(nplots_orig > npanels) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"panel: you have more plots than you have panels.\nOnly %d plots will be drawn.\n", npanels);
    nplots = npanels;
  }
  else {
    nplots = nplots_orig;
  }

/*
 * Check for special resources.
 */ 
  panel_save        = special_res->nglPanelSave;
  panel_debug       = special_res->nglDebug;
  panel_center      = special_res->nglPanelCenter;
  is_figure_strings = special_res->nglPanelFigureStringsCount;
  panel_labelbar    = special_res->nglPanelLabelBar;
  just              = special_res->nglPanelFigureStringsJust;
  para              = special_res->nglPanelFigureStringsParallelPosF;
  orth              = special_res->nglPanelFigureStringsOrthogonalPosF;
  calldraw          = special_res->nglDraw;
  callframe         = special_res->nglFrame;
  xwsp_perc         = special_res->nglPanelXWhiteSpacePercent;
  ywsp_perc         = special_res->nglPanelYWhiteSpacePercent;
  main_string_on    = 0;     /* Resource for this to be added later. */
  
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
 * drawn to a PostScript, PDF, or image workstation, because the
 * bounding box is already maximized for an NCGM/X11 window.
 */ 
  maxbb = 1;
  if(special_res->nglMaximize) {
    if( strcmp(NhlClassName(wks),"psWorkstationClass") &&
        strcmp(NhlClassName(wks),"pdfWorkstationClass") &&
        strcmp(NhlClassName(wks),"cairoPSPDFWorkstationClass") &&
        strcmp(NhlClassName(wks),"cairoImageWorkstationClass")) {
      maxbb = 0;
    }
  }

/*
 * Set some resources for the figure strings, if they exist.
 */
  if(is_figure_strings) {
    panel_strings = special_res->nglPanelFigureStrings;
    fs_perim_on   = special_res->nglPanelFigureStringsPerimOn;
    fs_bkgrn      = special_res->nglPanelFigureStringsBackgroundFillColor;

/*
 * Get and set resource values for figure strings on the plots.
 */
    if(just < 0 || just > 8) {
      just = 8;      /* Default to BottomRight */
      NhlPError(NhlWARNING,NhlEUNKNOWN,"panel_wrap: incorrect value for nglPanelFigureStringsJust, defaulting to 'BottomRight'");
    }

    paras[0] = -1;     /* TopLeft    */
    paras[1] = -1;     /* CenterLeft */
    paras[2] = -1;     /* BottomLeft   */

    paras[3] =  0;     /* TopCenter    */
    paras[4] =  0;     /* CenterCenter */
    paras[5] =  0;     /* BottomCenter */

    paras[6] =  1;     /* TopRight     */
    paras[7] =  1;     /* CenterRight  */
    paras[8] =  1;     /* BottomRight  */

    orths[0] = -1;     /* TopLeft      */
    orths[1] =  0;     /* CenterLeft   */
    orths[2] =  1;     /* BottomLeft   */

    orths[3] = -1;     /* TopCenter    */
    orths[4] =  0;     /* CenterCenter */
    orths[5] =  1;     /* BottomCenter */

    orths[6] = -1;     /* TopRight     */
    orths[7] =  0;     /* CenterRight  */
    orths[8] =  1;     /* BottomRight  */
 }

/*
 * Error check the values that the user has entered, to make sure
 * they are valid.
 */
  if(xwsp_perc < 0 || xwsp_perc >= 100.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"panel: attribute nglPanelXWhiteSpacePercent must be >= 0 and < 100.\nDefaulting to 1.\n");
    xwsp_perc = 1.;
  }

  if(ywsp_perc < 0 || ywsp_perc >= 100.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"panel: attribute nglPanelYWhiteSpacePercent must be >= 0 and < 100.\nDefaulting to 1.\n");
    ywsp_perc = 1.;
  }
  
  if(x_lft < 0. || x_lft >= 1.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"panel: attribute nglPanelLeft must be >= 0.0 and < 1.0\nDefaulting to 0.\n");
    x_lft = 0.0;
  }
  
  if(x_rgt <= 0. || x_rgt > 1.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"panel: attribute nglPanelRight must be > 0.0 and <= 1.0\nDefaulting to 1.\n");
    x_rgt = 1.0;
  }
  
  if(y_top <= 0. || y_top > 1.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"panel: attribute nglPanelTop must be > 0.0 and <= 1.0\nDefaulting to 1.\n");
    y_top = 1.0;
  }
  
  if(y_bot < 0. || y_bot >= 1.) {
    NhlPError(NhlWARNING,NhlEUNKNOWN,"Warning: panel: attribute nglPanelBottom must be >= 0.0 and < 1.0\nDefaulting to 0.\n");
    y_bot = 0.0;
  }
  
  if(x_rgt <= x_lft) {
    NhlPError(NhlFATAL,NhlEUNKNOWN,"panel: nglPanelRight (%g",x_rgt,")\n must be greater than nglPanelLeft (%g",x_lft,").\n");
    return;
  }
  
  if(y_top <= y_bot) {
    NhlPError(NhlFATAL,NhlEUNKNOWN,"panel: nglPanelTop (%g",y_top,")\n must be greater than nglPanelBottom (%g",y_bot,").\n");
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
    NhlPError(NhlFATAL,NhlEUNKNOWN,"panel: all of the plots passed to panel appear to be invalid\n");
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

  if(is_figure_strings || panel_labelbar) {
    if(plots[valid_plot].ncontour > 0 && plots[valid_plot].contour != NULL) {
      strcpy(plot_type,"contour");
/*
 * Get information on how contour plot is filled, so we can recreate 
 * labelbar, if requested.
 */
      NhlRLClear(grlist);
      NhlRLGetFloat(grlist,   "cnInfoLabelFontHeightF",     &font_height);
      NhlRLGetInteger(grlist, "cnFillOn",                   &fill_on);
      NhlRLGetInteger(grlist, "cnExplicitLabelBarLabelsOn", &explicit);
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
	  if(explicit) {
	    NhlRLGetStringArray(grlist,"lbLabelStrings",    
				&lstrings, &nstrings);
	  }
	  else {
	    NhlRLGetStringArray(grlist,"cnLineLabelStrings", 
				&lstrings, &nstrings);
	  }
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
      NhlPError(NhlWARNING,NhlEUNKNOWN,"Warning: panel: unrecognized plot type, thus unable to get information for font height.\nDefaulting to %g", font_height);
    }
/*
 * Use this font height for the panel strings, if any, unless the user
 * has set nglPanelFigureStringsFontHeightF.
 */
    if(special_res->nglPanelFigureStringsFontHeightF > 0) {
      fs_font_height = special_res->nglPanelFigureStringsFontHeightF;
    }
    else {
      fs_font_height = font_height;
    }
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
    lb_perim_on    = special_res->nglPanelLabelBarPerimOn;
    lb_auto_stride = special_res->nglPanelLabelBarLabelAutoStride;
    lb_alignment   = special_res->nglPanelLabelBarAlignment;
    lb_orient      = special_res->nglPanelLabelBarOrientation;
    lb_width       = special_res->nglPanelLabelBarWidthF;
    lb_height      = special_res->nglPanelLabelBarHeightF;

/*
 * Keep track if labelbar width and/or height was explicitly set
 * by user, so we know later whether the labelbar height/width we're
 * working with is a calculated one, or a specified one.
 */
    if(lb_width < 0.) {
      lb_width_set = 0;
    }
    else {
      lb_width_set = 1;
    } 
    if(lb_height < 0.) {
      lb_height_set = 0;
    }
    else {
      lb_height_set = 1;
    }

    if(lb_orient == NhlVERTICAL) {
      if(!lb_width_set) {
        lb_width = 0.20 * plot_width + 2.*xwsp;
      }
      if(!lb_height_set) {
/*
 * Adjust height depending on whether we have one row or multiple rows.
 */
        if(nplots > 1 && nrows > 1) {
          lb_height  = (nrows-1) * (2.*ywsp + plot_height);
        }
        else {
          lb_height  = plot_height;
        }
      }
    }
    else {
      if(!lb_height_set) {
        lb_height = 0.20 * plot_height + 2.*ywsp;
      }
      if(!lb_width_set) {
/*
 * Adjust width depending on whether we have one column or multiple 
 * columns.
 */
        if(nplots > 1 && ncols > 1) {
          lb_width  = (ncols-1) * (2.*xwsp + plot_width);
        }
        else {
          lb_width  = plot_width;
        }
      }
    }
  }
  else {
    lb_height = 0.;
    lb_width  = 0.;
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
    row_scale = yrange/(nrows*total_height+lb_height);
    col_scale = xrange/(ncols*total_width);
    scale     = min(col_scale,row_scale);
    if(lb_height_set) {
      yrange  = yrange - lb_height;
    }
    else {
      yrange  = yrange - scale * lb_height;
    }
  }
  else {
    row_scale = yrange/(nrows*total_height);
    col_scale = xrange/(ncols*total_width+lb_width);
    scale     = min(col_scale,row_scale);
    if(lb_width_set) {
      xrange  = xrange - lb_width;
    }
    else {
      xrange  = xrange - scale * lb_width;
    }
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
 * If we have figure strings, then allocate space for them, and determine
 * white spacing around the text box.
 */
  if(is_figure_strings) {
    anno = (int *)malloc(nplots * sizeof(int));

    len_pct = 0.025;          /* Percentage of width/height of plot */
                              /* for white space around text box.   */
    if(vpw < vph) {
      wsp_hpct = (len_pct * vpw) / vph;
      wsp_wpct = len_pct;
    }
    else {
      wsp_hpct = len_pct;
      wsp_wpct = (len_pct * vph) / vpw;
    }
/*
 * If we are calculating the parallel and/or orthogonal position for
 * the figure strings, first make sure it is not one of the "center"
 * values. Otherwise, add a little bit of a margin to it so it's not
 * flush with the edge of the plot. 
 */
    if(para == -999.) {
      para = paras[just];
      if(paras[just] != 0) para *= (0.5 - wsp_wpct);
    }
    if(orth == -999.) {
      orth = orths[just];
      if(orths[just] != 0) orth *= (0.5 - wsp_hpct);
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

        NhlRLClear(sr_list);
        NhlRLSetFloat(sr_list,"vpXF",     xpos[nc]);
        NhlRLSetFloat(sr_list,"vpYF",     ypos[nr]);
        NhlRLSetFloat(sr_list,"vpWidthF", scale*old_vp[nplot4+2]);
        NhlRLSetFloat(sr_list,"vpHeightF",scale*old_vp[nplot4+3]);
        (void)NhlSetValues(*(pplot.base), sr_list);

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
 * Add figure string if requested.
 */
        added_anno = 0;
        if(is_figure_strings) {
          if(nplot < special_res->nglPanelFigureStringsCount && 
             (panel_strings[nplot] != NULL) &&
             strcmp(panel_strings[nplot],"")) {
/*
 * Set the text resources for the figure string.
 */
            NhlRLSetString  (fs_rlist, "txString",      panel_strings[nplot]);
            NhlRLSetFloat   (fs_rlist, "txFontHeightF", fs_font_height);
            NhlRLSetInteger (fs_rlist, "txPerimOn",     fs_perim_on);
            NhlRLSetInteger (fs_rlist, "txBackgroundFillColor", fs_bkgrn);
            NhlCreate(&fs_text,"text",NhltextItemClass,wks,fs_rlist);

/*
 * Add annotation to plot.
 */
            anno[nplot] = NhlAddAnnotation(*(pplot.base),fs_text);
            added_anno = 1;
            am_rlist = NhlRLCreate(NhlSETRL);
            NhlRLClear(am_rlist);
            NhlRLSetInteger (am_rlist,"amZone"           , 0);
            NhlRLSetInteger (am_rlist,"amJust"           , just);
            NhlRLSetFloat   (am_rlist,"amParallelPosF"   , para);
            NhlRLSetFloat   (am_rlist,"amOrthogonalPosF" , orth);
            NhlRLSetInteger (am_rlist,"amResizeNotify"   , True);
            (void)NhlSetValues(anno[nplot], am_rlist);
          }
          else {
            anno[i] = -1;
          }
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
      if(!lb_height_set) lb_height *= scale;
      if(!lb_width_set)  lb_width  *= scale;

      if(special_res->nglPanelLabelBarLabelFontHeightF > 0.) {
        lb_fh = special_res->nglPanelLabelBarLabelFontHeightF;
      }
      else {
        lb_fh = font_height;
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
          lb_y = max(ywsp+lb_height,bottom-ywsp);
        }

        if(special_res->nglPanelLabelBarXF > 0.) {
          lb_x = special_res->nglPanelLabelBarXF;
        }
        else if(ncols == 1 && lb_width <= scaled_width) {
          lb_x = min_xpos + (scaled_width-lb_width)/2.;
        }
        else {
          tmp_range = x_rgt - x_lft;
          lb_x      = x_lft + (tmp_range - lb_width)/2.;
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
          lb_x = min(1.-(xwsp+lb_width),max_rgt+xwsp);
        }
        if(special_res->nglPanelLabelBarYF > 0.) {
          lb_y = special_res->nglPanelLabelBarYF;
        }
        else if(nrows == 1 && lb_height <= scaled_height) {
          lb_y = ypos[0]-(scaled_height - lb_height)/2.;
        }
        else {
          tmp_range = y_top - y_bot;
          lb_y      = y_top-(tmp_range - lb_height)/2.;
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
      NhlRLSetFloat       (lb_rlist,"vpWidthF",          lb_width);
      NhlRLSetFloat       (lb_rlist,"vpHeightF",         lb_height);
      NhlRLSetString      (lb_rlist,"lbAutoManage",      "False");
      NhlRLSetInteger     (lb_rlist,"lbOrientation",     lb_orient);
      NhlRLSetIntegerArray(lb_rlist,"lbFillColors",      colors, ncolors);
      NhlRLSetInteger     (lb_rlist,"lbBoxCount",        (int)ncolors);
      if(!strcmp(plot_type,"contour")) {
	NhlRLSetStringArray (lb_rlist,"lbLabelStrings", lstrings, nstrings);
      }
      else {
	NhlRLSetFloatArray  (lb_rlist,"lbLabelStrings", levels, nlevels);
      }
      
      NhlRLSetInteger     (lb_rlist,"lbPerimOn",         lb_perim_on);
      NhlRLSetInteger     (lb_rlist,"lbLabelAutoStride", lb_auto_stride);
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
      if(strcmp(plot_type,"contour")) {
	free(levels);
      }
      free(colors);
      if(!mono_fill_pat) free(patterns);
      if(!mono_fill_scl) free(scales);
    }
/*
 * The plot_type is not either vector or contour, so we have to bail on
 * creating a labelbar. 
 */
    else {
      NhlPError(NhlWARNING,NhlEUNKNOWN,"panel: unrecognized plot type for getting labelbar information. Ignoring labelbar request.");
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
        if(added_anno && anno[i] > 0) {
          NhlRemoveAnnotation(*(plots[i].base),anno[i]);
        }

        nplot4 = 4 * i;
        
        NhlRLClear(sr_list);
        NhlRLSetFloat(sr_list,"vpXF",     old_vp[nplot4]);
        NhlRLSetFloat(sr_list,"vpYF",     old_vp[nplot4+1]);
        NhlRLSetFloat(sr_list,"vpWidthF", old_vp[nplot4+2]);
        NhlRLSetFloat(sr_list,"vpHeightF",old_vp[nplot4+3]);
        (void)NhlSetValues(*(plots[i].base), sr_list);
      }
    }
  }
  if(!is_row_spec) free(row_spec);
  free(newplots);
  free(old_vp);
  free(ypos);
  free(newbb);
}

void c_wmbarbp(int wksid, float x, float y, float u, float v) {

  int ezf;
  float xt,yt,xtn,ytn,ang1,ang2,utmp,vtmp,vlen,d2r=0.01745329,xtm,ytm;
  
  gactivate_ws (wksid);
  c_wmgeti("ezf",&ezf);
  if (ezf != -1) {
/*
 * Find a small vector *on the map* in the direction of the wind barb.
 * The cos term is introduced to accommodate for the latitude of the
 * barb - as you approach the poles, a given spacial distance in latitude
 * in degrees is less than the same spacial distance in degrees
 * longitude.
 */
    ang1 = atan2(u,v);
    c_maptrn(x, y, &xt, &yt);
    if (xt != 1.e12) {
      xtm = x + 0.1 * cos(ang1);            
      ytm = y + 0.1 * sin(ang1)/cos(d2r*x);
      c_maptrn(xtm, ytm, &xtn, &ytn);
      ang2 = atan2(ytn-yt,xtn-xt);
      vlen = sqrt(u*u + v*v);
      utmp = vlen*cos(ang2);
      vtmp = vlen*sin(ang2);
      c_wmbarb(xt, yt, utmp,vtmp);
    }
  }
  else {
    c_wmbarb(x, y, u, v);
  }

  gdeactivate_ws (wksid);
}

void c_wmstnmp(int wksid, float x, float y, char *imdat) {

  int ezf,iang;
  float xt,yt,xtt,ytt,fang;
  char tang[3];
  
  gactivate_ws (wksid);
  c_wmgeti("ezf",&ezf);
  if (ezf != -1) {
    c_maptrn(x,y,&xt,&yt);
    if (xt != 1.e12) {
      tang[0] = imdat[6];
      tang[1] = imdat[7];
      tang[2] = 0;
      fang = 10.*0.0174532925199*atof(tang);
      c_maptrn(x+0.1*cos(fang),
           y+0.1/cos(0.0174532925199*x)*sin(fang), &xtt, &ytt);
      fang = fmod(57.2957795130823*atan2(xtt-xt, ytt-yt)+360., 360.);
      iang = (int) max(0,min(35,NINT(fang/10.)));
      sprintf(tang,"%2d",iang);
      imdat[6] = tang[0];
      imdat[7] = tang[1];
      c_wmstnm(xt, yt, imdat);
    }
  }
  else {
    c_wmstnm(x, y, imdat);
  }
  gdeactivate_ws (wksid);
}

void c_wmsetip(char* string, int v) {
  c_wmseti(string, v);
}
void c_wmsetrp(char* string, float v) {
  c_wmsetr(string, v);
}
void c_wmsetcp(char* string, char *value) {
  c_wmsetc(string, value);
}
int c_wmgetip(char* string) {
  int *iret;
  iret = (int *) malloc(sizeof(int));
  c_wmgeti(string,iret);
  return *iret;
}
float c_wmgetrp(char* string) {
  float *fret;
  fret = (float *) malloc(sizeof(float));
  c_wmgetr(string, fret);
  return *fret;
}
char *c_wmgetcp(char* string) {
  char *cret;
  cret = (char *) malloc(20*sizeof(char));
  c_wmgetc(string, cret, strlen(cret));
  return cret;
}
