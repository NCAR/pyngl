#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gsun.h"

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
 * This function maximizes and draws the plot, and advances the frame.
 */

void draw_and_frame(int wks, int plot, gsnRes *special_res)
{
  if(special_res->gsnMaximize) maximize_plot(wks, plot);
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
 * used with the XY object. Note that X and or Y can be 1 or
 * 2-dimensional, but they must match in the rightmost dimension.
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

  if(x != NULL) {
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

  special_res2.gsnDraw  = 0;
  special_res2.gsnFrame = 0;
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

  special_res2.gsnDraw  = 0;
  special_res2.gsnFrame = 0;
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

  special_res2.gsnDraw  = 0;
  special_res2.gsnFrame = 0;
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

  special_res2.gsnDraw  = 0;
  special_res2.gsnFrame = 0;
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
 * Draw plots and advance frame.
 */

  draw_and_frame(wks, map, special_res);

/*
 * Return.
 */

  return(map);
}

int gsn_text_ndc_wrap(int wks, char* string, float x, float y, 
					  int tx_rlist, gsnRes *special_res)
{
  int text;
  gsnRes special_res2;

  NhlRLSetString(tx_rlist, "txString", string);
  NhlRLSetFloat (tx_rlist, "txPosXF" , x);
  NhlRLSetFloat (tx_rlist, "txPosYF" , y);
  NhlCreate(&text,"text",NhltextItemClass,wks,tx_rlist);
/*
 * Draw plots and advance frame.
 */

  special_res2.gsnMaximize = 0;
  special_res2.gsnFrame    = 0;

  draw_and_frame(wks, text, &special_res2);

/*
 * Return.
 */

  return(text);
}


int gsn_text_wrap(int wks, int plot, char* string, float x, float y, 
				  int tx_rlist, gsnRes *special_res)
{
  float xndc, yndc, oor = 0.;
  int text, status;

/*
 * Convert from plot's data space to NDC space.
 */

  (void)NhlDataToNDC(plot,&x,&y,1,&xndc,&yndc,NULL,NULL,&status,&oor);

  if(special_res->gsnDebug) {
	printf("x = %g y = %g xndc = %g yndc = %g\n", x, y, xndc, yndc);
  }

  text = gsn_text_ndc_wrap(wks, string, xndc, yndc, tx_rlist, special_res);

/*
 * Return.
 */

  return(text);
}

