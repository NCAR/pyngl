#include <ncarg/hlu/hlu.h>
#include <ncarg/hlu/NresDB.h>
#include <ncarg/hlu/ResList.h>
#include <ncarg/hlu/App.h>
#include <ncarg/hlu/XWorkstation.h>
#include <ncarg/hlu/NcgmWorkstation.h>
#include <ncarg/hlu/PSWorkstation.h>
#include <ncarg/hlu/PDFWorkstation.h>
#include <ncarg/hlu/ContourPlot.h>
#include <ncarg/hlu/VectorPlot.h>
#include <ncarg/hlu/StreamlinePlot.h>
#include <ncarg/hlu/ScalarField.h>
#include <ncarg/hlu/VectorField.h>
#include <netcdf.h>


#define max(x,y)   ((x) > (y) ? (x) : (y))

/*
 * Prototype GSUN and supplemental functions.
 */

extern int scalar_field(void *, NrmQuark, int, int, int, void *, int);

extern int vector_field(void *, void *, NrmQuark, NrmQuark, int, int, int,
						int, void *, void *, int);

extern void compute_ps_device_coords(int,int);

extern void maximize_plot(int, int);

extern int gsn_open_wks(char *, char *, int);

extern int gsn_contour_wrap(int, void *, NrmQuark, int, int, int, void *, 
							int, int);

extern int gsn_vector_wrap(int, void *, void *, NrmQuark, NrmQuark, int, int,
						   int, int, void *, void *, int, int);

extern int gsn_streamline_wrap(int, void *, void *, NrmQuark, NrmQuark, int,
							   int, int, int, void *, void *, int, int);

