#include <ncarg/hlu/hlu.h>
#include <ncarg/hlu/NresDB.h>
#include <ncarg/hlu/ResList.h>
#include <ncarg/hlu/App.h>
#include <ncarg/hlu/XWorkstation.h>
#include <ncarg/hlu/NcgmWorkstation.h>
#include <ncarg/hlu/PSWorkstation.h>
#include <ncarg/hlu/PDFWorkstation.h>
#include <ncarg/hlu/ContourPlot.h>
#include <ncarg/hlu/XyPlot.h>
#include <ncarg/hlu/MapPlot.h>
#include <ncarg/hlu/VectorPlot.h>
#include <ncarg/hlu/StreamlinePlot.h>
#include <ncarg/hlu/ScalarField.h>
#include <ncarg/hlu/CoordArrays.h>
#include <ncarg/hlu/VectorField.h>
#include <netcdf.h>

#define max(x,y)   ((x) > (y) ? (x) : (y))

/*
 * The length of the variable type must be big enough to hold the 
 * longest *numeric* type name, which in this case is "integer".
 */
#define TYPE_LEN 8     

/*
 * Prototype GSUN and supplemental functions.
 */

extern int scalar_field(void *, const char *, int, int, int, void *, 
                        const char *, int, void *, const char *, int,
                        void *, int);

extern int coord_array(void *, void *, const char *, const char *, 
					   int, int *, int, int *, int, int, void *, void *, int);

extern int vector_field(void *, void *, const char *, const char *, int,
                        int, int, void *, const char *, int, void*, 
                        const char *, int, int, void *, void *, int);

extern void compute_ps_device_coords(int,int);

extern void maximize_plot(int, int);

extern int gsn_open_wks(const char *, const char *, int);

extern int gsn_contour_wrap(int, void *, const char *, int, int, 
                            int, void *, const char *, int, void *, 
                            const char *, int, void *, int, int);

extern int gsn_xy_wrap(int, void *, void *, const char *,
					   const char *, int, int *, int, int *, int, int,
					   void *, void *, int, int, int);

extern int gsn_vector_wrap(int, void *, void *, const char *, 
                           const char *, int, int, int, void *, 
                           const char *, int, void *, const char *, int,
                           int, void *, void *, int, int);

extern int gsn_streamline_wrap(int, void *, void *, const char *, 
                               const char *, int, int, int, void *, 
                               const char *, int, void *, const char *, 
                               int, int, void *, void *, int, int);

extern int gsn_map_wrap(int, int);

extern int gsn_contour_map_wrap(int, void *, const char *, int, int, 
                                int, void *, const char *, int, void *, 
                                const char *, int, void *, int, int, int);
