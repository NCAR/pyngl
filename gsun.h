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
#include <ncarg/hlu/LabelBar.h>
#include <ncarg/hlu/StreamlinePlot.h>
#include <ncarg/hlu/ScalarField.h>
#include <ncarg/hlu/CoordArrays.h>
#include <ncarg/hlu/VectorField.h>
#include <ncarg/hlu/TextItem.h>
#include <ncarg/hlu/GraphicStyle.h>
#include <ncarg/hlu/LogLinPlot.h>
#include <ncarg/hlu/Primitive.h>
#include <netcdf.h>

#define min(x,y)   ((x) < (y) ? (x) : (y))
#define max(x,y)   ((x) > (y) ? (x) : (y))

#define nglPlot      0
#define nglPrimitive 1

/*
 * The length of the variable type must be big enough to hold the 
 * longest *numeric* type name, which in this case is "integer".
 */
#define TYPE_LEN 8     


/*
 * Define a structure to hold special resources that aren't
 * set in the normal HLU way (that is, via NhlRLSetxxx).
 */

typedef struct {
  int   nglMaximize;
  int   nglDraw;
  int   nglFrame;
  int   nglScale;
  int   nglDebug;

/*
 * Special resources for spanning a full color map when
 * filling vectors and/or contours.
 */
  int nglSpreadColors;
  int nglSpreadColorStart;
  int nglSpreadColorEnd;

/*
 * Special resources for PS/PDF output.
 */
  int   nglPaperOrientation;
  float nglPaperWidth;
  float nglPaperHeight;
  float nglPaperMargin;

/*
 * Special resources for paneling.
 */
  int   nglPanelSave;
  int   nglPanelCenter;
  int   nglPanelRowSpec;
  float nglPanelXWhiteSpacePercent;
  float nglPanelYWhiteSpacePercent;
  int   nglPanelBoxes;
  float nglPanelLeft;
  float nglPanelRight;
  float nglPanelBottom;
  float nglPanelTop;
  float nglPanelInvsblTop;
  float nglPanelInvsblLeft;
  float nglPanelInvsblRight;
  float nglPanelInvsblBottom;

/*
 * Special resources for a panel labelbar.
 */
  int   nglPanelLabelBar;  
  float nglPanelLabelBarXF;
  float nglPanelLabelBarYF;
  float nglPanelLabelBarWidthF;
  float nglPanelLabelBarHeightF;
  int   nglPanelLabelBarOrientation;
  int   nglPanelLabelBarPerimOn;
  int   nglPanelLabelBarAlignment;
  float nglPanelLabelBarFontHeightF;
  float nglPanelLabelBarOrthogonalPosF;
  float nglPanelLabelBarParallelPosF;
} nglRes;

/*
 * Define a structure to hold all the possible types of objects
 * that the ngl_* functions create.
 */

typedef struct {
  int   *base;
  int   *contour;
  int   *vector;
  int   *streamline;
  int   *map;
  int   *xy;
  int   *xydspec;
  int   *text;
  int   *primitive;
  int   *cafield;
  int   *sffield;
  int   *vffield;

  int   nbase;
  int   ncontour;
  int   nvector;
  int   nstreamline;
  int   nmap;
  int   nxy;
  int   nxydspec;
  int   ntext;
  int   nprimitive;
  int   ncafield;
  int   nsffield;
  int   nvffield;
} nglPlotId;

/*
 * Supplemental functions. 
 */

extern int imax_array(int *, int);
extern float xmax_array(float *, int);
extern float xmin_array(float *, int);
extern float *fspan(float, float, int);
extern int *ispan(int, int, int);

extern void compute_ps_device_coords(int, nglPlotId *, int, nglRes *);

extern void maximize_plot(int, nglPlotId *, int, int, nglRes *);

extern void spread_colors(int, int, int, int, char*, char*, int);

extern void scale_plot(int);

extern float *coerce_to_float(void *, const char *, int);

extern int *get_non_missing_pairs(float *, float *, int, int, float *, 
                                  float *, int, int *);

extern void collapse_nomsg_xy(float *, float *, float **, float **, int, int,
                              int, float *, float *, int *);

extern void set_resource(char *, int, void *, const char *, int, int *);

extern int create_graphicstyle_object(int);

extern void initialize_ids(nglPlotId *);

extern void initialize_resources(nglRes *, int);

extern void draw_and_frame(int, nglPlotId *, int, int, nglRes *);

/*
 * Data object routines.
 */

extern int scalar_field(void *, const char *, int, int, int, void *, 
                        const char *, int, void *, const char *, int,
                        void *, int);

extern int coord_array(void *, void *, const char *, const char *, 
                       int, int *, int, int *, int, int, void *, void *,
                       int);

extern int vector_field(void *, void *, const char *, const char *, int,
                        int, int, void *, const char *, int, void*, 
                        const char *, int, int, void *, void *, int);

/*
 * Workstation routine.
 */

extern int ngl_open_wks_wrap(const char *, const char *, int);

/*
 * Plotting routines.
 */

extern nglPlotId ngl_contour_wrap(int, void *, const char *, int, int, 
                                  int, void *, const char *, int, void *, 
                                  const char *, int, void *, int, int, 
                                  nglRes *);

extern nglPlotId ngl_xy_wrap(int, void *, void *, const char *,
                             const char *, int, int *, int, int *, int, int,
                             void *, void *, int, int, int, nglRes *);

extern nglPlotId ngl_y_wrap(int, void *, const char *, int, int *, int, 
                            void *, int, int, int, nglRes *);

extern nglPlotId ngl_vector_wrap(int, void *, void *, const char *, 
                                 const char *, int, int, int, void *, 
                                 const char *, int, void *, const char *, 
                                 int, int, void *, void *, int, int, 
                                 nglRes *);

extern nglPlotId ngl_streamline_wrap(int, void *, void *, const char *, 
                                     const char *, int, int, int, void *, 
                                     const char *, int, void *, 
                                     const char *, int, int, void *, void *,
                                     int, int, nglRes *);

extern nglPlotId ngl_map_wrap(int, int, nglRes *);

extern nglPlotId ngl_contour_map_wrap(int, void *, const char *, int, int, 
                                      int, void *, const char *, int, 
                                      void *, const char *, int, void *, 
                                      int, int, int, nglRes *);

extern nglPlotId ngl_vector_map_wrap(int, void *, void *, const char *, 
                                     const char *, int, int, int, void *, 
                                     const char *, int, void *, 
                                     const char *, int, int, void *, void *,
                                     int, int, int, nglRes *);

extern nglPlotId ngl_streamline_map_wrap(int, void *, void *, const char *, 
                                         const char *, int, int, int, 
                                         void *, const char *, int, void *,
                                         const char *, int, int, void *, 
                                         void *, int, int, int, nglRes *);

extern nglPlotId ngl_vector_scalar_wrap(int, void *, void *, void *, 
                                        const char *, const char *, 
                                        const char *, int, int, int, 
                                        void *, const char *, int, void *, 
                                        const char *, int, int, int, void *, 
                                        void *, void *, int, int, int, 
                                        nglRes *);

extern nglPlotId ngl_vector_scalar_map_wrap(int, void *, void *, void *, 
                                            const char *, const char *, 
                                            const char *, int, int, int,
                                            void *, const char *, int, 
                                            void *, const char *, int, int, 
                                            int, void *, void *, void *, 
                                            int, int, int, int, nglRes *);

/*
 * Text routines.
 */

extern nglPlotId ngl_text_ndc_wrap(int, char *, void *, void *, const char *, 
                                   const char *, int, nglRes *);
extern nglPlotId ngl_text_wrap(int, nglPlotId *, char *, void *, void *,
                               const char *, const char *, int, nglRes *);

/*
 * Primitive drawing routines.
 */

extern void ngl_poly_wrap(int, nglPlotId *, void *, void *, const char *type_x,
                          const char *type_y, int, int, int, void *, void*,
                          NhlPolyType, int, nglRes *);


extern nglPlotId ngl_add_poly_wrap(int, nglPlotId *, void *, void *, 
                                   const char *, const char *, int, int,
                                   int, void *, void *,NhlPolyType, int,
                                   nglRes *);

extern void ngl_polymarker_ndc_wrap(int, void *, void *, const char *, 
                                    const char *, int, int, int, void *,
                                    void *, int, nglRes *);

extern void ngl_polyline_ndc_wrap(int, void *, void *, const char *, 
                                  const char *, int, int, int, void *,
                                  void *, int, nglRes *);

extern void ngl_polygon_ndc_wrap(int, void *, void *, const char *, 
                                 const char *, int, int, int, void *, 
                                 void *, int, nglRes *);

extern void ngl_polymarker_wrap(int, nglPlotId *, void *, void *,
                                const char *, const char *, int, int, int,
                                void *, void *, int, nglRes *);

extern void ngl_polyline_wrap(int, nglPlotId *, void *, void *, const char *, 
                              const char *, int, int, int, void *, void *, 
                              int, nglRes *);

extern void ngl_polygon_wrap(int, nglPlotId *, void *, void *, const char *, 
                             const char *, int, int, int, void *, void *, 
                             int, nglRes *);

extern nglPlotId ngl_add_polyline_wrap(int, nglPlotId *, void *, void *, 
                                       const char *, const char *, int, int, 
                                       int, void *, void *, int, nglRes *);

extern nglPlotId ngl_add_polymarker_wrap(int, nglPlotId *, void *, void *, 
                                         const char *, const char *, int, 
                                         int, int, void *, void *, int, 
                                         nglRes *);

extern nglPlotId ngl_add_polygon_wrap(int, nglPlotId *, void *, void *, 
                                      const char *, const char *, int, int,
                                      int, void *, void *, int, nglRes *);

extern nglPlotId ngl_add_text_wrap(int, nglPlotId *, char *, void *, void *, 
                                   const char *, const char *, int, int,
                                   nglRes *special_res);

extern void ngl_draw_colormap_wrap(int);

extern void ngl_panel_wrap(int, nglPlotId *, int, int *, int, int, nglRes *);
