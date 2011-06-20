#include <ncarg/ncargC.h>
#include <ncarg/hlu/hlu.h>
#include <ncarg/hlu/NresDB.h>
#include <ncarg/hlu/ResList.h>
#include <ncarg/hlu/App.h>
#include <ncarg/hlu/XWorkstation.h>
#include <ncarg/hlu/NcgmWorkstation.h>
#include <ncarg/hlu/PSWorkstation.h>
#include <ncarg/hlu/PDFWorkstation.h>
#include <ncarg/hlu/CairoWorkstation.h>
#include <ncarg/hlu/ImageWorkstation.h>
#include <ncarg/hlu/ContourPlot.h>
#include <ncarg/hlu/XyPlot.h>
#include <ncarg/hlu/MapPlot.h>
#include <ncarg/hlu/VectorPlot.h>
#include <ncarg/hlu/LabelBar.h>
#include <ncarg/hlu/Legend.h>
#include <ncarg/hlu/LogLinPlot.h>
#include <ncarg/hlu/StreamlinePlot.h>
#include <ncarg/hlu/ScalarField.h>
#include <ncarg/hlu/MeshScalarField.h>
#include <ncarg/hlu/CoordArrays.h>
#include <ncarg/hlu/VectorField.h>
#include <ncarg/hlu/TextItem.h>
#include <ncarg/hlu/GraphicStyle.h>
#include <ncarg/hlu/LogLinPlot.h>
#include <ncarg/hlu/IrregularPlot.h>
#include <ncarg/hlu/Primitive.h>
#include <ncarg/hlu/TransObj.h>
#include <ncarg/hlu/View.h>

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
 * Special resource for figure strings.
 */
  NhlString *nglPanelFigureStrings;
  int     nglPanelFigureStringsCount;
  int     nglPanelFigureStringsJust;
  float   nglPanelFigureStringsOrthogonalPosF;
  float   nglPanelFigureStringsParallelPosF;
  int     nglPanelFigureStringsPerimOn;
  int     nglPanelFigureStringsBackgroundFillColor;
  float   nglPanelFigureStringsFontHeightF;

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
  int   nglPanelLabelBarLabelAutoStride;
  int   nglPanelLabelBarAlignment;
  float nglPanelLabelBarLabelFontHeightF;
  float nglPanelLabelBarOrthogonalPosF;
  float nglPanelLabelBarParallelPosF;

  NhlString nglAppResFileName;

/*
 * Special resources for linearizing or logizing one or both axes.
 */
  int nglXAxisType;
  int nglYAxisType;

  int nglPointTickmarksOutward;
/*
 * Special resources for X and/or Y reference lines.
 */
  float nglXRefLine;
  float nglYRefLine;
  float nglXRefLineThicknessF;
  float nglYRefLineThicknessF;
  int   nglXRefLineColor;
  int   nglYRefLineColor;

/*
 * Special resources for masking lambert conformal projections
 */
  int nglMaskLambertConformal;
  int nglMaskLambertConformalOutlineOn;
} nglRes;

/*
 * Define a structure to hold all the possible types of objects
 * that the functions create.
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
  int   *labelbar;
  int   *legend;
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
  int   nlabelbar;
  int   nlegend;
  int   ncafield;
  int   nsffield;
  int   nvffield;
} nglPlotId;

/*
 *  Structure for passing resource list information.
 */
typedef struct _ResInfo {
  int id;
  int nstrings;
  char **strings;
} ResInfo;

/*
 * Supplemental functions. 
 */

extern int imax_array(int *, int);
extern float xmax_array(float *, int);
extern float xmin_array(float *, int);
extern float *fspan(float, float, int);
extern int *ispan(int, int, int);
extern ng_size_t *convert_dims(int *,int);
extern int is_res_set(ResInfo *, char *);
extern void getbb (int pid, float *t, float *b, float *l, float *r);
extern void compute_ps_device_coords(int, nglPlotId *, int, nglRes *);

extern void maximize_plots(int, nglPlotId *, int, int, nglRes *);

extern void overlay_on_irregular(int, nglPlotId *asplot, ResInfo *, 
                                 ResInfo *, nglRes*);

extern void spread_colors(int, int, int, int, char*, char*, int);

extern void scale_plot(int,ResInfo *,int);

extern void add_ref_line(int,int,nglRes*);

extern void point_tickmarks_out(int,ResInfo *);

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

extern int open_wks_wrap(const char *, const char *, ResInfo *, ResInfo *,
                         nglRes *);

/*
 * Plotting routines.
 */

extern nglPlotId blank_plot_wrap(int, ResInfo *, nglRes *);

extern nglPlotId contour_wrap(int, void *, const char *, int, int, 
                              int, void *, const char *, int, void *, 
                              const char *, int, void *, ResInfo *,
                              ResInfo *, ResInfo *, nglRes *);

extern nglPlotId xy_wrap(int, void *, void *, const char *,
                         const char *, int, int *, int, int *, int, int,
                         void *, void *, ResInfo *, ResInfo *,
                         ResInfo *, nglRes *);

extern nglPlotId y_wrap(int, void *, const char *, int, int *, int, 
                        void *, ResInfo *, ResInfo *, ResInfo *,
                        nglRes *);

extern nglPlotId vector_wrap(int, void *, void *, const char *, 
                             const char *, int, int, int, void *, 
                             const char *, int, void *, const char *, 
                             int, int, void *, void *, ResInfo *,
                             ResInfo *, ResInfo *, nglRes *);

extern nglPlotId streamline_wrap(int, void *, void *, const char *, 
                                 const char *, int, int, int, void *, 
                                 const char *, int, void *, const char *,
                                 int, int, void *, void *, ResInfo *,
                                 ResInfo *, ResInfo *, nglRes *);

extern nglPlotId map_wrap(int, ResInfo *, nglRes *);

extern nglPlotId contour_map_wrap(int, void *, const char *, int, int, 
                                  int, void *, const char *, int, 
                                  void *, const char *, int, void *, 
                                  ResInfo *, ResInfo *, ResInfo *,
                                  nglRes *);

extern nglPlotId vector_map_wrap(int, void *, void *, const char *, 
                                 const char *, int, int, int, void *, 
                                 const char *, int, void *, 
                                 const char *, int, int, void *, void *,
                                 ResInfo *, ResInfo *, ResInfo *,
                                 nglRes *);

extern nglPlotId streamline_map_wrap(int, void *, void *, const char *, 
                                     const char *, int, int, int, 
                                     void *, const char *, int, void *,
                                     const char *, int, int, void *, 
                                     void *, ResInfo *, ResInfo *,
                                     ResInfo *, nglRes *);

extern nglPlotId streamline_scalar_wrap(int, void *, void *, void *, 
                                        const char *, const char *, 
                                        const char *, int, int, int, 
                                        void *, const char *, int, void *, 
                                        const char *, int, int, int, void *, 
                                        void *, void *, ResInfo *, ResInfo *,
                                        ResInfo *, ResInfo *, nglRes *);

extern nglPlotId streamline_scalar_map_wrap(int, void *, void *, void *, 
                                           const char *, const char *, 
                                           const char *, int, int, int,
                                           void *, const char *, int, 
                                           void *, const char *, int, int, 
                                           int, void *, void *, void *, 
                                           ResInfo *, ResInfo *,
                                           ResInfo *, ResInfo *, nglRes *);

extern nglPlotId vector_scalar_wrap(int, void *, void *, void *, 
                                    const char *, const char *, 
                                    const char *, int, int, int, 
                                    void *, const char *, int, void *, 
                                    const char *, int, int, int, void *, 
                                    void *, void *, ResInfo *, ResInfo *,
                                    ResInfo *, ResInfo *, nglRes *);

extern nglPlotId vector_scalar_map_wrap(int, void *, void *, void *, 
                                        const char *, const char *, 
                                        const char *, int, int, int,
                                        void *, const char *, int, 
                                        void *, const char *, int, int, 
                                        int, void *, void *, void *, 
                                        ResInfo *, ResInfo *,
                                        ResInfo *, ResInfo *, nglRes *);

/*
 * Text routines.
 */

extern nglPlotId text_ndc_wrap(int, char *, void *, void *,
                               const char *, const char *, ResInfo *,
                               nglRes *);

extern nglPlotId text_wrap(int, nglPlotId *, char *, void *, void *,
                           const char *, const char *, ResInfo *,
                           nglRes *);

/*
 * Labelbar routine.
 */
extern nglPlotId labelbar_ndc_wrap(int, int, NhlString *, int, void *, void *,
                                   const char *type_x, const char *type_y,
                                   ResInfo *, nglRes *);

/*
 * Legend routine.
 */
extern nglPlotId legend_ndc_wrap(int, int, NhlString *, int, void *, void *,
                                 const char *type_x, const char *type_y,
                                 ResInfo *, nglRes *);


/*
 * Primitive drawing routines.
 */

extern void poly_wrap(int, nglPlotId *, void *, void *,
                      const char *type_x, const char *type_y, int, int,
                      int, void *, void*, NhlPolyType, ResInfo *,
                      nglRes *);


extern nglPlotId add_poly_wrap(int, nglPlotId *, void *, void *, 
                               const char *, const char *, int, int,
                               int, int, void *, void *,NhlPolyType,
                               ResInfo *, nglRes *);

extern void polymarker_ndc_wrap(int, void *, void *, const char *, 
                                const char *, int, int, int, void *,
                                void *, ResInfo *, nglRes *);

extern void polyline_ndc_wrap(int, void *, void *, const char *, 
                              const char *, int, int, int, void *,
                              void *, ResInfo *, nglRes *);

extern void polygon_ndc_wrap(int, void *, void *, const char *, 
                             const char *, int, int, int, void *, 
                             void *, ResInfo *, nglRes *);

extern void polymarker_wrap(int, nglPlotId *, void *, void *,
                            const char *, const char *, int, int, int,
                            void *, void *, ResInfo *, nglRes *);

extern void polyline_wrap(int, nglPlotId *, void *, void *, const char *,
                          const char *, int, int, int, void *, void *,
                          ResInfo *, nglRes *);

extern void polygon_wrap(int, nglPlotId *, void *, void *, const char *, 
                         const char *, int, int, int, void *, void *, 
                         ResInfo *, nglRes *);

extern nglPlotId add_polyline_wrap(int, nglPlotId *, void *, void *, 
                                   const char *, const char *, int, int, 
                                   int, void *, void *, ResInfo*,
                                   nglRes *);

extern nglPlotId add_polymarker_wrap(int, nglPlotId *, void *, void *, 
                                     const char *, const char *, int, 
                                     int, int, void *, void *,
                                     ResInfo *, nglRes *);

extern nglPlotId add_polygon_wrap(int, nglPlotId *, void *, void *, 
                                  const char *, const char *, int, int,
                                  int, void *, void *, ResInfo *,
                                  nglRes *);

extern nglPlotId add_text_wrap(int, nglPlotId *, char *, void *, void *, 
                               const char *, const char *, ResInfo *,
                               ResInfo *, nglRes *);

extern void draw_colormap_wrap(int);

extern void panel_wrap(int, nglPlotId *, int, int *, int, ResInfo *,
                       ResInfo *, nglRes *);

