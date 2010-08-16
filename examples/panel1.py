#
#  File:
#    panel1.py
#
#  Synopsis:
#    Illustrates how to draw several visualizations on a single frame.
#
#  Category:
#    Paneling
#    Contours over maps
#    Label bar
#
#  Author:
#    Fred Clare (based on a code of Mary Haley)
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    Illustrates how to put multiple plots on a page and how
#    to have each plot have its own label bar or have a common
#    label bar for all plots on the page.
#
#  Effects illustrated:
#    o  How to put multiple plots on a page.
#    o  Drawing contours over maps.
#    o  Some special paneling resources.
#    o  Label bars on individual plots and a common label bar for all plots.
#    o  How to put a label on each individual plot on a multi-plot frame.
# 
#  Output:
#    This examples produces three visualizations:
#      1.) Four contour plots on a single frame with individual label bars.
#      2.) Same as 1.), but with additional white space around the plots.
#      3.) Six plots on a page with labels applied and a common label bar.
#
#  Notes:
#     

#
#  Import numpy.
#
import numpy, os

#
#  Import Ngl support functions.
#
import Ngl

#
#  Import Nio.
#
import Nio

#
# Open a netCDF file containing storm data.
#
dirc  = Ngl.pynglpath("data")
tfile = Nio.open_file(os.path.join(dirc,"cdf","Tstorm.cdf"))

#
# Access the temperature arrays for the first 6 time steps.
#
nplots = 6
temp = tfile.variables["t"]

#
# Save lat and lon to numpy arrays.
#
lat = tfile.variables["lat"][:]
lon = tfile.variables["lon"][:]

#
# Define a color map.
#
cmap = numpy.array([[1.00,1.00,1.00],[0.00,0.00,0.00],[1.00,.000,.000],\
                    [.950,.010,.000],[.870,.050,.000],[.800,.090,.000],\
                    [.700,.090,.000],[.700,.120,.000],[.700,.180,.000],\
                    [.700,.260,.000],[.700,.285,.000],[.680,.330,.000],\
                    [.570,.420,.000],[.560,.530,.000],[.550,.550,.000],\
                    [.130,.570,.000],[.060,.680,.000],[.000,.690,.000],\
                    [.000,.700,.100],[.000,.600,.300],[.000,.500,.500],\
                    [.000,.400,.700],[.000,.300,.700],[.000,.200,.700],\
                    [.000,.100,.700],[.000,.000,.700],[.100,.100,.700],\
                    [.200,.200,.700],[.300,.300,.700],[.420,.400,.700],\
                    [.560,.500,.700],[.610,.600,.700],[.700,.700,.700]],
                    'f')
                   
#
# Set the color map and open a workstation.
#
rlist               = Ngl.Resources()
rlist.wkColorMap    = cmap

wks_type = "ps"
if(wks_type == "ps" or wks_type == "pdf"):
  rlist.wkOrientation = "Portrait"      # For PS or PDF output only.

wks = Ngl.open_wks(wks_type,"panel1",rlist)  # Open an X11 workstation.

#
# Turn off draw for the individual plots, since we are going to
# panel them later.
#
resources          = Ngl.Resources()
resources.nglDraw  = False
resources.nglFrame = False

#
# Loop through the timesteps and create each plot, titling each
# one according to which timestep it is.
#
plot = []
resources.cnFillOn            = True    # Turn on contour fill.
resources.lbLabelStride       = 2       # Label every other box

#
# Set some font heights to make them slightly bigger than the default.
# Turn off nglScale, because this resource wants to set the axes font
# heights for you.
#
resources.nglScale               = False
resources.cnLineLabelFontHeightF = 0.025
resources.tiMainFontHeightF      = 0.037
resources.lbLabelFontHeightF     = 0.032
resources.tmXBLabelFontHeightF   = 0.030
resources.tmYLLabelFontHeightF   = 0.030

for i in range(0,nplots):
  resources.tiMainString  = "Temperature at time = " + str(i)
  plot.append(Ngl.contour(wks,temp[i,:,:],resources))

Ngl.panel(wks,plot[0:4],[2,2])    # Draw 2 rows/2 columns of plots.

#
# Now add some extra white space around each plot.
#

panelres                            = Ngl.Resources()
panelres.nglPanelYWhiteSpacePercent = 5.
panelres.nglPanelXWhiteSpacePercent = 5.
Ngl.panel(wks,plot[0:4],[2,2],panelres)    # Draw 2 rows/2 columns of plots.

#
# This section will set resources for drawing contour plots over a map.
#
del resources.tiMainString  # Don't set a main title.

resources.sfXArray       = lon  # Portion of map on which to overlay
resources.sfYArray       = lat  # contour plot.

resources.cnLineLabelsOn = False   # Turn off contour line labels.
resources.cnLinesOn      = False   # Turn off contour lines.
resources.cnFillOn       = True    # Turn on contour fill.

resources.cnLevelSelectionMode = "ManualLevels"  # Select contour levels.
resources.cnMinLevelValF       = 245.
resources.cnMaxLevelValF       = 302.5
resources.cnLevelSpacingF      =   2.5

resources.tmXBLabelFontHeightF   = 0.020
resources.tmYLLabelFontHeightF   = 0.020

resources.mpLimitMode    = "LatLon"  # Limit portion of map that is viewed.
resources.mpMinLatF      = float(min(lat))
resources.mpMaxLatF      = float(max(lat))
resources.mpMinLonF      = float(min(lon))
resources.mpMaxLonF      = float(max(lon))

resources.pmLabelBarDisplayMode = "Never"   # Turn off labelbar, since we
                                            # will use a global labelbar
                                            # in the panel.

resources.mpPerimOn       = True            # Turn on map perimeter.
resources.mpGridAndLimbOn = False           # Turn off map grid.

plot = []
for i in range(0,nplots):
  plot.append(Ngl.contour_map(wks,temp[i,:,:],resources))

#
# Draw two titles at the top. Draw these before the panel stuff,
# so that the maximization works properly.
#
textres               = Ngl.Resources()
textres.txFontHeightF = 0.025   # Size of title.

Ngl.text_ndc(wks,"~F22~Temperature (K) at every six hours",0.5,.97,textres)

textres.txFontHeightF = 0.02    # Make second title slightly smaller.

Ngl.text_ndc(wks,"~F22~January 1996",0.5,.935,textres)

#
# Set some resources for the paneled plots.
#
del panelres
panelres          = Ngl.Resources()

#
# Set up some labelbar resources.  Set nglPanelLabelBar to True to
# indicate you want to draw a common labelbar at the bottom of the
# plots. 
#
panelres.nglPanelLabelBar                 = True     # Turn on panel labelbar
panelres.nglPanelLabelBarLabelFontHeightF = 0.015    # Labelbar font height
panelres.nglPanelLabelBarHeightF          = 0.1750   # Height of labelbar
panelres.nglPanelLabelBarWidthF           = 0.700    # Width of labelbar
panelres.lbLabelFont                      = "helvetica-bold" # Labelbar font
panelres.nglPanelTop                      = 0.935
panelres.nglPanelFigureStrings            = ["A","B","C","D","E","F"]
panelres.nglPanelFigureStringsJust        = "BottomRight"

#
# You can have PyNGL selection the best paper orientation for
# the shape of plots you are drawing.  This resource is for PDF or
# PS output only.
#
if(wks_type == "ps" or wks_type == "pdf"):
  panelres.nglPaperOrientation = "Auto"   

#
# Draw 3 rows and 2 columns of plots.
#
Ngl.panel(wks,plot[0:nplots],[3,2],panelres)  

Ngl.end()
