#
#  Import NumPy.
#
import Numeric

#
#  Import Ngl support functions.
#
from Ngl import *

#
#  Import all names from the NetCDF module.
#
from Scientific.IO.NetCDF import *

#
# Open a netCDF file containing storm data.
#
dirc  = ncargpath("data")
tfile = NetCDFFile(dirc + "/cdf/Tstorm.cdf","r")

#
# Access the temperature arrays for the first 6 time steps.
#
nplots = 6
temp = tfile.variables["t"]

#
# Save lat and lon to Numeric arrays.
#
lat = tfile.variables["lat"][:]
lon = tfile.variables["lon"][:]

#
# Define a color map.
#
cmap = Numeric.array([[1.00,1.00,1.00],[0.00,0.00,0.00],[1.00,.000,.000],\
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
                      Numeric.Float0)
                   
rlist = Resources()
rlist.wkColorMap = cmap
wks = ngl_open_wks("x11","panel1",rlist)  # Open an X11 workstation.

#
# The next set of resources will apply to four plots.
#
resources = Resources()
resources.nglDraw  = False
resources.nglFrame = False

#
#  Set the scalarfield missing value if temp has one specified.
#
if hasattr(temp,"_FillValue"):
  resources.sfMissingValueV = temp._FillValue

#
# Loop through four of the timesteps and create each plot. Title each
# plot according to which timestep it is.
#
plot = []
resources.cnFillOn       = True    # Turn on contour fill.

for i in range(0,nplots):
  resources.tiMainString  = "Temperature at time = " + str(i)
  plot.append(ngl_contour(wks,temp[i,:,:],resources))


ngl_panel(wks,plot[0:4],[2,2]) # Draw 2 rows/2 columns of plots.

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

resources.mpLimitMode    = "LatLon"  # Limit portion of map that is viewed.
resources.mpMinLatF      = min(lat)
resources.mpMaxLatF      = max(lat)
resources.mpMinLonF      = min(lon)
resources.mpMaxLonF      = max(lon)
resources.pmLabelBarDisplayMode = "Never"

resources.mpPerimOn       = True    # Turn on map perimeter.
resources.mpGridAndLimbOn = False   # Turn off map grid.

plot = []
for i in range(0,nplots):
  plot.append(ngl_contour_map(wks,temp[i,:,:],resources))

panelres                = Resources()
panelres.nglFrame       = False # Don't advance the frame.

#
# Set up some labelbar resources.  Set nglPanelLabelBar to True to
# indicate you want to draw a common labelbar at the bottom of the
# plots. 
#
panelres.nglPanelLabelBar   = True
panelres.lbLabelFont        = "helvetica-bold" # Labelbar font
panelres.lbLabelStride      = 2                # Draw every other label
panelres.nglPanelLabelBarLabelFontHeightF = 0.015   # Labelbar font height
panelres.nglPanelLabelBarHeightF     = 0.1750   # Height of labelbar
panelres.nglPanelLabelBarWidthF      = 0.700   # Width of labelbar
panelres.nglPanelTop                 = 0.935
panelres.nglPanelFigureStrings     = ["A","B","C","D","E","F"]
panelres.nglPanelFigureStringsJust = "BottomRight"
# panelres.nglPanelLabelBarLabelAutoStride     = False
# panelres.nglPanelFigureStringsOrthogonalPosF = 0.05
# panelres.nglPanelFigureStringsFontHeightF    = 0.05
# panelres.nglPanelFigureStringsBackgroundFillColor = 1
# panelres.nglPanelFigureStringsParallelPosF   = 0.025

#
# Draw 3 rows and 2 columns of plots.
#
ngl_panel(wks,plot[0:nplots],[3,2],panelres)  

#
# Draw two titles at the top.
#
textres               = Resources()
textres.txFontHeightF = 0.025   # Size of title.

ngl_text_ndc(wks,":F26:Temperature (K) at every six hours",0.5,.97,textres)

textres.txFontHeightF = 0.02    # Make second title slightly smaller.

ngl_text_ndc(wks,":F26:January 1996",0.5,.935,textres)

ngl_frame(wks)   # Advance the frame.

ngl_end()
