#
#  Import NumPy.
#
import Numeric,sys

#
#  Import NGL support functions.
#
from Ngl import *

def addcyclic(data):
#
# Add a cyclic point in "x" to a 2D array
# for a lat/lon plot "x"  corresponds to "lon"
#                    "ny" corresponds to "nlat"
#                    "mx" corresponds to "mlon"
#
  if (type(data) != type(Numeric.array([0]))):
    print "addcyclic: input must be a Numeric array."
    sys.exit()

  dims = data.shape
  if (len(dims) != 2):
    print "addcyclic: input must be a 2D Numeric array."
    sys.exit()
    
  ny   = dims[0]
  mx   = dims[1]
  mx1  = mx+1

  newdata = Numeric.zeros((ny,mx1),Numeric.Float0)
  
  newdata[:,0:mx] = data
  newdata[:,mx]   = data[:,0]

  return newdata

#
#  Define a color table and open a workstation.
#
cmap = Numeric.zeros((2,3),Numeric.Float0)
cmap[0] = [1.,1.,1.]
cmap[1] = [0.,0.,0.]
rlist = Resources()
rlist.wkColorMap = cmap
wks = ngl_open_wks("ncgm","ngl11p",rlist)

dirc     = ncargpath("data")
data     = ngl_asciiread(dirc+"/asc/u.cocos",(39,14),"float")

pressure  = data[:,0]    # First column of data is pressure (mb).
height    = data[:,1]    # Second column is height (km).
u         = data[:,2:14] # Rest of columns are climatological zonal winds
                           # (u: m/s)
unew = addcyclic(u)  # Copy fi

#----------- Begin first plot -----------------------------------------

resources = Resources()

resources.tiMainString   = ":F26:Cocos Island"   # Main title.
resources.tiYAxisString  = ":F25:Pressure (mb)"  # Y axes label.

resources.sfYCStartV = max(pressure)   # Indicate start and end of left
resources.sfYCEndV   = min(pressure)   # Y axes values.

resources.trYReverse  = True    # Reverse the Y values.
resources.trYLog      = True    # Use log scale.

resources.tmXBMode      = "Explicit"   # Define your own tick mark labels.
resources.tmXBLabelFont = 25           # Change font of labels.
resources.tmXBLabelFontHeightF = 0.015 # Change font height of labels.
resources.tmXBMinorOn   = False        # No minor tick marks.
resources.tmXBValues    = range(0,13,1) # Location to put tick mark labels
                                         # (13 points with January repeated).
resources.tmXBLabels    = ["Jan","Feb","Mar","Apr","May","Jun",\
                           "Jul","Aug","Sep","Oct","Nov","Dec","Jan"]

resources.tmYUseLeft    = False      # Keep right axis independent of left.
resources.tmYRLabelsOn  = True       # Turn on right axis labels.
resources.tmYRLabelFont = 25         # Change font of labels.
resources.tmYROn        = True       # Turn on right axis tick marks.
resources.tmYRMinorOn   = False      # No minor tick marks.

resources.tmYRMode      = "Explicit"  # Define own tick mark labels.
hnice = range(0,23,2)                 # Set range of "nice" height values.
pnice = ftcurv(height,pressure,hnice) # Calculate "nice" pressure values.
resources.tmYRValues    = pnice       # At each "nice" pressure value,
resources.tmYRLabels    = hnice       # put a "height" value label.

resources.tmYLMode      = "Explicit" # Define own tick mark labels.
resources.tmYLLabelFont = 25         # Change the font.
resources.tmYLValues    = [1000.,  800., 700., 500., 400., 300., \
                               250.,  200., 150., 100.,  50.]
resources.tmYLLabels    = ["1000","800","700","500","400","300", \
                               "250","200","150","100", "50"]
resources.tmYLMinorOn   = False        # No minor tick marks.

resources.cnFillOn          = True  # Turn on contour level fill.
resources.cnMonoFillColor   = True  # Use one fill color.
resources.cnMonoFillPattern = False # Use multiple fill patterns.

resources.cnLineLabelAngleF = 0. # Draw contour line labels right-side up.
resources.cnLevelSpacingF   = 1.0

resources.nglDraw  = False  # Don't draw the plot or advance the
resources.nglFrame = False  # frame in the call to ngl_contour.

resources.nglMaximize = False
resources.pmLabelBarDisplayMode = "Never"    # Turn off label bar.
contour = ngl_contour(wks, unew, resources)  # Create a contour plot.

levels = ngl_get_float_array(contour,"cnLevels")

patterns = Numeric.zeros((len(levels)+1),Numeric.Int)
patterns[:] = -1

for i in xrange(len(levels)):
  if (levels[i] <= -6.):
    patterns[i] = 5
  else:
    if (levels[i] > 0.):
      patterns[i] = 17.
patterns[-1]  = 17 # last pattern

rlist = Resources()
rlist.cnFillPatterns = patterns
rlist.cnFillScaleF = 0.8
ngl_set_values(contour,rlist)

ngl_draw(contour)  # Draw the contour plot.

txres               = Resources()    # Annotate plot with some text.
txres.txFontHeightF = 0.015
ngl_text_ndc(wks,":F25:U Component",  .270,.815,txres)
ngl_text_ndc(wks,":F25:(m-s:S:-1:N:)",.765,.815,txres)


txres.txFontHeightF = 0.025   # Label right Y axis.
txres.txAngleF      = 90.
ngl_text_ndc(wks,":F25:Height (km)",.89,.5,txres)

ngl_frame(wks) # Advance the frame.

#----------- Begin second plot -----------------------------------------

del resources 
resources = Resources()

resources.tiMainString  = ":F26:Cocos Island"
resources.tiXAxisString = ":F25:Month"
resources.tiYAxisString = ":F25:Amplitude (m/s)"

resources.tmXBMode      = "Explicit"   # Define your own tick mark labels.
resources.tmXBLabelFont = 25           # Change font of labels.
resources.tmXBLabelFontHeightF = 0.015 # Change font height of labels.
resources.tmXBMinorOn   = False        # No minor tick marks.
resources.tmXBValues    = range(0,13,1)# Values from 0 to 12.
resources.tmXBLabels    = ["Jan","Feb","Mar","Apr","May","Jun",\
                           "Jul","Aug","Sep","Oct","Nov","Dec","Jan"]

xy = ngl_xy(wks,range(0,13,1),unew,resources) # Create and draw an XY plot.

ngl_end()
