#
#  File:
#    scatter1.py
#
#  Synopsis:
#    Draws a scatter visualization using polymarkers.
#
#  Categories:
#    Polymarkers
#    Tickmarks
#    Text
#    XY plots
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    October, 2004
#
#  Description:
#    This example reads in some dummy of XY coordinates and
#    associated color indices in the range from 1 to 4.  A
#    scatter plot is drawn using markers colored and sized
#    as per the color indices.  A quadratic least squares fit 
#    is calculated.
#
#  Effects illustrated:
#    o  Polymarkers.
#    o  Least squares fit.
#    o  XY plot.
#    o  Tickmark resource settings.
# 
#  Output:
#    A single visualization is produced.
#
#  Notes:
#    1.)  This visualization is similar in appearance to one provided 
#         by Joel Norris of GFDL, but dummmy data are used.
#    2.)  This example requires importing the Scientific package.
#     

#
#  Import Numeric.
#
import Numeric

#
# Import Nio for reading netCDF files.
#
import Nio

#
#  From Scientific import the the polynomial least squares function.
#
# You can download ScientificPython from:
#
#  http://starship.python.net/~hinsen/ScientificPython/
#
from Scientific.Functions.LeastSquares import polynomialLeastSquaresFit

#

#
#  Import PyNGL.
#
import Ngl

#
# This plot is very similar to one done by Joel Norris of GFDL. The
# original data is no longer available, so dummy data is used in this
# case.
#
#  Read the scattered data and extract the x, y, and color variables.
#
dirc  = Ngl.pynglpath("data")
ncdf = Nio.open_file(dirc + "/cdf/scatter1.nc","r")
x = ncdf.variables["x"][:]
y = ncdf.variables["y"][:]
colors = ncdf.variables["colors"][:]
color_index = colors.astype(Numeric.Int)

#
#  Specify a color map and open an output workstation.
#
cmap = Numeric.array([[1., 1., 1.], [0., 0., 0.], [1., 0., 0.], \
                      [0., 0., 1.], [0., 1., 0.]], Numeric.Float0)
rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"scatter1",rlist)  # Open a workstation.
  
#
#  Label the plot.
#
resources = Ngl.Resources()
resources.tiMainString           = "2000 Mar 19 1040-1725Z"
resources.tiMainFont             = "helvetica-bold"
resources.tiXAxisString          = "Cloud Thickness (m)"
resources.tiXAxisFont            = "helvetica-bold"
resources.tiYAxisString          = "Liquid Water Path (mm)"
resources.tiYAxisFont            = "helvetica-bold"
resources.tiXAxisFontHeightF     = 0.025     # Change the font size.
resources.tiYAxisFontHeightF     = 0.025
  
resources.xyLineColors           = [1]       # Set the line colors.
resources.xyMonoLineThickness    = True      # Set the line colors.
resources.xyLineThicknessF       = 3.0       # Triple the width.
  
resources.tmXBMajorLengthF        = 0.02    # Force tickmarks to point
resources.tmXBMajorOutwardLengthF = 0.02    # out by making the outward
resources.tmXBMinorLengthF        = 0.01    # Force tickmarks to point
resources.tmXBMinorOutwardLengthF = 0.01    # out by making the outward
resources.tmXBLabelFont           = "helvetica-bold"
resources.tmYLMajorLengthF        = 0.02    # tick length equal to the
resources.tmYLMajorOutwardLengthF = 0.02    # total tick length
resources.tmYLMinorLengthF        = 0.01
resources.tmYLMinorOutwardLengthF = 0.01
resources.tmYLLabelFont           = "helvetica-bold"

resources.trXMaxF                 = 800.   # Force the X-axis to run to 800.
resources.trYMaxF                 = 0.35   # Force the Y-axis to run to 0.35
  
txres = Ngl.Resources()
txres.txFont = "helvetica-bold"
txres.txFontHeightF = 0.02
txres.txJust = "CenterLeft"
Ngl.text_ndc(wks,"LWP=0.5 x 1.55 x 10~S~-6~N~ x Thickness~S1~2", \
                  0.24,0.85,txres)

xpos = 0.245
delx = 0.02
ypos = 0.78
dely = 0.035
gsres = Ngl.Resources()
gsres.gsMarkerIndex = 16        # Change marker type to a filled circle.

gsres.gsMarkerColor = "black"   # Change marker color.
gsres.gsMarkerSizeF =  4.0      # Increase marker size.
Ngl.polymarker_ndc(wks,[xpos],[ypos],gsres)         # Draw a polymarker.
txres.txFontColor = "black"
Ngl.text_ndc(wks,"20 sec",xpos+delx,ypos,txres)

gsres.gsMarkerColor = "red"     # Change marker color.
gsres.gsMarkerSizeF =  7.       # Increase marker size.
Ngl.polymarker_ndc(wks,[xpos],[ypos-dely],gsres)    # Draw a polymarker.
txres.txFontColor = "red"
Ngl.text_ndc(wks,"1 min",xpos+delx,ypos-dely,txres)

gsres.gsMarkerColor = "blue"    # Change marker color.
gsres.gsMarkerSizeF = 10.       # Increase marker size.
Ngl.polymarker_ndc(wks,[xpos],[ypos-2*dely],gsres)  # Draw a polymarker.
txres.txFontColor = "blue"
Ngl.text_ndc(wks,"5 min",xpos+delx,ypos-2*dely,txres)

gsres.gsMarkerColor = "green"   # Change marker color.
gsres.gsMarkerSizeF = 13.       # Increase marker size.
Ngl.polymarker_ndc(wks,[xpos],[ypos-3*dely],gsres)  # Draw a polymarker.
txres.txFontColor = "green"
Ngl.text_ndc(wks,"20 min",xpos+delx,ypos-3*dely,txres)


#
#  Suppress frame call.
#
resources.nglFrame = False

#
#  Put the data in the correct format for the least squares 
#  function and do the fit.
#
data = []
for j in xrange(len(x)):
  data.append((x[j],y[j]))
params = [0.,0.,1.e-7]
a = polynomialLeastSquaresFit(params, data)

#
#  Draw the least squares quadratic curve.
#
num  = 301
delx = 1000./num
x    = Numeric.zeros(num,Numeric.Float0)
y    = Numeric.zeros(num,Numeric.Float0)
for i in xrange(num):
  x[i] = float(i)*delx
  y[i] = a[0][0]+a[0][1]*x[i]+a[0][2]*x[i]*x[i]
plot = Ngl.xy(wks,x,y,resources) # Draw least squares quadratic.

#
#  Draw a marker at each data point using the specified color index.
#
mres = Ngl.Resources()
mres.gsMarkerIndex = 1
for i in xrange(len(data)):
  if (color_index[i] == 1): 
    mres.gsMarkerColor = "black"
    mres.gsMarkerSizeF =  0.01 #Increase marker size by a factor of 10.
  elif (color_index[i] == 2):
    mres.gsMarkerColor = "red"
    mres.gsMarkerSizeF =  0.02 #Increase marker size by a factor of 10.
  elif (color_index[i] == 3):
    mres.gsMarkerColor = "blue"
    mres.gsMarkerSizeF =  0.04 #Increase marker size by a factor of 10.
  elif (color_index[i] == 4):
    mres.gsMarkerColor = "green"
    mres.gsMarkerSizeF =  0.05 #Increase marker size by a factor of 10.
  Ngl.polymarker(wks,plot,[data[i][0]],[data[i][1]],mres) # Draw polymarkers.

Ngl.frame(wks)

# Clean up and end.
del plot 
del resources
Ngl.end()
