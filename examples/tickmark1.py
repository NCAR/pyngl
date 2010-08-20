#
#  File:
#    tickmark1.py
#
#  Synopsis:
#    Illustrates how to label minor tickmarks differently, by
#    overlaying a blank plot that contains the labeled minor
#    tickmarks.
#
#  Categories:
#    tickmarks
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    August 2010, from an NCL script
#
#  Description:
#    This example also shows how to attach a curve with missing values.
#
#  Effects illustrated:
#    o  Customizing tickmark labels
#    o  Adding a curve with missing values
#    o  Generating dummy data
#    o  Creating a blank plot
#
#  Output:
#     This example produces similar visualizations, the second containing
#     a curve with missing values.
#
import Ngl, numpy, random

#---Import masked arrays
from numpy import ma


#---Generate some dummy data.
random.seed(10)   # set a seed for the random number generator
years = numpy.arange(1950,2005,1)
npts1 = len(years)
y1    = numpy.zeros([npts1],'f')
for i in range(npts1):
  y1[i] = 4.0 - 8.*random.random()

#---Start the graphics section.
wks_type = "ps"
wks      = Ngl.open_wks(wks_type,"tickmark1")

#---Set some plot resources.
res = Ngl.Resources()

res.vpWidthF  = 0.8           # Set width and height of plot.
res.vpHeightF = 0.3

res.trYMinF = -4.0            # Set minimum Y-axis value.
res.trYMaxF =  4.0            # set maximum Y-axis value.
res.trXMinF = 1949            # Set minimum X-axis value.
res.trXMaxF = 2006            # Set maximum X-axis value.

res.nglPointTickmarksOutward = True   # Point tickmarks outward.

res.tmYROn = False            # Turn off right tickmarks.
res.tmXTOn = False            # Turn off top tickmarks.

res.tiMainString = "Labeling major and minor tickmarks"

res.nglDraw  = False          # Don't draw plot or advance frame.
res.nglFrame = False

plot = Ngl.xy(wks,years,y1,res)

#
# Create a blank plot with the special labels that we want. This 
# plot will be overlaid on the original XY plot.
#
# Make sure the blank plot is drawn in same location, by setting
# the viewport coordinates to the same values.
#

#---Retrieve viewport coordinates and set them for blank plot.
bres           = Ngl.Resources()
bres.vpXF      = Ngl.get_float(plot,"vpXF")
bres.vpYF      = Ngl.get_float(plot,"vpYF")
bres.vpHeightF = Ngl.get_float(plot,"vpHeightF")
bres.vpWidthF  = Ngl.get_float(plot,"vpWidthF" )

#
# Create the values that represent the locations of the minor tickmarks
# in the previous plot.
#
values = numpy.arange(1950,2005,2)

#
# Create an array of labels for these locations. Since we already
# have labels at 1950, 1960, etc, set these to "".
#

lvalues = numpy.where(values >= 2000,values-2000,values-1900).tolist()

labels  = []
for l in lvalues:
  if l%10:
    labels.append("'%0.2i" % l)
  else:
    labels.append('')

#
# Set some resources to customize the tickmarks so only the
# minor tickmarks are drawn.
#
bres.tmXBMode                = "Explicit"
bres.tmXBValues              = values
bres.tmXBLabels              = labels
bres.tmXBLabelFontHeightF    = 0.01    # Make these labels smaller.
bres.tmXBMajorOutwardLengthF = 0.0     # Don't draw tickmarks b/c they
bres.tmXBMajorLengthF        = 0.0     # were drawn on previous plot.
bres.tmXBLabelDeltaF         = 0.6     # Move label away from tickmarks.

bres.tmXBLabelFontColor      = "Brown"

bres.tmYROn = False            # Turn off right tickmarks.
bres.tmXTOn = False            # Turn off top tickmarks.
bres.tmYLOn = False            # Turn off left tickmarks.

#---Create the blank plot.
blank = Ngl.blank_plot(wks,bres)

#---Overlay on existing XY plot.
Ngl.overlay(plot,blank)

#---Drawing the original plot also draws the overlaid blank plot
Ngl.draw(plot)
Ngl.frame(wks)

#
# Second plot section.
#

#---Create dummy data for additional curve.
npts2 = 40
y2    = numpy.zeros([npts2],'f')
for i in range(npts2):
  y2[i] = 3.0 - 6.*random.random()

#---Force some of the y2 points be missing
y2 = ma.array(y2,mask=[0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,\
                       1,1,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,1,1,1])

#---Set some resources for the polyline
gsres                   = Ngl.Resources()
gsres.gsLineColor       = "Blue"
gsres.gsLineThicknessF  = 3.

#---Add the line to existing plot
x = Ngl.fspan(min(years),max(years),npts2)
prim1 = Ngl.add_polyline(wks,plot,x,y2,gsres)

#---Change the main title of original plot.
srlist = Ngl.Resources()
srlist.tiMainString  = "Adding a blue curve with missing values"
Ngl.set_values(plot,srlist)

#---Draw plot (and its attached line)
Ngl.draw(plot)
Ngl.frame(wks)

Ngl.end()
