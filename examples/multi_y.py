#
#  File:
#    multi_y.py
#
#  Synopsis:
#    Shows how to conflate two XY plots and how to add and remove annotations.
#
#  Categories:
#    Annotations
#    XY plots
#    Special effects
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    August, 2005
#
#  Description:
#    This example shows:
#       1.)  how to merge two XY plots into a single visualization
#       2.)  how to do the same thing as in 1.) by specifying
#            the second plot as an annotation to the first
#       3.)  how to remove an annotation
#
#  Effects illustrated:
#    o  Reading data from an ASCII file.
#    o  Controlling two separate plots so that they will have the
#       same size an position on the frame.
#    o  Setting several tickmark resources to turn some off and
#       label others with user-specified labels and so forth.
#    o  XY plots.
#    o  Adding and removing annotations.
#    o  Creating your own dash pattern.
# 
#  Output:
#    Five visualizations are produced:
#      1.)  A simple XY plot.
#      2.)  A second XY plot.
#      3.)  An XY plot that merges the plots in 1.) and 2.).
#      4.)  The same thing as in 3.) accomplished by making the
#           second XY plot and annotation to the first.
#      5.)  The plot in 4.) with the annotation removed, reproducing
#           the plot in 1.).
#
#  Notes:
#     

#
#  Import Ngl support functions.
#
import Ngl

#
#  xy.asc has 4 vars of length 129 longitudes, u, v, and t.
#
#  The data is taken at 43N latitude.  Longitude is an index
#  1-129 standing for 0 deg - 360 deg in steps of 360/128.
#  u and v are in m/s, and t is in deg K.
#
dirc     = Ngl.pynglpath("data")
data     = Ngl.asciiread(dirc+"/asc/xy.asc",[516],"float")

lon  = data[0:512:4]    # First column of data is longitude.
u    = data[1:513:4]    # Second column is u
v    = data[2:514:4]    # Third column is v (not used in this example)
t    = data[3:515:4]    # Fourth column is t

lon = (lon-1.) * 360./128.
t   = (t-273.15) * 9./5. + 32.

#
# Open a workstation.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"multi_y")

#
# Set up two separate resource lists, since we'll be creating two
# different XY plots.
#
res1 = Ngl.Resources()
res2 = Ngl.Resources()

#
# Don't maximize the plot in the frame, because we are controlling the
# size and location of both plots with the vpXXX resources.
#

res1.nglMaximize = False
res1.vpHeightF   = 0.75
res1.vpWidthF    = 0.65
res1.vpXF        = 0.20
res1.vpYF        = 0.90

#
# Set some axis strings, including the main title.
#
res1.tiMainString     = "First plot"
res1.tiYAxisString    = "Temperature in C~S~o~N~"
res1.tiXAxisString    = "Longitude (degrees)"
res1.tiYAxisFontColor = "red"

#
# Change line color and thickness.
#
res1.xyLineColor      = "red"
res1.xyLineThicknessF = 2.0

#
# Set the actual limits of the X and Y axes.
#
res1.trXMinF     =   0.
res1.trXMaxF     =  360.
res1.trYMinF     =  -80.
res1.trYMaxF     =  -55.
res1.trYReverse  = False

#
# Set some tickmark resources. Turn off the right tickmarks completely.
# Label the left tickmarks with values we choose.
#
res1.tmYLMode     = "Explicit"
res1.tmYLValues   = [ -80., -75., -70., -65., -60., -55.]
res1.tmYLLabels   = ["-80","-75","-70","-65","-60","-55"]
res1.tmYUseLeft   = False
res1.tmYROn       = False
res1.tmYRMinorOn  = False
res1.tmXBMinorOn  = False

#
# Create and draw the first plot.  If you don't want this
# plot drawn, set:
#
# res1.nglDraw  = False
# res1.nglFrame = False
#
plot1 = Ngl.xy(wks,lon,t,res1)

#
# Define resources for second plot.
#
# There is no need to maximize the second plot either, because we
# are using the same size as the first plot.
#
res2.nglMaximize = False
res2.vpHeightF   = res1.vpHeightF
res2.vpWidthF    = res1.vpWidthF
res2.vpXF        = res1.vpXF
res2.vpYF        = res1.vpYF

#
# Since the two plots share the same X data, the two trX**** resources
# make sure the X axis limits are the same for the second plot.
#
res2.trXMinF = res1.trXMinF
res2.trXMaxF = res1.trXMaxF
res2.trYMaxF = 60.
res2.trYMinF = 10.

#
# Turn off the bottom and left tickmarks, since we will use the ones from
# the first plot.
#
res2.tmXBOn       = False
res2.tmXBLabelsOn = False
res2.tmXBMinorOn  = False
res2.tmYLOn       = False
res2.tmYLLabelsOn = False
res2.tmYLMinorOn  = False

#
# Turn on the right Y labels and tickmarks and move the axis string to
# the right.
#
res2.tmYRLabelsOn = True
res2.tmYROn       = True
res2.tmYUseLeft   = False  
res2.tmYRFormat   = "f"      # Gets rid of unnecessary trailing zeros

#
# Move the Y axis string to the right.
#
res2.tiYAxisString      = "U component of wind (m/s)"
res2.tiYAxisSide        = "Right"
res2.tiYAxisFontColor   = "purple"
res2.tiXAxisFontHeightF = Ngl.get_float(plot1,"tiXAxisFontHeightF")

#
# Make sure the font heights and tickmark lengths are the same as
# the first plot.
#
res2.tmYRLabelFontHeightF = Ngl.get_float(plot1,"tmYLLabelFontHeightF")
res2.tmYRMajorLengthF     = Ngl.get_float(plot1,"tmYLMajorLengthF")

#
# Change line pattern, color and thickness.
#
pattern = "$_____$_____$$_____$$_____$$_____$$_____$$___"
res2.xyDashPattern    = Ngl.new_dash_pattern(wks,pattern)
res2.xyLineColor      = "purple"
res2.xyLineThicknessF = 3.0

#
# Main title.
#
res2.tiMainString     = "Second plot"

#
# Create and draw the first plot.  If you don't want this
# plot drawn, set:
#
# res2.nglDraw  = False
# res2.nglFrame = False
#
plot2 = Ngl.xy(wks,lon,u,res2)

#
# Now, draw the two plots and advance the frame. Since we went through
# all the trouble to make sure the plots were in the same location, 
# and that the appropriate tickmarks were turned off in certain areas,
# these two plots should look like one plot, but with two separate
# curves and two separate Y axes.
#
# Also, change the main title of the first plot to reflect that
# we have two vars here, and remove the main title from the second plot.
#
sres              = Ngl.Resources()
sres.tiMainString = "Two variables with individual scales"
Ngl.set_values(plot1,sres)

sres          = Ngl.Resources()
sres.tiMainOn =  False
Ngl.set_values(plot2,sres)

Ngl.draw(plot1)
Ngl.draw(plot2)
Ngl.frame(wks)

# 
# Just to show it can be done, add the second plot as an annotation
# of the first plot.
#
sres                = Ngl.Resources()
sres.amZone         = 0     # '0' means centered over base plot.
sres.amResizeNotify = True
anno = Ngl.add_annotation(plot1,plot2,sres)

#
# Change the main title to reflect this is an added annotation.
#
sres              = Ngl.Resources()
sres.tiMainString = "One plot added as annotation of another"
Ngl.set_values(plot1,sres)

#
# Now if you draw plot1, and plot2 will be drawn automatically.  Also,
# if you resize plot1, the attached plot2 will resize automatically.
#
Ngl.draw(plot1)
Ngl.frame(wks)

#
# Show how Ngl.remove_annotation works.
#
Ngl.remove_annotation(plot1,anno)

#
# Change the main title to reflect that an annotation has been removed.
#
sres              = Ngl.Resources()
sres.tiMainString = "Annotation removed, back to original first plot"
Ngl.set_values(plot1,sres)


Ngl.draw(plot1)
Ngl.frame(wks)

Ngl.end()
