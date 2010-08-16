#
#   File:
#     cn05p.py
#
#   Synopsis:
#     Draws an animation of global surface temperature over a map.
#
#   Category:
#     Contours over maps
#
#   Author:
#     Mary Haley (based on example of Tim Scheitlin)
#
#   Date of initial publication:    
#     November, 2005
#  
#   Description:
#     This example draws an animation of filled contours over a map 
#     showing surface temperatures (if "Animate" is set to True).
#     Instead of calling Ngl.contour_map for every time step,
#     Ngl.set_values is used to change the data and title after the
#     initial time step.
#
#  Effects illustrated
#     o  Reading data from a netCDF file.
#     o  Creating a color map using RGB triplets.
#     o  Drawing color filled contours over a map.
#     o  Using Ngl.set_values to change the data for the animation.
#     o  Using a resource list to set many resources, for example to:
#          + set color map
#          + set contour levels
#          + set fill colors
#          + turn off contour lines and line labels
#          + set some labelbar, title,  and tickmark resources
#
#  Output:
#     If "Animate" is set to True, then an animation of 31 frames
#     (one per day on January) is produced. Otherwise, just one frame
#     is produced.
#
#  Notes: 

#
#  Import numpy, Ngl, and Nio support functions.
#
import numpy 
import Ngl 
import Nio
import os

#
# Read some variables off the file.
#
dirc = Ngl.pynglpath("data")
nf   = Nio.open_file(os.path.join(dirc,"cdf","meccatemp.cdf"))
T    = nf.variables["t"][:,:,:]
lat  = nf.variables["lat"][:]
lon  = nf.variables["lon"][:]

#
# Set up a color map using RGB triplets.
#
cmap =  numpy.array(\
        [ [.000,.000,.000], [1.00,1.00,1.00], [.700,.700,.700], \
          [.650,.650,.700], [.610,.600,.700], [.550,.550,.700], \
          [.560,.500,.700], [.450,.450,.700], [.420,.400,.700], \
          [.350,.350,.700], [.300,.300,.700], [.250,.250,.700], \
          [.200,.200,.700], [.150,.150,.700], [.100,.100,.700], \
          [.050,.050,.700], [.000,.000,.700], [.000,.050,.700], \
          [.000,.100,.700], [.000,.150,.700], [.000,.200,.700], \
          [.000,.250,.700], [.000,.300,.700], [.000,.350,.700], \
          [.000,.400,.700], [.000,.450,.600], [.000,.500,.500], \
          [.000,.550,.400], [.000,.600,.300], [.000,.650,.200], \
          [.000,.700,.100], [.000,.725,.000], [.000,.690,.000], \
          [.030,.685,.000], [.060,.680,.000], [.100,.575,.000], \
          [.130,.570,.000], [.160,.565,.000], [.550,.550,.000], \
          [.555,.545,.000], [.560,.530,.000], [.565,.485,.000], \
          [.570,.420,.000], [.675,.375,.000], [.680,.330,.000], \
          [.690,.300,.000], [.700,.285,.000], [.700,.270,.000], \
          [.700,.260,.000], [.700,.240,.000], [.700,.180,.000], \
          [.700,.130,.000], [.700,.120,.000], [.700,.100,.000], \
          [.700,.090,.000], [.750,.090,.000], [.800,.090,.000], \
          [.830,.070,.000], [.870,.050,.000], [.900,.030,.000], \
          [.950,.010,.000], [.990,.000,.000], [1.00,.000,.000], \
          [1.00,.000,.000] ])

#
# Open a PS file and change the color map.
#
wres = Ngl.Resources()
wks_type = "ps"
wres.wkColorMap = cmap
wks = Ngl.open_wks(wks_type,"cn05p",wres)

#
# Set up variable to hold the various plot
# resources we want to set.
#
res              = Ngl.Resources()

#
# Set some scalar field resources.
#
res.sfXArray = lon
res.sfYArray = lat

#
# Set some contour resources
#
res.cnFillOn             = True
res.cnFillColors         = range(2,62)
res.cnLinesOn            = False
res.cnLineLabelsOn       = False

res.cnLevelSelectionMode = "ManualLevels"
res.cnMinLevelValF       = 195.0
res.cnMaxLevelValF       = 328.0
res.cnLevelSpacingF      = 2.25

#
# Set some labelbar resources.
#
res.lbBoxLinesOn  = False
res.lbLabelStride = 6
res.lbOrientation = "Horizontal"

#
# Set a map resource.
#
res.mpGridAndLimbOn        = False
res.mpGeophysicalLineColor = "Black"

#
# Set some title resources.
#
res.tiMainString      = "January Global Surface Temperature (K)"
res.tiMainFontHeightF = 0.02

#
# Create a couple of variables to hold resources that will change
# inside the for loop below.
#
srlist1 = Ngl.Resources()
srlist2 = Ngl.Resources()

#
# Generate at least the first frame. Set "Animate" to True
# if you want to generate the other 30 frames.
#
print "plotting day 1"
res.lbTitleString = "Day 1"
map = Ngl.contour_map(wks,T[0,:,:],res)

Animate = False

if Animate:
#
# Loop through time (the rightmost dimension of "T") and generate
# a contour/map plot at each time step.
#
  for nt in range(1,T.shape[0]):
    print "plotting day ",nt+1
#
# There's no need to recreate the contour/map plot since
# the only thing changing is the data and the title, so
# use Ngl.set_values to change the necessary two resources.
# Using Ngl.set_values will result in slightly faster code.
#
    srlist2.sfDataArray   = T[nt,:,:]  
    srlist1.lbTitleString = "Day " + str(nt+1)

    Ngl.set_values(map.contour,srlist1)      # changing labelbar string
    Ngl.set_values(map.sffield,srlist2)      # changing data

#
# Draw new contour/map plot and advance the frame.
#
    Ngl.draw(map.base)
    Ngl.frame(wks)
else:
  print "Only one frame was generated."
  print "Set 'Animate' to True if you want to generate all 31 frames."

Ngl.end()
