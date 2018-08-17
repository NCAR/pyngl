#
#  File:
#    mask1.py
#
#  Synopsis:
#    Illustrates masking certain areas in a contour/map plot by drawing the 
#    same plot twice with different options.
#
#  Categories:
#    contour plots
#    maps
#
#  Author:
#    Mary Haley, based on NCL example
#  
#  Date of initial publication:
#    September 2017
#
#  Description:
#    This example shows how draw precipitation over land filled in tan,
#    while masking out contours over the ocean. The filled contours
#    are first drawn over a map with tan land, and the same plot
#    is drawn again, but with transparent contours and white ocean,
#    effectively masking the previous plot.
#
#    See mask2.py for another method you can use for masking. It
#    uses the same data file.
#
#
#  Effects illustrated:
#    o  Reading group data off an HDF5 file
#    o  Zooming in on South America
#    o  Using transparency to allow graphical features to show through
#    o  Masking ocean
#    o  Increasing the thickness of map outlines
#    o  Drawing a map using the medium resolution map outlines
#    o  Transposing data
# 
#  Output:
#     One visualization is produced.
#
import Nio,Ngl
import sys, os
import numpy as np

# Test if file exists
filename = "3B-MO.MS.MRG.3IMERG.20140701-S000000-E235959.07.V03D.HDF5"
if(not os.path.exists(filename)):
  print("You do not have the necessary '{}' HDF5 file to run this example.".format(filename))
  print("You need to supply your own HDF5 data or download the file from http://www.ncl.ucar.edu/Applications/Data/")
  sys.exit()

# Be sure to read this file using the advanced file structure.
opt               = Nio.options()
opt.FileStructure = 'advanced'
f = Nio.open_file(filename, "r", options=opt)

# Open group "Grid" which will now look like a regular NioFile
g = f.groups['Grid']

# Read data from this group
precip = g.variables['precipitation']
lat    = g.variables['lat'][:]
lon    = g.variables['lon'][:]

# Print the metadata of precip, and min/max values
print(precip)
print("min/max = {:g} / {:g}".format(precip[:].min(), precip[:].max()))

wks_type = "png"
wks = Ngl.open_wks(wks_type,"mask1")

# Set up resource list for plot options.
res                             = Ngl.Resources()

res.nglFrame                    = False        # Will advance the frame later

res.cnLinesOn                   = False        # Turn off contour lines
res.cnLineLabelsOn              = False        # Turn off contour labels
res.cnFillOn                    = True         # Turn on contour fill
res.cnFillMode                  = "RasterFill" # "AreaFill" is the default and can be slow for large grids.

# Define the contour leves and the colors for each.
res.cnLevelSelectionMode        = "ExplicitLevels"
res.cnLevels                    = [ 0.01, 0.02, 0.04, 0.08, 0.16, \
                                     0.32, 0.64, 0.96]
res.cnFillColors                = ["transparent","cyan", "green","yellow",\
                                    "darkorange","red","magenta","purple",\
                                    "black"]

res.lbOrientation               = "Horizontal"
res.pmLabelBarHeightF           = 0.08      # make labelbar thinner
res.pmLabelBarWidthF            = 0.50      # make labelbar wider
res.lbLabelFontHeightF          = 0.012     # make labels smaller

# Turn on map fill
res.mpFillOn                    = True
res.mpLandFillColor             = "tan"

# Set resources for better map outlines
res.mpOutlineOn                 = True
res.mpOutlineBoundarySets       = "AllBoundaries"
res.mpDataBaseVersion           = "MediumRes"
res.mpDataSetName               = "Earth..4"       # gives us some provincial boundaries

# Increase thickness of map outlines
res.mpPerimOn                   = True
res.mpPerimLineThicknessF       = 2.0
res.mpNationalLineThicknessF    = 4
res.mpProvincialLineThicknessF  = 4
res.mpGeophysicalLineThicknessF = 4

# Zoom in on South America
res.mpLimitMode                 = "LatLon"
res.mpMinLatF                   = -60
res.mpMaxLatF                   = 20
res.mpMinLonF                   = -90
res.mpMaxLonF                   = -30

res.sfMissingValueV             = precip._FillValue
res.sfXArray                    = lon
res.sfYArray                    = lat

# 
# In order to display filled contours over land and have the ocean
# white, but keep the gray filled land underneath, we have to draw
# the plot twice.  
#
# First, draw the filled contours with the gray land (the default).
# Second, we simply want to draw the ocean part in white, so we need
# to effectively turn off the land fill and the contour fill by
# setting them both to transparent.
#

#
# Draw the first plot, but don't advance frame.
#
# Note that the data is ordered lon x lat; you must reorder before
# plotting using transpose. 
#
plot = Ngl.contour_map(wks, precip[:].transpose(), res)

# Change some map resources and apply to existing contour/map plot.
res2 = Ngl.Resources()
res2.mpLandFillColor             = "transparent"  # Make sure land doesn't get filled again
res2.mpOceanFillColor            = "white"        # Fill water areas in white.
res2.mpInlandWaterFillColor      = "white"
Ngl.set_values(plot,res2)

# Change one resource to the contours in the existing plot, making them transparent.
res3 = Ngl.Resources()
res3.cnFillOpacityF = 0.0
Ngl.set_values(plot.contour,res3)

# Draw the plot again with the new settings, and advance the frame.
Ngl.draw(plot)
Ngl.frame(wks)
Ngl.end()


