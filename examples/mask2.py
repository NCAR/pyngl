#
#  File:
#    mask2.py
#
#  Synopsis:
#    Illustrates masking certain areas in a contour/map plot by using annotations.
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
#    are first drawn over a map with tan land, and then a map plot
#    with white ocean is added to this using annotations. By using
#    annotations, one plot becomes part of the other, and hence you
#    can treat the as one object, say, in a panel plot.
#
#    See mask1.py for another method you can use for masking. It
#    uses the same data file.
#
#  Effects illustrated:
#    o  Reading group data off an HDF5 file
#    o  Zooming in on Europe
#    o  Adding a map plot as an annotation of a contour/map plot
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

# Set up map resources
def set_map_resources():
  res                             = Ngl.Resources()
  res.nglDraw                     = False
  res.nglFrame                    = False

# Turn on map fill
  res.mpFillOn                    = True
  res.mpLandFillColor             = "tan"

# Better map outlines
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

# Zoom in on part of Europe
  res.mpLimitMode                 = "LatLon"
  res.mpMinLatF                   =  30
  res.mpMaxLatF                   =  80
  res.mpMinLonF                   = -12
  res.mpMaxLonF                   =  35

  return(res)

# -------------------------------------------------------------------
# Main code
# -------------------------------------------------------------------

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

# Select an area that covers South America
minlat = -60
maxlat =  20
minlon = -90
maxlon =  -30

wks_type = "png"
wks = Ngl.open_wks(wks_type,"mask2")

# -------------------------------------------------------------------
# Code for creating contour map plot.
# -------------------------------------------------------------------
cnres                           = set_map_resources()
cnres.cnLinesOn                   = False        # Turn off contour lines
cnres.cnLineLabelsOn              = False        # Turn off contour labels
cnres.cnFillOn                    = True         # Turn on contour fill
cnres.cnFillMode                  = "RasterFill" # "AreaFill" is the default and can be slow for large grids.

# Define the contour leves and the colors for each.
cnres.cnLevelSelectionMode        = "ExplicitLevels"
cnres.cnLevels                    = [ 0.01, 0.02, 0.04, 0.08, 0.16, \
                                       0.32, 0.64, 0.96]
cnres.cnFillColors                = ["transparent","cyan", "green","yellow",\
                                      "darkorange","red","magenta","purple",\
                                      "black"]

cnres.lbOrientation               = "Horizontal"
cnres.pmLabelBarHeightF           = 0.08      # make labelbar thinner
cnres.pmLabelBarWidthF            = 0.50      # make labelbar wider
cnres.lbLabelFontHeightF          = 0.012     # make labels smaller

cnres.sfMissingValueV             = precip._FillValue
cnres.sfXArray                    = lon
cnres.sfYArray                    = lat

#
# The data is ordered lon x lat; you must reorder before plotting
# using transpose. 
#
contour_map_plot = Ngl.contour_map(wks, precip[:].transpose(),cnres)

# -------------------------------------------------------------------
# Code for creating map only plot.
# -------------------------------------------------------------------

# Get size of contour/map plot
# Retrieve size of contour/map plot
bres     = Ngl.Resources()
bres.vpx = Ngl.get_float(contour_map_plot,"vpXF")
bres.vpy = Ngl.get_float(contour_map_plot,"vpYF")
bres.vph = Ngl.get_float(contour_map_plot,"vpHeightF")
bres.vpw = Ngl.get_float(contour_map_plot,"vpWidthF" )

mpres = set_map_resources()

# Make sure map plot is same size as contour/map plot.
mpres.nglMaximize      = False      # this must be turned off otherwise plot will be resized!
mpres.vpXF             = bres.vpx       
mpres.vpYF             = bres.vpy       
mpres.vpWidthF         = bres.vpw
mpres.vpHeightF        = bres.vph

# Turn off since they are already drawn in contour/map plot.
mpres.pmTickMarkDisplayMode = "Never"   

# Make the ocean white and land transparent. This will mask out the contour fill over ocean.
mpres.mpOceanFillColor       = "white"
mpres.mpInlandWaterFillColor = "white"
mpres.mpLandFillColor        = "transparent"

# Create a map plot with the white ocean and transparent land.
map_plot = Ngl.map(wks, mpres)

#
# If you need to resize the plots later---for example, to use in
# a panel---then it's easier to make one plot an annotation of
# the other.
#
annoid = Ngl.add_annotation(contour_map_plot, map_plot)
Ngl.draw(contour_map_plot)   # This draws both plots
Ngl.frame(wks)

Ngl.end()

