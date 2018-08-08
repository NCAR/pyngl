#
#  File:
#    wrf2.py
#
#  Synopsis:
#    Draws contours over a map of a "HGT" variable read off a WRF output file.
#
#  Categories:
#    Contouring
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    April 2015
#
#  Description:
#    This example shows how to contour data read off a WRF output file
#    using the native map projection information on the WRF file.
#    Shapefile outlines are added to the second plot for comparison.
#
#  Effects illustrated:
#    o  Plotting WRF data
#    o  Plotting curvilinear data
#    o  Using cnFillPalette to assign a color palette to contours
#    o  Explicitly defining contour levels.
#    o  Using gsSegments to specify shapefile segments
# 
#  Output:
#    This example produces two filled contour plots
#     
#  Notes:
#     You will need to include your own WRF output file in place
#     of the one referenced by this example.
#
#     You must download the desired "XXX_adm" shapefiles from:
#      http://gadm.org/country/
#======================================================================
from __future__ import print_function
import numpy, Nio, Ngl, os, sys

#======================================================================
# This function attaches outlines from the given shapefile to the
# given NCL plot.
#======================================================================
def add_shapefile_outlines(wks,plot,filename):
#---Read data off shapefile
  f        = Nio.open_file(filename, "r")
  lon      = numpy.ravel(f.variables["x"][:])
  lat      = numpy.ravel(f.variables["y"][:])

  plres                  = Ngl.Resources()      # resources for polylines
  plres.gsLineColor      = "navyblue"
  plres.gsLineThicknessF = 2.0                  # default is 1.0
  plres.gsSegments       = f.variables["segments"][:,0]

  return Ngl.add_polyline(wks, plot, lon, lat, plres)

#----------------------------------------------------------------------
# Main code
#----------------------------------------------------------------------
wrf_filename = "wrfout_d01_2005-12-14_13:00:00"
shp_filename = "USA_adm/USA_adm1.shp"
if(not os.path.exists(wrf_filename)):
  print("You do not have the necessary '{}' file to run this example.".format(wrf_filename))
  print("You need to supply your own WRF output file")
  print("WRF output files usually have names like '{}'".format(filename))
  sys.exit()

if(not os.path.exists(shp_filename)):
  print("You do not have the '{}' shapefile.".format(shp_filename))
  print("See the comments at the top of this script for more information.")
  print("No shapefile outlines will be added.")
  ADD_SHAPE_OUTLINES = False
else:
  ADD_SHAPE_OUTLINES = True

#---Read data
a   = Nio.open_file("{}.nc".format(wrf_filename))
hgt = a.variables["HGT"][0,:,:]    # Read first time step ( nlat x nlon)

#---Send graphics to PNG file
wks_type = "png"
wks = Ngl.open_wks(wks_type,"wrf2")

#---Set some plot options
res                   = Ngl.Resources()

# Contour options
res.cnFillOn          = True          # turn on contour fill
res.cnLinesOn         = False         # turn off contour lines
res.cnLineLabelsOn    = False         # turn off line labels
res.cnFillPalette     = "OceanLakeLandSnow"

# You may need to change these levels depending on your WRF output file
# Try first running this script with these two lines commented out.
res.cnLevelSelectionMode = "ExplicitLevels"
res.cnLevels  = [0.5,10,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1100]

# Set some map options based on information in WRF output file
res = Ngl.wrf_map_resources("{}.nc".format(wrf_filename), res)

# Labelbar options
res.lbOrientation      = "horizontal"
res.lbTitleString      = "Height (m)"
res.lbTitleFontHeightF = 0.008

# Main Title
res.tiMainString      = "WRF data plotted in native projection"
plot = Ngl.contour_map(wks,hgt,res)

if(ADD_SHAPE_OUTLINES):
# Recreate plot without map outlines so we can add some from shapefile.
  res.nglDraw      = False
  res.nglFrame     = False
  res.mpOutlineOn  = False
  res.tiMainString = "WRF plot with shapefile outlines added"

  plot = Ngl.contour_map(wks,hgt,res)

  lnid = add_shapefile_outlines(wks,plot,"USA_adm/USA_adm1.shp")

  Ngl.draw(plot)       # This draws map and attached polylines
  Ngl.frame(wks)       # Advance frame.

Ngl.end()

