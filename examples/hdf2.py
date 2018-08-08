#
#  File:
#    hdf2.py
#
#  Synopsis:
#    Unpacks and plots HDF-EOS data
#
#  Category:
#    Contours over maps
#    Labelbar
#    Maps
#
#  Author:
#    Mary Haley (based on NCL example from Dennis Shea)
#  
#  Date of initial publication:
#    April, 2015
#
#  Description:
#    This example uses special variable attributes scale_factor and
#    add_offset to unpack short data read off an HDF-EOS file.
#
#  Effects illustrated:
#    o  Reading from an HDF file.
#    o  Unpacking data read off a file.
#    o  Contouring in triangular mesh mode.
#
#  Output:
#    A single visualization is produced showing cloud top
#    temperature over a map


# Import support modules
from __future__ import print_function
import os,sys
import numpy as np
import Nio, Ngl

#---Test if file exists
filename = "MOD06_L2.A2010031.1430.005.2010031221343.hdf" 
if(not os.path.exists(filename)):
  print("You do not have the necessary {} file to run this example.".format(filename))
  print("You can get the file from")
  print("    http://www.ncl.ucar.edu/Applications/Data/#hdf")
  print("or else use your own data.")
  sys.exit()

vname    = "Cloud_Top_Temperature"
a        = Nio.open_file(filename)
ctts     = a.variables[vname]
cttf     = ctts.scale_factor * (ctts[:] - ctts.add_offset)

# Read lat/lon off the file for plotting
lat      = a.variables["Latitude"][:]
lon      = a.variables["Longitude"][:]

wks_type = "png"
wks = Ngl.open_wks(wks_type,"hdf2")

# Set some plot options
res                   = Ngl.Resources()

# Contour options
res.cnFillOn          = True          # turn on contour fill
res.cnLinesOn         = False         # turn off contour lines
res.cnLineLabelsOn    = False         # turn off line labels
res.cnFillMode        = "RasterFill"        # These two resources
res.cnLevelSpacingF   = 2.5              # contour level spacing
res.trGridType        = "TriangularMesh"    # can speed up plotting.
res.cnFillPalette     = "WhViBlGrYeOrRe"

# Map options
res.mpDataBaseVersion = "MediumRes"
res.mpLimitMode       = "LatLon"
res.mpMinLatF         = np.min(lat)-1
res.mpMaxLatF         = np.max(lat)+1
res.mpMinLonF         = np.min(lon)-1
res.mpMaxLonF         = np.max(lon)+1
res.mpGridAndLimbOn   = False

# Main Title
res.tiMainString     = "{} ({})".format(ctts.hdf_name, ctts.units)

# Labelbar options
res.lbLabelFontHeightF = 0.01

# Additional resources needed for putting contours on map
res.sfXArray          = lon
res.sfYArray          = lat
res.sfMissingValueV   = float(ctts._FillValue)

plot = Ngl.contour_map(wks,cttf,res)

Ngl.end()



