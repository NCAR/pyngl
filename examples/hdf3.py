#
#  File:
#    hdf3.py
#
#  Synopsis:
#    Unpacks and plots HDF5 data
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
#    This example reads precipitation from a group called "Grid"
#    off an HDF5 file, and creates a color contour plot.  The
#    data comes in as lon x lat, so you have to transpose it
#    before plotting.
#
#  Effects illustrated:
#    o  Reading group data from an HDF5 file.
#    o  Drawing colored contours using named colors
#    o  Contouring in triangular mesh mode.
#    o  Transposing data
#
#  Output:
#    A single visualization is produced showing precipitation
#    over a map.

import Nio,Ngl
import sys, os
import numpy as np

# Test if file exists
filename = "3B-MO.MS.MRG.3IMERG.20140701-S000000-E235959.07.V03D.HDF5"
if(not os.path.exists(filename)):
  print("You do not have the necessary %s HDF5 file to run this example." % filename)
  print("You need to supply your own HDF5 data")
  sys.exit()

# Be sure to read this file using the advanced file structure.
opt               = Nio.options()
opt.FileStructure = 'advanced'
f = Nio.open_file(filename, "r", options=opt)

# Open group "Grid" which will now look like a regular NioFile
g = f.groups['Grid']
#print(g)

# Read data from this group
precip = g.variables['precipitation']
lat    = g.variables['lat'][:]
lon    = g.variables['lon'][:]
yyyymmdd = filename.split(".")[4].split('-')[0]

# Print the metadata of precip, and min/max values
print(precip)
print("min/max = %g / %g" % (precip[:].min(), precip[:].max()))

wks_type = "png"
wks = Ngl.open_wks(wks_type,"hdf3")

res                      = Ngl.Resources()
res.cnFillOn             = True               # turn on contour fill
res.cnLinesOn            = False              # turn off contour lines
res.cnLineLabelsOn       = False              # turn off line labels

res.cnFillMode           = "RasterFill"       # These two resources
res.trGridType           = "TriangularMesh"   # can speed up plotting.

res.cnLevelSelectionMode = "ExplicitLevels"
res.cnLevels             = [0.01,0.02,0.04,0.08,0.16,0.32,0.64, 0.96]
res.cnFillColors         = ["white","cyan", "green","yellow",
                            "darkorange","red","magenta","purple","black"]

# make Dateline the center of the plot (default is GM)
res.mpCenterLonF         = 180  

res.tiMainString         = "precipitation (%s) (%s)" % (precip.units,yyyymmdd)
res.lbLabelFontHeightF   = 0.01              # default is a bit large
res.lbOrientation        = "horizontal"

res.sfMissingValueV      = precip._FillValue
res.sfXArray             = lon
res.sfYArray             = lat

# Be sure to transpose data before plotting
plot = Ngl.contour_map(wks,precip[:].transpose(),res)

Ngl.end()




