#
#  File:
#    mpas1.py
#
#  Synopsis:
#    Draws contours on an MPAS grid with 163,842 cells.
#
#  Categories:
#    Contouring
#
#  Author:
#    Mary Haley, based on an NCL example by Dave Brown
#  
#  Date of initial publication:
#    September 2014
#
#  Description:
#    This example shows how to draw filled contours of a variable on an MPAS grid
#
#  Effects illustrated:
#    o  Plotting MPAS data
#    o  Plotting unstructured data
#    o  Plotting data with missing lat/lon coordinates
#    o  Using cnFillPalette to assign a color palette to contours
#    o  Reading an existing color map and subsetting it.
# 
#  Output:
#    This example produces a filled contour plot
#     
#  Notes:
#     The MPAS data file is too large to be included with this 
#     software. Either use your own MPAS grid, or send email to
#     pyngl-talk@ucar.edu.
#
#     http://mailman.ucar.edu/mailman/listinfo/pyngl-talk
#
#======================================================================
import os, numpy, math, sys
import Ngl, Nio

#---Read data from MPAS Grid
filename = "MPAS.nc"
if(not os.path.exists(filename)):
  print "You do not have the necessary file to run this example."
  print "See the comments at the top of this script for more information."
  sys.exit()

f  = Nio.open_file(filename)
sp = f.variables["surface_pressure"][0,:]
sp = sp/1000.   # Not sure what the pressure units are, there's
                # not much metadata info on this file

lonCell = f.variables["lonCell"][:]
latCell = f.variables["latCell"][:]

#---Convert to degrees from radians
RAD2DEG   = 180.0/(math.atan(1)*4.0)  # Radian to Degree
lonCell   = lonCell * RAD2DEG
latCell   = latCell * RAD2DEG

#---Start the graphics
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"mpas1")

#---Read in desired color map so we can subset it later
cmap = Ngl.read_colormap_file("WhiteBlueGreenYellowRed")

res                      = Ngl.Resources()              # Plot mods desired.

res.cnFillOn             = True              # color plot desired
res.cnFillPalette        = cmap[48:208,:]    # Don't use white
res.cnLinesOn            = False             # turn off contour lines
res.cnLineLabelsOn       = False             # turn off contour labels
res.lbOrientation        = "Horizontal"      # vertical by default

res.trGridType           = "TriangularMesh"  # This is required to allow
                                             # missing coordinates.
res.cnLevelSelectionMode = "ManualLevels"
res.cnMinLevelValF       = 55
res.cnMaxLevelValF       = 100
res.cnLevelSpacingF      = 2.5
res.mpFillOn             = False
res.mpGridAndLimbOn      = False

res.sfXArray             = lonCell
res.sfYArray             = latCell

res.cnFillMode           = "RasterFill"      # turn raster on      
res.tiMainString         = "Surface pressure on MPAS grid (" + \
                           str(sp.shape[0]) + " cells)"
res.tiMainFontHeightF   = 0.018

plot = Ngl.contour_map(wks,sp,res)  

Ngl.end()

