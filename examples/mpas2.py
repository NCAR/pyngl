#
#  File:
#    mpas2.py
#
#  Synopsis:
#    Compares raster and cell filled contours on an MPAS grid with 2,621,442 cells.
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
#    This example shows how to draw raster and celled filled contours of 
#    a 2M cell array on an MPAS grid.
#
#    NOTE: The second plot causes a large output file to be created,
#    which may take a while to view.
#
#  Effects illustrated:
#    o  Plotting MPAS data
#    o  Plotting unstructured data
# 
#  Output:
#    This example produces one or two contour plots.
#     
#  Notes:
#     The MPAS data file is too large to be included with this 
#     software. Either use your own MPAS grid, or send email to
#     pyngl-talk@ucar.edu.
#
#     http://mailman.ucar.edu/mailman/listinfo/pyngl-talk
#
#======================================================================
import os, numpy, math, time, sys
import Ngl, Nio

#----------------------------------------------------------------------
# Function to add three subtitles to the top of a plot, left-justified,
# centered, and right-justified.
#----------------------------------------------------------------------
def subtitles(wks, plot, left_string, center_string, right_string):
  ttres         = Ngl.Resources()
  ttres.nglDraw = False          # Make sure string is just created, not drawn.
#
# Retrieve font height of left axis string and use this to calculate
# size of subtitles.
#
  font_height = Ngl.get_float(plot.base,"tiXAxisFontHeightF")
  ttres.txFontHeightF = font_height*0.8    # Slightly smaller

#
# Set some some annotation resources to describe how close text
# is to be attached to plot.
#
  amres = Ngl.Resources()
  amres.amOrthogonalPosF = -0.51   # Top of plot plus a little extra
                                   # to stay off the border.
#
# Create three strings to put at the top, using a slightly
# smaller font height than the axis titles.
#
  if left_string != "":
    txidl = Ngl.text(wks, plot, left_string, 0., 0., ttres)

    amres.amJust         = "BottomLeft"
    amres.amParallelPosF = -0.5   # Left-justified
    annoidl              = Ngl.add_annotation(plot, txidl, amres)

  if center_string != "":
    txidc = Ngl.text(wks, plot, center_string, 0., 0., ttres)

    amres.amJust         = "BottomCenter"
    amres.amParallelPosF = 0.0   # Centered
    annoidc              = Ngl.add_annotation(plot, txidc, amres)

  if right_string != "":
    txidr = Ngl.text(wks, plot, right_string, 0., 0., ttres)

    amres.amJust         = "BottomRight"
    amres.amParallelPosF = 0.5   # Right-justifed
    annoidr              = Ngl.add_annotation(plot, txidr, amres)

  return

#----------------------------------------------------------------------
# Function to print the min/max of a given variable.
#----------------------------------------------------------------------
def print_min_max(var,varname):
  print(varname + ": min/max = %.2f " % numpy.min(var) + "/" + "%.2f" % numpy.max(var))

#----------------------------------------------------------------------
# Main code
#----------------------------------------------------------------------
mpas_file = "x1.2621442.partial.nc"
if(not os.path.exists(mpas_file)):
  print "You do not have the necessary file to run this example."
  print "See the comments at the top of this script for more information."
  sys.exit()

#---Read a timestep of "t2m" 
varname  = "t2m"
ntime    = 3                                        # ntime=0 is a constant field for t2m
f        = Nio.open_file(mpas_file)
t2m      = f.variables[varname][ntime,:]
time_str = ''.join(f.variables["xtime"][ntime][:]).strip()

lonCell        = f.variables["lonCell"][:]
latCell        = f.variables["latCell"][:]
lonVertex      = f.variables["lonVertex"][:]
latVertex      = f.variables["latVertex"][:]
verticesOnCell = f.variables["verticesOnCell"][:]
nCells         = verticesOnCell.shape[0]
maxEdges       = verticesOnCell.shape[1]

#---Convert to degrees from radians
RAD2DEG   = 180.0/(math.atan(1)*4.0)
latCell   = latCell * RAD2DEG
lonCell   = lonCell * RAD2DEG
latVertex = latVertex * RAD2DEG
lonVertex = lonVertex * RAD2DEG

#
# In order to do a CellFill plot, you need to provide
# boundaries of each cell.  
#
latvoc = numpy.ndarray((nCells, maxEdges),'d')
lonvoc = numpy.ndarray((nCells, maxEdges),'d')

for i in range(maxEdges):
  latvoc[:,i] = latVertex[verticesOnCell[:,i] - 1]
  lonvoc[:,i] = lonVertex[verticesOnCell[:,i] - 1]

#---Clean up; don't need these anymore
del latVertex,lonVertex,verticesOnCell

#---Debug prints
print("This MPAS file has " + str(nCells) + " cells.")

print_min_max(t2m,"t2m")
print_min_max(latCell,"latCell")
print_min_max(lonCell,"lonCell")
print_min_max(latvoc,"latvoc")
print_min_max(lonvoc,"lonvoc")

#---Start the graphics
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"mpas2")

res                      = Ngl.Resources()              # Plot mods desired.

res.nglDraw              = False             # Turn off plot draw and frame advance. We will
res.nglFrame             = False             # do it later after adding subtitles.

res.cnFillOn             = True              # color plot desired
res.cnFillPalette        = "ncl_default"
res.cnLinesOn            = False             # turn off contour lines
res.cnLineLabelsOn       = False             # turn off contour labels

res.cnLevelSelectionMode = "ExplicitLevels"
res.cnLevels             = Ngl.fspan(min(t2m),max(t2m),253)   # 253 levels (hence 254 colors)

res.lbOrientation        = "Horizontal"      # vertical by default
res.lbBoxLinesOn         = False             # turn off labelbar boxes
res.lbLabelFontHeightF   = 0.01

res.mpFillOn             = False
res.mpGridAndLimbOn      = False

res.sfXArray             = lonCell      # where to overlay contours
res.sfYArray             = latCell

#
# It may be necessary to use RasterFill or CellFill for large grids.
#
# The default "AreaFill" can be too slow and/or will run out of 
# memory easily.
#

#----------------------------------------------------------------------
# Raster fill plot
#----------------------------------------------------------------------
res.cnFillMode           = "RasterFill"
res.tiMainString         = "MPAS grid - raster fill"
res.tiMainOffsetYF       = 0.03      # Move main title towards plot

t1 = time.time()
plot = Ngl.contour_map(wks,t2m,res)  

#---Attach some subtitles to the plot
left_string   = varname
center_string = time_str
right_string  = str(nCells) + " cells"
subtitles(wks,plot,left_string,center_string,right_string)

#---Drawing the plot will also draw the subtitles
Ngl.draw(plot)
Ngl.frame(wks)

t2 = time.time()
print "RasterFill took %0.3f seconds" % (t2-t1)

# Clean up, otherwise next plot gets drawn on top of this one (?)
del plot

#----------------------------------------------------------------------
# Cell fill plot
#
# NOTE: This plot causes a large output file to be created, which may 
# take a while to view. This is turned off for now.
#
# The cell fill plot should look a little better than the raster fill
# because it will fill the corners of the map plot. Raster fill leaves
# little gaps in these areas.
#----------------------------------------------------------------------
CELL_FILL_PLOT = False
if CELL_FILL_PLOT:
  res.cnFillMode           = "CellFill"
  res.sfXCellBounds        = lonvoc       # necessary for CellFill 
  res.sfYCellBounds        = latvoc
  res.tiMainString         = "MPAS grid - cell fill"

  t1 = time.time()
  plot = Ngl.contour_map(wks,t2m,res)  

  subtitles(wks,plot,left_string,center_string,right_string)

#---Drawing the plot will also draw the subtitles
  Ngl.draw(plot)
  Ngl.frame(wks)

  t2 = time.time()
  print "CellFill took %0.3f seconds" % (t2-t1)

Ngl.end()

