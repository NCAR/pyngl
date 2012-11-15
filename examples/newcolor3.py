#
#  File:
#    newcolor3.py
#
#  Synopsis:
#    Illustrates new color capabilities in PyNGL 1.5.0
#
#  Categories:
#    Contouring
#
#  Author:
#    Mary Haley, based on NCL example by Rick Brownrigg
#  
#  Date of initial publication:
#    November 2012
#
#  Description:
#    This example shows how to draw partially opaque filled vectors
#
#  Effects illustrated:
#    o  Using the new "vcGlyphOpacityF" resource.
#    o  Reading an existing color map and subsetting it.
# 
#  Output:
#    This example produces two visualizations: 
#      1.  Fully opaque vectors
#      2.  Partially opaque vectors
#     
import os, numpy
import Ngl, Nio

#
# Create some dummy data for the contour plot.
#
dirc = Ngl.pynglpath("data")
f    = Nio.open_file(os.path.join(dirc,"cdf","uv300.nc"))
u    = f.variables["U"][1,:,:]
v    = f.variables["V"][1,:,:]
lat  = f.variables["lat"][:]
lon  = f.variables["lon"][:]
spd  = numpy.sqrt(u**2 + v**2)
 
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"newcolor3")

cnres                             = Ngl.Resources()
cnres.nglDraw                     = False
cnres.nglFrame                    = False

cmap = Ngl.read_colormap_file("WhiteBlueGreenYellowRed")

cnres.cnLinesOn                   = False
cnres.cnLineLabelsOn              = False
cnres.cnFillOn                    = True
cnres.cnFillPalette               = cmap[15:,:]
cnres.lbOrientation               = "horizontal"

cnres.mpFillOn                    = False
cnres.mpGeophysicalLineColor      = "Grey18"
cnres.mpGeophysicalLineThicknessF = 1.5

cnres.sfXArray                    = lon
cnres.sfYArray                    = lat

vcres                         = Ngl.Resources()
vcres.nglDraw                 = False
vcres.nglFrame                = False

vcres.vfXArray                = lon
vcres.vfYArray                = lat

vcres.vcFillArrowsOn          = True
vcres.vcRefMagnitudeF         = 30.0             # define vector ref mag
vcres.vcRefLengthF            = 0.02             # define length of vec ref
vcres.vcMinFracLengthF        = 0.3
vcres.vcMinDistanceF          = 0.02
vcres.vcRefAnnoOrthogonalPosF = -0.20
vcres.vcRefAnnoFontHeightF    = 0.005

cnres.tiMainString    = "Fully opaque filled vectors over filled contours"

#---Draw fully opaque vectors
uv_plot  = Ngl.vector(wks,u,v,vcres)
spd_plot = Ngl.contour_map(wks,spd,cnres)
Ngl.overlay(spd_plot,uv_plot)
Ngl.maximize_plot(wks, spd_plot)
Ngl.draw(spd_plot)
Ngl.frame(wks)

#---This time make vectors partially transparent
vcres.vcGlyphOpacityF = 0.3
cnres.tiMainString    = "Partially transparent vectors over filled contours"

uv_plot  = Ngl.vector(wks,u,v,vcres)
spd_plot = Ngl.contour_map(wks,spd,cnres)

Ngl.overlay(spd_plot,uv_plot)
Ngl.maximize_plot(wks, spd_plot)
Ngl.draw(spd_plot)
Ngl.frame(wks)


Ngl.end()
