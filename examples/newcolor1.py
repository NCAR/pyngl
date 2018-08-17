#
#  File:
#    newcolor1.py
#
#  Synopsis:
#    Illustrates new color capabilities in PyNGL 1.5.0.
#
#  Categories:
#    Contouring
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    November 2012
#
#  Description:
#    This example shows how to use the new cnFillPalette resource.
#
#  Effects illustrated:
#    o  Using the new "cnFillPalette" resource.
#    o  Using a named color without having to add it to color map
# 
#  Output:
#    This example produces one visualization
#
#  Notes:
#     
from __future__ import print_function
import os, numpy
import Ngl, Nio

#
# Create some dummy data for the contour plot.
#
dirc = Ngl.pynglpath("data")
f    = Nio.open_file(os.path.join(dirc,"cdf","uv300.nc"))
u    = f.variables["U"][1,:,:]
lat  = f.variables["lat"][:]
lon  = f.variables["lon"][:]
 
wks_type = "png"
wks = Ngl.open_wks(wks_type,"newcolor1")

cnres                 = Ngl.Resources()

# Contour resources
cnres.cnFillOn        = True
cnres.cnFillPalette   = "BlueYellowRed"      # New in PyNGL 1.5.0
cnres.cnLinesOn       = False
cnres.cnLineLabelsOn  = False

# Labelbar resource
cnres.lbOrientation   = "horizontal"

# Scalar field resources
cnres.sfXArray        = lon
cnres.sfYArray        = lat

# Map resources
cnres.mpFillOn               = True
cnres.mpFillDrawOrder        = "PostDraw"
cnres.mpLandFillColor        = "Gray"
cnres.mpOceanFillColor       = "Transparent"
cnres.mpInlandWaterFillColor = "Transparent"

contour = Ngl.contour_map(wks,u,cnres)

Ngl.end()
