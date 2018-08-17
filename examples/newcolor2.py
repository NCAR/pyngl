#
#  File:
#    newcolor2.py
#
#  Synopsis:
#    Illustrates new color capabilities in PyNGL 1.5.0.
#
#  Categories:
#    xy plots
#
#  Author:
#    Mary Haley, based on NCL example by Rick Brownrigg
#  
#  Date of initial publication:
#    November 2012
#
#  Description:
#    This example shows how to draw partially opaque text on a plot
#
#  Effects illustrated:
#    o Using the new txFontOpacityF resource
#    o Subscripting a variable using coordinate values
# 
#  Output:
#     A single visualization with two XY curves 
#
from __future__ import print_function
import numpy,os
import Ngl,Nio

dirc = Ngl.pynglpath("data")
f    = Nio.open_file(os.path.join(dirc,"cdf","uv300.nc"))
u    = f.variables["U"][0,:,8]
lat  = f.variables["lat"][:]

#---Start the graphics section
wks_type = "png"
wks = Ngl.open_wks (wks_type,"newcolor2")     # Open "newcolor2.png" for graphics

#---Set some resources.
res          = Ngl.Resources()
res.nglDraw  = False
res.nglFrame = False
res.xyLineThicknessF = 4.0

plot  = Ngl.xy (wks,lat,u,res)              # Create plot

txres                = Ngl.Resources()         # text mods desired
txres.txFont         = 30
txres.txFontHeightF  = 0.04                    # font smaller. default big
txres.txFontOpacityF = 0.10                    # highly transparent
#txres.txFontOpacityF = 0.5                    # half transparent
txres.txAngleF       = 45.

txid = Ngl.add_text(wks,plot,"Preliminary Data",10,15,txres) 

Ngl.draw(plot)        # Drawing plot will draw attached text
Ngl.frame(wks)        # Advance frame


Ngl.end()

