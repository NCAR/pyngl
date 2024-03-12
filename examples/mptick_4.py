#
#  File:
#    mptick_4.py
#
#  Synopsis:
#    Demonstrates how to adjust the tickmark spacing.
#
#  Categories:
#    map plot
#    annotation
#
#  Based on NCL example:
#    mptick_4.ncl
#
#  Author:
#    Karin Meier-Fleischer
#  
#  Date of initial publication:
#    November 2018
#
#  Description:
#    This example shows how to create a map.
#
#  Effects illustrated:
#    o  Setting the latitude/longitude spacing for map tickmarks
# 
#  Output:
#    A single visualization is produced.
#
'''
  PyNGL Example: 	mptick_4.py

  - Setting the latitude/longitude spacing for map tickmarks
  
'''
from __future__ import print_function
import numpy as np
import Ngl

lat_spacing = 20                                    #-- y-axis label increment 
lon_spacing = 50                                    #-- x-axis label increment

wks = Ngl.open_wks("png","plot_mptick_4")           #-- open workstation

mpres                        =  Ngl.Resources()     #-- resource object
mpres.nglDraw                =  False               #-- Don't draw map
mpres.nglFrame               =  False               #-- Don't advance frame

mpres.mpFillOn               =  True                #-- turn map fill on
mpres.mpOutlineOn            =  False               #-- outline map
mpres.mpOceanFillColor       = "Transparent"        #-- set ocean fill color to transparent
mpres.mpLandFillColor        = "Gray90"             #-- set land fill color to gray
mpres.mpInlandWaterFillColor = "Gray90"             #-- set inland water fill color to gray
mpres.mpProjection           = "CylindricalEquidistant" #-- map projection
mpres.mpGridAndLimbOn        =  False               #-- turn off grid lines
mpres.pmTickMarkDisplayMode  = "Never"              #-- turn on built-in tickmarks

map = Ngl.map(wks,mpres)                            #-- create the base plot

#-- create x-axis labels starting with 150 degrees west, every 50th degrees
lon_values = np.arange(-150.0,211.0,lon_spacing)
lon_labels = []
for l in lon_values:
   if l < 0:
      lon_labels.append("{}~S~o~N~W".format(np.fabs(l)))
   elif l > 0:
      lon_labels.append("{}~S~o~N~E".format(l))
   else:
      lon_labels.append("0")
  
#-- create y-axis labels starting with 80 degrees south, every 20th degrees
lat_values = np.arange(-80.0,101.0,lat_spacing)
lat_labels = []
for l in lat_values:
   if l < 0:
      lat_labels.append("{}~S~o~N~S".format(np.fabs(l)))
   elif l > 0:
      lat_labels.append("{}~S~o~N~N".format(l))
   else:
      lat_labels.append("0")

#-- new resources for the blank plot
bres                         =  Ngl.Resources()
bres.nglMaximize             =  False
bres.nglPointTickmarksOutward = True                #-- tick marks outside

bres.vpXF                    =  Ngl.get_float(map,"vpXF")
bres.vpYF                    =  Ngl.get_float(map,"vpYF")
bres.vpWidthF                =  Ngl.get_float(map,"vpWidthF")
bres.vpHeightF               =  Ngl.get_float(map,"vpHeightF")

bres.trXMinF                 =  Ngl.get_float(map,"trXMinF")
bres.trXMaxF                 =  Ngl.get_float(map,"trXMaxF")
bres.trYMinF                 =  Ngl.get_float(map,"trYMinF")
bres.trYMaxF                 =  Ngl.get_float(map,"trYMaxF")
  
bres.tmXBMode                = "Explicit"           #-- use explicit x-axis values and labels
bres.tmXBValues              =  lon_values          #-- x-axis values
bres.tmXBLabels              =  lon_labels          #-- x-axis labels
bres.tmXBLabelFontHeightF    =  0.015               #-- larger x-axis labels

bres.tmYLMode                = "Explicit"           #-- use explicit y-axis values and labels
bres.tmYLValues              =  lat_values          #-- y-axis values
bres.tmYLLabels              =  lat_labels          #-- y-axis labels
bres.tmYLLabelFontHeightF    =  0.015               #-- larger y-axis labels

#-- create a blank plot with the axes annotations
blank  = Ngl.blank_plot(wks,bres)

#-- attach the blank plot to the map
sres                         = Ngl.Resources()
sres.amZone                  = 0                    #-- '0' means centered over base plot
sres.amResizeNotify          = True                 #-- adjust size of plot to base plot
Ngl.add_annotation(map,blank,sres)

#-- maximize the plot, draw the plot, and advance the frame
Ngl.maximize_plot(wks,map)                          #-- maximize the map plot
Ngl.draw(map)
Ngl.frame(wks)

Ngl.end()
