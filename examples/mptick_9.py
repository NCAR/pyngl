#
#  File:
#    mptick_9.py
#
#  Synopsis:
#    Demonstrates how to explicitly label the axes with the tickmarks and labels you want. 
#    You can use this for rectangular map projections only.
#
#  Categories:
#    map plot
#
#  Based on NCL example:
#    mptick_9.ncl
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
#    o  Explicitly setting map tickmarks and labels
#    o  Attaching a plot as an annotation of another plot
#    o  Drawing a lambert conformal map
#    o  Drawing a mercator map
#    o  Maximizing plots after they've been created
#    o  Converting NDC values to lat/lon values
#    o  Using "get_float" to retrieve the size of a plot
#    o  Creating a blank plot
#    o  Drawing a map using the medium resolution map outlines
#    o  Moving the main title up
# 
#  Output:
#    A single visualization is produced.
#
'''
  PyNGL Example: 	mptick_9.py

  -  Explicitly setting map tickmarks and labels
  -  Attaching a plot as an annotation of another plot
  -  Drawing a lambert conformal map
  -  Drawing a mercator map
  -  Maximizing plots after they've been created
  -  Converting NDC values to lat/lon values
  -  Using "get_float" to retrieve the size of a plot
  -  Creating a blank plot
  -  Drawing a map using the medium resolution map outlines
  -  Moving the main title up
  
'''
from __future__ import print_function
import numpy as np
import Ngl

#-------------------------------------------------------
# Function to attach lat/lon labels to a Robinson plot
#-------------------------------------------------------
def add_labels(wks,map,dlat,dlon):

#-- generate lat/lon values
  dla = (180.0/dlat)+1                  #-- number of lat labels
  dlo = (360.0/dlon)+1                  #-- number of lon lables
  
  lat_values = np.linspace( -90.0,  90.0, dla, endpoint=True)
  lon_values = np.linspace(-180.0, 180.0, dlo, endpoint=True)
  nlat       = len(lat_values)
  nlon       = len(lon_values)

#-- assign arrays to hold the labels
  lft, rgt                     = [],[]
  lat_lft_label, lat_rgt_label = [],[]
  lon_bot_label                = []

#-- text resources
  txres               = Ngl.Resources()
  txres.txFontHeightF = 0.01

#-- add degree sign and S/N to the latitude labels
#-- don't write 90S label which would be too close to the lon labels
  for l in lat_values:
      if l == -90.0:  
         lat_lft_label.append("".format(l)) 
         lat_rgt_label.append("".format(l))
      elif l < 0:
         lat_lft_label.append("{}~S~o~N~S    ".format(np.fabs(l)))
         lat_rgt_label.append("    {}~S~o~N~S".format(np.fabs(l)))
      elif l > 0:
         lat_lft_label.append("{}~S~o~N~N    ".format(l))
         lat_rgt_label.append("    {}~S~o~N~N".format(l))
      else:
         lat_lft_label.append("0  ")
         lat_rgt_label.append("   0")

#-- add degree sign and W/E to the longitude labels
  for l in lon_values:
      if l < 0:
         lon_bot_label.append("{}~S~o~N~W".format(np.fabs(l)))
      elif l > 0:
         lon_bot_label.append("{}~S~o~N~E".format(l))
      else:
         lon_bot_label.append("0")

#-- add the latitude labels left and right to the plot
  for n in range(0,nlat):
     txres.txJust = "CenterRight"
     lft.append(Ngl.add_text(wks,map,lat_lft_label[n],-180.0,\
                                      lat_values[n],txres))
     txres.txJust = "CenterLeft"
     rgt.append(Ngl.add_text(wks,map,lat_rgt_label[n],180.0,\
                                       lat_values[n],txres))
#-- add the longitude labels at the bottom of the plot
  bot = []
  for n in range(0,nlon):
     txres.txJust = "TopCenter"
     bot.append(Ngl.add_text(wks,map,lon_bot_label[n],lon_values[n],\
                                      -90.0,txres))
  return

#-------------------------------------------------------
#                       MAIN
#-------------------------------------------------------
wks = Ngl.open_wks("png","plot_mptick_9")               #-- open workstation

#-----------------------------------
#-- first plot: Lambert Conformal
#-----------------------------------
mpres                        =  Ngl.Resources()         #-- resource object
mpres.nglDraw                =  False                   #-- turn off plot draw and frame advance. We will
mpres.nglFrame               =  False                   #-- do it later after adding subtitles.

mpres.mpFillOn               =  True                    #-- turn map fill on
mpres.mpOutlineOn            =  False                   #-- outline map
mpres.mpOceanFillColor       = "Transparent"            #-- set ocean fill color to transparent
mpres.mpLandFillColor        = "Gray90"                 #-- set land fill color to gray
mpres.mpInlandWaterFillColor = "Gray90"                 #-- set inland water fill color to gray

mpres.tiMainString           = "Lambert Conformal Map"
mpres.tiMainOffsetYF         =  0.05
 
mpres.mpProjection           = "LambertConformal"
mpres.mpLambertParallel1F    =  10
mpres.mpLambertParallel2F    =  70
mpres.mpLambertMeridianF     = -100
mpres.mpLimitMode            = "LatLon"
mpres.mpMinLatF              =  20
mpres.mpMaxLatF              =  55
mpres.mpMinLonF              = -140
mpres.mpMaxLonF              = -60
mpres.mpGridAndLimbOn        =  True
mpres.mpGridSpacingF         =  10.

mpres.pmTickMarkDisplayMode  = "Always"
  
#-- create and draw the basic map
map = Ngl.map(wks,mpres)

#-- add labels to the plot
tx = add_labels(wks,map,30,90)

#-- draw the plot and advance the frame
Ngl.draw(map)
Ngl.frame(wks)

del(mpres)  #-- delete resource object

#-----------------------------------
#-- second plot: Mercator projection
#-----------------------------------
mpres                        =  Ngl.Resources()
mpres.nglDraw                =  False                   #-- turn off plot draw and frame advance. We will
mpres.nglFrame               =  False                   #-- do it later after adding subtitles.

mpres.tiMainString           = "Mercator Map"           #-- title
mpres.tiMainFontHeightF      =  0.020                   #-- increase font size

mpres.mpFillOn               =  False                   #-- turn off land fill
mpres.mpDataBaseVersion      = "MediumRes"              #-- use finer database
mpres.mpProjection           = "mercator"               #-- projection
mpres.mpLimitMode            = "Corners"                #-- method to zoom
mpres.mpLeftCornerLatF       =   32
mpres.mpLeftCornerLonF       =  128
mpres.mpRightCornerLatF      =   55
mpres.mpRightCornerLonF      =  144

mpres.pmTickMarkDisplayMode  = "Always"                 #-- turn on tickmarks

#-- create and draw the basic map
map = Ngl.map(wks,mpres)

#-- maximize the plot, draw it and advance the frame
Ngl.maximize_plot(wks,map)
Ngl.draw(map)
Ngl.frame(wks)

Ngl.end()
