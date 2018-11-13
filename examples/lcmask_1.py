#
#  File:
#    lcmask_1.py
#
#  Synopsis:
#    Demonstrates how to mask Lambert Conformal Projection map.
#    Shows how to add your own longitude/latitude labels to a 
#    masked Lambert Conformal plot.
#
#  Categories:
#    map plot
#
#  Based on NCL example:
#    lcmask_1.ncl
#
#  Author:
#    Karin Meier-Fleischer
#  
#  Date of initial publication:
#    November 2018
#
#  Description:
#    This example shows how to create a masked lambert conformal projection map.
#
#  Effects illustrated:
#    o  Drawing filled contours over a Lambert Conformal map
#    o  Drawing a filled contours over a masked Lambert Conformal plot
#    o  Zooming in on a particular area on a Lambert Conformal map
#    o  Using a blue-white-red color map
#    o  Setting contour levels using a min/max contour level and a spacing
#    o  Turning off the addition of a longitude cyclic point
# 
#  Output:
#    Two visualizations are produced.
#
'''
  PyNGL Example: 	lcmask_1.py

  -  Drawing filled contours over a Lambert Conformal map
  -  Drawing a filled contours over a masked Lambert Conformal plot
  -  Zooming in on a particular area on a Lambert Conformal map
  -  Using a blue-white-red color map
  -  Setting contour levels using a min/max contour level and a spacing
  -  Turning off the addition of a longitude cyclic point
  
'''
from __future__ import print_function
import numpy as np
import os, sys
import Ngl, Nio

#--  define variables
diri  = "/Users/k204045/NCL/tests/"     #-- data directory
fname = "atmos.nc"                      #-- data file name

#---Test if file exists
if(not os.path.exists(diri+fname)):
   print("You do not have the necessary file (%s) to run this example." % (diri+fname))
   print("You can get the files from the NCL website at:")
   print("http://www.ncl.ucar.edu/Document/Manuals/NCL_User_Guide/Data/")
   sys.exit()

#-- open file and read variables
a   =  Nio.open_file(diri + fname,"r")                  #-- open data file
t   =  a.variables["V"][0,0,:,:]                        #-- variable V
lon =  a.variables["lon"][:]                            #-- get whole "lon" variable
lat =  a.variables["lat"][:]                            #-- get whole "lat" variable

tnew,lonnew = Ngl.add_cyclic(t,lon)                     #-- add cyclic point

#-- open workstation
wks = Ngl.open_wks("png","plot_lcmask_1")               #-- open workstation

#-- set sub-region
minlon, maxlon = -90. , 40.                             #-- min/max lon to mask
minlat, maxlat =  20. , 80.                             #-- min/max lat to mask

#-- retrieve the indices of lat/lon sub-region
indminlon = min(range(len(lon)), key=lambda i: abs(lon[i]-minlon))
indmaxlon = max(range(len(lon)), key=lambda i: abs(lon[i]+maxlon))
indminlat = min(range(len(lat)), key=lambda i: abs(lat[i]-minlat))
indmaxlat = max(range(len(lat)), key=lambda i: abs(lat[i]+maxlat))

#-- select sub-region
tregion = t[indminlat:indmaxlat,indminlon:indmaxlon]

#-- set common resources
mpres                        =  Ngl.Resources()         #-- resource object
mpres.nglMaximize            =  True                    #-- maximize plot output
mpres.nglFrame               =  False                   #-- don't advance the frame

mpres.cnFillOn               =  True
mpres.cnFillPalette          = "BlWhRe"                 #-- set color map
mpres.cnLinesOn              =  False                   #-- turn off contour lines
mpres.cnLineLabelsOn         =  False                   #-- turn off contour line labels
mpres.cnLevelSelectionMode   = "ManualLevels"           #-- set manual contour levels
mpres.cnMinLevelValF         =  -40                     #-- set min contour level
mpres.cnMaxLevelValF         =   40                     #-- set max contour level
mpres.cnLevelSpacingF        =    4                     #-- set contour spacing

mpres.mpFillOn               =  False                   #-- turn map fill on
mpres.mpOutlineOn            =  True                    #-- outline map
mpres.mpProjection           = "LambertConformal"
mpres.mpGridAndLimbOn        =  False                   #-- turn off grid lines

mpres.pmTickMarkDisplayMode  = "Never"                  #-- turn off axis labels
mpres.lbOrientation          = "horizontal"             #-- horizontal labelbar

#------------------------------------
#-- first plot: unmasked plot
#------------------------------------
mpres.sfXArray               =  lonnew                  #-- use scalar field
mpres.sfYArray               =  lat                     #-- use scalar field

#-- create and draw the basic map
map = Ngl.contour_map(wks,tnew,mpres)

#-- write variable long_name and units to the plot
txres               = Ngl.Resources()
txres.txFontHeightF = 0.016
Ngl.text_ndc(wks,"meridional wind component",             0.27,0.99,txres)
Ngl.text_ndc(wks,a.variables["V"].attributes['units'],    0.85,0.99,txres)

#-- advance the frame
Ngl.frame(wks)

#------------------------------------
#-- second plot: masked plot
#------------------------------------
lon =  lon - 180                                        #-- make lon go -180 to 180 

mpres.nglMaskLambertConformal = True                    #-- turn on lc masking
mpres.mpLambertParallel1F    =  10
mpres.mpLambertParallel2F    =  70
mpres.mpLambertMeridianF     = -100
mpres.mpLimitMode            = "LatLon"
mpres.mpMinLonF              =  minlon
mpres.mpMaxLonF              =  maxlon
mpres.mpMinLatF              =  minlat
mpres.mpMaxLatF              =  maxlat

mpres.sfXArray               =  lon[indminlon:indmaxlon]  #-- use scalar field
mpres.sfYArray               =  lat[indminlat:indmaxlat]  #-- use scalar field

#-- create and draw the basic map
map = Ngl.contour_map(wks,tregion,mpres)

#-- write variable long_name and units to the plot
Ngl.text_ndc(wks,"meridional wind component",             0.14,0.86,txres)
Ngl.text_ndc(wks,a.variables["V"].attributes['units'],    0.95,0.86,txres)

#-- advance the frame
Ngl.frame(wks)

Ngl.end()
