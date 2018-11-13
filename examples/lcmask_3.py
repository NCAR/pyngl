#
#  File:
#    lcmask_3.py
#
#  Synopsis:
#    Demonstrates how to plot vectors on mask Lambert Conformal Projection map.
#
#  Categories:
#    map plot
#
#  Based on NCL example:
#    lcmask_3.ncl
#
#  Author:
#    Karin Meier-Fleischer
#  
#  Date of initial publication:
#    November 2018
#
#  Description:
#    This example shows how to create a masked lambert conformal projection map
#    and plot vectors on the map.
#
#  Effects illustrated:
#    o  Drawing curly vectors over a masked Lambert Conformal map
#    o  Maximizing the size of the plot
#    o  Changing the length of the vectors
#    o  Decreasing the number of vectors drawn
# 
#  Output:
#    Two visualizations are produced.
#
'''
  PyNGL Example: 	lcmask_3.py

  -  Drawing curly vectors over a masked Lambert Conformal map
  -  Maximizing the size of the plot
  -  Changing the length of the vectors
  -  Decreasing the number of vectors drawn
  
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
u   =  a.variables["U"][0,0,:,:]                        #-- variable U
v   =  a.variables["V"][0,0,:,:]                        #-- variable V
lon =  a.variables["lon"][:]                            #-- get whole "lon" variable
lat =  a.variables["lat"][:]                            #-- get whole "lat" variable

#tnew,lonnew = Ngl.add_cyclic(t,lon)                     #-- add cyclic point

#-- open workstation
wks = Ngl.open_wks("png","plot_lcmask_3")               #-- open workstation

#-- set sub-region
minlon, maxlon = -90. , 40.                             #-- min/max lon to mask
minlat, maxlat =  20. , 80.                             #-- min/max lat to mask

#-- retrieve the indices of lat/lon sub-region
indminlon = min(range(len(lon)), key=lambda i: abs(lon[i]-minlon))
indmaxlon = max(range(len(lon)), key=lambda i: abs(lon[i]+maxlon))
indminlat = min(range(len(lat)), key=lambda i: abs(lat[i]-minlat))
indmaxlat = max(range(len(lat)), key=lambda i: abs(lat[i]+maxlat))

#-- select sub-region
uregion = u[indminlat:indmaxlat,indminlon:indmaxlon]
vregion = v[indminlat:indmaxlat,indminlon:indmaxlon]

#-- set common resources
res                        =  Ngl.Resources()         #-- resource object
res.nglMaximize            =  True                    #-- maximize plot output
res.nglFrame               =  False                   #-- don't advance the frame

res.mpFillOn               =  True                    #-- turn map fill on
res.mpOutlineOn            =  False                    #-- outline map
res.mpOceanFillColor       = "Transparent"            #-- set ocean fill color to transparent
res.mpLandFillColor        = "Gray90"                 #-- set land fill color to gray
res.mpInlandWaterFillColor = "Gray90"                 #-- set inland water fill color to gray
res.mpProjection           = "LambertConformal"
res.mpGridAndLimbOn        =  False                   #-- turn off grid lines

res.pmTickMarkDisplayMode  = "Never"                  #-- turn off axis labels
res.lbOrientation          = "horizontal"             #-- horizontal labelbar

res.nglMaskLambertConformal = True                    #-- turn on lc masking
res.mpLambertParallel1F    =  10
res.mpLambertParallel2F    =  70
res.mpLambertMeridianF     = -100
res.mpLimitMode            = "LatLon"
res.mpMinLonF              =  minlon
res.mpMaxLonF              =  maxlon
res.mpMinLatF              =  minlat
res.mpMaxLatF              =  maxlat

#-- vector rescoures
res.vcRefMagnitudeF        =  15.0                    #-- define vector ref mag
res.vcRefLengthF           =  0.045                   #-- define length of vec ref
res.vcGlyphStyle           = "CurlyVector"            #-- turn on curly vectors
res.vcMinDistanceF         =  0.017

res.tiMainString           = "Vectors on Masked Grid" #-- title
res.tiMainFontHeightF      =  0.012                   #-- main title font size

lon =  lon - 180                                      #-- make lon go -180 to 180 

res.vfXArray               =  lon[indminlon:indmaxlon]  #-- use scalar field
res.vfYArray               =  lat[indminlat:indmaxlat]  #-- use scalar field

#-- create and draw the basic map
plot = Ngl.vector_map(wks,uregion,vregion,res)        #-- create plot
  
#-- write variable long_name and units to the plot
txres               = Ngl.Resources()
txres.txFontHeightF = 0.016
Ngl.text_ndc(wks,"winds",                                 0.14,0.76,txres)
Ngl.text_ndc(wks,a.variables["V"].attributes['units'],    0.95,0.76,txres)

#-- advance the frame
Ngl.frame(wks)

Ngl.end()
