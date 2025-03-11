#
#  File:
#    lcmask_4.py
#
#  Synopsis:
#    Demonstrates how to plot vectors on mask Lambert Conformal Projection map.
#
#  Categories:
#    map plot
#
#  Based on NCL example:
#    lcmask_4.ncl
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
#    o  Adding longitude/latitude labels to a masked Lambert Conformal map
#    o  Moving the main title up
#    o  Attaching text strings to the outside of a plot
#    o  Converting lat/lon values to NDC values
#    o  Changing the angle of text strings
#    o  Adding a carriage return to a text string using a function code
# 
#  Output:
#    Single visualization is produced.
#
'''
  PyNGL Example: 	lcmask_4.py

  -  Adding longitude/latitude labels to a masked Lambert Conformal map
  -  Moving the main title up
  -  Attaching text strings to the outside of a plot
  -  Converting lat/lon values to NDC values
  -  Changing the angle of text strings
  -  Adding a carriage return to a text string using a function code
  
'''
from __future__ import print_function
import numpy as np
import os, sys
import Ngl, Nio

#-------------------------------------------------------
# Function to attach lat/lon labels to a Robinson plot
#-------------------------------------------------------
def add_labels_lcm(wks,map,dlat,dlon):
  PI         = 3.14159
  RAD_TO_DEG = 180./PI

#-- determine whether we are in northern or southern hemisphere
  if (float(minlat) >= 0. and float(maxlat) > 0.):
     HEMISPHERE = "NH"
  else:
     HEMISPHERE = "SH"

#-- pick some "nice" values for the latitude labels.
  lat_values = np.arange(int(minlat),int(maxlat),10)
  lat_values = lat_values.astype(float)
  nlat       = len(lat_values)

#-- We need to get the slope of the left and right min/max longitude lines.
#-- Use NDC coordinates to do this.
  lat1_ndc = 0.
  lon1_ndc = 0.
  lat2_ndc = 0.
  lon2_ndc = 0.
  lon1_ndc,lat1_ndc = Ngl.datatondc(map,minlon,lat_values[0])
  lon2_ndc,lat2_ndc = Ngl.datatondc(map,minlon,lat_values[nlat-1])
  slope_lft         = (lat2_ndc-lat1_ndc)/(lon2_ndc-lon1_ndc)

  lon1_ndc,lat1_ndc = Ngl.datatondc(map,maxlon,lat_values[0])
  lon2_ndc,lat2_ndc = Ngl.datatondc(map,maxlon,lat_values[nlat-1])
  slope_rgt         = (lat2_ndc-lat1_ndc)/(lon2_ndc-lon1_ndc)
  
#-- set some text resources
  txres               = Ngl.Resources()
  txres.txFontHeightF = 0.01
  txres.txPosXF       = 0.1

#-- Loop through lat values, and attach labels to the left and right edges of
#-- the masked LC plot. The labels will be rotated to fit the line better.
  dum_lft       = []                            #-- assign arrays
  dum_rgt       = []                            #-- assign arrays
  lat_label_lft = []                            #-- assign arrays
  lat_label_rgt = []                            #-- assign arrays

  for n in range(0,nlat):
#-- left label
    if(HEMISPHERE == "NH"):
       rotate_val = -90.
       direction  = "N"
    else:
       rotate_val =  90.
       direction  = "S"

#-- add extra white space to labels
    lat_label_lft.append("{}~S~o~N~{}              ".format(str(np.abs(lat_values[n])),direction))
    lat_label_rgt.append("              {}~S~o~N~{}".format(str(np.abs(lat_values[n])),direction))
        
    txres.txAngleF = RAD_TO_DEG * np.arctan(slope_lft) + rotate_val
                             
    dum_lft.append(Ngl.add_text(wks,map,lat_label_lft[n],minlon,lat_values[n],txres))

#-- right label
    if(HEMISPHERE == "NH"):
       rotate_val =  90
    else:
       rotate_val = -90

    txres.txAngleF = RAD_TO_DEG * np.arctan(slope_rgt) + rotate_val

    dum_rgt.append(Ngl.add_text(wks,map,lat_label_rgt[n],maxlon,lat_values[n],txres))

#----------------------------------------------------------------------
# Now do longitude labels. These are harder because we're not adding
# them to a straight line.
# Loop through lon values, and attach labels to the bottom edge for
# northern hemisphere, or top edge for southern hemisphere.
#----------------------------------------------------------------------
  del(txres.txPosXF)
  txres.txPosYF = -5.0

#-- pick some "nice" values for the longitude labels
  lon_values = np.arange(int(minlon+10),int(maxlon-10),10).astype(float)
  lon_values = np.where(lon_values > 180, 360-lon_values, lon_values)
  nlon       = lon_values.size

  dum_bot    = []                            #-- assign arrays
  lon_labels = []                            #-- assign arrays

  if(HEMISPHERE == "NH"):
     lat_val    = minlat
  else:
     lat_val    = maxlat

  ctrl = "~C~"

  for n in range(0,nlon):
    if(lon_values[n] < 0):
       if(HEMISPHERE == "NH"):
          lon_labels.append("{}~S~o~N~W{}".format(str(np.abs(lon_values[n])),ctrl))
       else:
          lon_labels.append("{}{}~S~o~N~W".format(ctrl,str(np.abs(lon_values[n]))))
    elif(lon_values[n] > 0):
       if(HEMISPHERE == "NH"):
          lon_labels.append("{}~S~o~N~E{}".format(str(lon_values[n]),ctrl))
       else:
          lon_labels.append("{}{}~S~o~N~E".format(ctrl,str(lon_values[n])))
    else:
       if(HEMISPHERE == "NH"):
          lon_labels.append("{}0~S~o~N~{}".format(ctrl,ctrl))
       else:
          lon_labels.append("{}0~S~o~N~{}".format(ctrl,ctrl))

#-- For each longitude label, we need to figure out how much to rotate
#-- it, so get the approximate slope at that point.
    if(HEMISPHERE == "NH"):             #-- add labels to bottom of LC plot
       lon1_ndc,lat1_ndc = Ngl.datatondc(map, lon_values[n]-0.5, minlat)
       lon2_ndc,lat2_ndc = Ngl.datatondc(map, lon_values[n]+0.5, minlat)
       txres.txJust = "TopCenter"
    else:                               #-- add labels to top of LC plot
       lon1_ndc,lat1_ndc = Ngl.datatondc(map, lon_values[n]+0.5, maxlat)
       lon2_ndc,lat2_ndc = Ngl.datatondc(map, lon_values[n]-0.5, maxlat)
       txres.txJust = "BottomCenter"

    slope_bot = (lat1_ndc-lat2_ndc)/(lon1_ndc-lon2_ndc)
    txres.txAngleF  =  RAD_TO_DEG * np.arctan(slope_bot)
    
#-- attach to map
    dum_bot.append(Ngl.add_text(wks, map, str(lon_labels[n]), \
                                lon_values[n], lat_val, txres))
  return

#-------------------------------------------------------
#                       MAIN
#-------------------------------------------------------
#--  define variables
diri  = "./"                            #-- data directory
fname = "atmos.nc"                      #-- data file name

#---Test if file exists
if(not os.path.exists(diri+fname)):
   print("You do not have the necessary file (%s) to run this example." % (diri+fname))
   print("You can get the files from the NCL website at:")
   print("http://www.ncl.ucar.edu/Document/Manuals/NCL_User_Guide/Data/")
   sys.exit()

#-- open file and read variables
a   =  Nio.open_file(diri + fname,"r")                  #-- open data file
v   =  a.variables["V"][0,0,:,:]                        #-- read variable V
lon =  a.variables["lon"][:]                            #-- get whole "lon" variable
lat =  a.variables["lat"][:]                            #-- get whole "lat" variable

#tnew,lonnew = Ngl.add_cyclic(t,lon)                     #-- add cyclic point

#-- open workstation
wks = Ngl.open_wks("png","plot_lcmask_4")               #-- open workstation

#-- set sub-region
minlon, maxlon = -90. , 40.                             #-- min/max lon to mask
minlat, maxlat =  20. , 80.                             #-- min/max lat to mask

#-- retrieve the indices of lat/lon sub-region
indminlon = min(range(len(lon)), key=lambda i: abs(lon[i]-minlon))
indmaxlon = max(range(len(lon)), key=lambda i: abs(lon[i]+maxlon))
indminlat = min(range(len(lat)), key=lambda i: abs(lat[i]-minlat))
indmaxlat = max(range(len(lat)), key=lambda i: abs(lat[i]+maxlat))

#-- select sub-region
vregion = v[indminlat:indmaxlat,indminlon:indmaxlon]

#-- set common resources
res                        =  Ngl.Resources()         #-- resource object
res.nglMaximize            =  True                    #-- maximize plot output
res.nglDraw                =  False                   #-- don't draw plot
res.nglFrame               =  False                   #-- don't advance the frame

res.cnFillOn               =  True                    #-- turn on color
res.cnLinesOn              =  False                   #-- turn off contour lines
res.cnLineLabelsOn         =  False                   #-- turn off contour line labels
res.cnFillPalette          = "rainbow"                #-- set color map
res.cnLevelSelectionMode   = "ManualLevels"           #-- select manual levels
res.cnMinLevelValF         =  0                       #-- minimum contour value
res.cnMaxLevelValF         =  40                      #-- maximum contour value
res.cnLevelSpacingF        =  4                       #-- contour increment

res.mpFillOn               =  False                   #-- turn map fill on
res.mpOutlineOn            =  False                   #-- outline map
res.mpProjection           = "LambertConformal"
res.pmTickMarkDisplayMode  = "Always"
res.mpGridLatSpacingF      =  10
res.mpGridLonSpacingF      =   5
  
res.nglMaskLambertConformal = True                    #-- turn on lc masking
res.mpLambertParallel1F    =  10
res.mpLambertParallel2F    =  70
res.mpLambertMeridianF     = -100
res.mpLimitMode            = "LatLon"
res.mpMinLonF              =  minlon
res.mpMaxLonF              =  maxlon
res.mpMinLatF              =  minlat
res.mpMaxLatF              =  maxlat

res.lbOrientation          = "horizontal"             #-- horizontal labelbar
res.lbLabelFontHeightF     =  0.01                    #-- make the labelbar labels smaller

res.tiMainString           = "Adding lat/lon labels to a masked LC map" #-- title
res.tiMainFontHeightF      =  0.012                   #-- main title font size

lon =  lon - 180                                      #-- make lon go -180 to 180 

res.sfXArray               =  lon[indminlon:indmaxlon]  #-- use scalar field
res.sfYArray               =  lat[indminlat:indmaxlat]  #-- use scalar field

#-- create and draw the basic map
plot = Ngl.contour_map(wks,vregion,res)               #-- create plot

#-- add lat/lon labels to the plot
tx = add_labels_lcm(wks,plot,10,10)

#-- draw the plot and advance the frame
Ngl.draw(plot)
Ngl.frame(wks)

Ngl.end()
