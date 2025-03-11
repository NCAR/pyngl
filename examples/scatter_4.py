#
#  File:
#    scatter_4.py
#
#  Synopsis:
#    Demonstrates a scatter plot with a regression line.
#
#  Category:
#    xy plots
#    overlay
#
#  Based on NCL example:
#    scatter_4.ncl
#
#  Author:
#    Karin Meier-Fleischer
#  
#  Date of initial publication:
#    November, 2018
#
#  Description:
#    This example shows how to create a scatter plot with a regression line.
#
#  Effects illustrated:
#    o  Define function runave
#    o  Drawing a scatter plot with a regression line
#    o  Drawing a time series plot
#    o  Calculating the least squared regression for a one dimensional array
#    o  Smoothing data so that seasonal cycle is less prominent
#    o  Changing the markers in an XY plot
#    o  Changing the marker mode in an XY plot
#    o  Changing the marker color in an XY plot
#    o  Changing the marker size in an XY plot
#    o  Overlay
#
#  Output:
#     A single visualization is produced.     
#
'''
  PyNGL Example: 	scatter_4.py

  -  Define function runave
  -  Drawing a scatter plot with a regression line
  -  Drawing a time series plot
  -  Calculating the least squared regression for a one dimensional array
  -  Smoothing data so that seasonal cycle is less prominent
  -  Changing the markers in an XY plot
  -  Changing the marker mode in an XY plot
  -  Changing the marker color in an XY plot
  -  Changing the marker size in an XY plot
  -  Overlay
  
'''
from __future__ import print_function
import numpy as np
import os,sys
import Ngl,Nio

#---------------------------------------------------------------
# Function runave(data, xrange): compute the running mean
#
# 						Input:	data 	  data array
#								xrange    time range
#
#						Return:	running mean values
#---------------------------------------------------------------
def runave(data, xrange):
    return np.convolve(data, np.ones((xrange,))/xrange, mode='valid')

#---------------------------------------------------------------
# Function regline(xin,yin): 	 compute the regression line
#
# 						Input:	xin 	  x array
#								yin    	  data array
#
#						Return:	x,y		  x- and y-values of regression line
#---------------------------------------------------------------
def regline(xin,yin):
	x    = np.vstack([xin, np.ones(len(xin))]).T
	m, t = np.linalg.lstsq(x, yin, rcond=None)[0]            #-- returns gradient and y-intersection
	y    = (m*x)+t                                           #-- regression line equation
	return x, y
    
    
#---------------------------------------------------------------
#                     MAIN
#---------------------------------------------------------------
#-- data path and file name
ncarg_root = os.environ.get('NCARG_ROOT')
diri  = ncarg_root + '/lib/ncarg/data/nug/'
fname = 'tas_rectilinear_grid_2D.nc'

#---Test if file exists
if(not os.path.exists(diri+fname)):
   print("You do not have the necessary file (%s) to run this example." % (diri+fname))
   print("You can get the files from the NCL website at:")
   print("http://www.ncl.ucar.edu/Document/Manuals/NCL_User_Guide/Data/")
   sys.exit()

#-- open file and read variables
a    =  Nio.open_file(diri + fname,"r")                 #-- open data file
tas  =  a.variables["tas"][:,:,:]                       #-- read variable tas
lat  =  a.variables["lat"][:]                           #-- get whole "lat" variable
lon  =  a.variables["lon"][:]                           #-- get whole "lon" variable
time =  a.variables["time"][:]                          #-- number of timesteps
ntime = time.size

#-- set lat/lon point
latval =  60                                            #-- extract lat 60N
lonval = 180                                            #-- extract lon 180W

#-- retrieve the indices of lat/lon point
indlat = min(range(len(lat)), key=lambda i: abs(lat[i]-latval))
indlon = min(range(len(lon)), key=lambda i: abs(lon[i]-lonval))

#-- select the data of lat/lon point for all timesteps
ts = tas[:,indlat,indlon]

#-----------------------------------------------------------
# Smooth data so that seasonal cycle is less prominent. This 
# is for demo purposes only so that the regression line is 
# more sloped.
#-----------------------------------------------------------
rmean = runave(ts,40)

#-----------------------------------------------------------
# Create x and calculate the regression coefficient.
#-----------------------------------------------------------
x, y = regline(time,ts)

#-----------------------------------------------------------
# Graphics
#-----------------------------------------------------------
wks = Ngl.open_wks("png","plot_scatter_4")

#-- plot resources
res                 =  Ngl.Resources()
res.nglDraw         =  False                    #-- don't draw plot
res.nglFrame        =  False                    #-- don't advance the frame
res.nglMaximize     =  True                     #-- maximize plot
 
res.tiMainString    = "Output from np.linalg.lstsq (regline)" #-- add title
res.tiXAxisString   = "simulation time" 
res.tiYAxisString   = "Surface temperature" 

res.xyMarkLineModes = "Markers"                 #-- use markers
res.xyMarker        =  16                       #-- filled circle
res.xyMarkerSizeF   =  0.005                    #-- default: 0.01
res.xyMarkerColor   = "red"                     #-- marker color

#-- create xy-plot of ts
plot0 = Ngl.xy(wks,time,ts,res)                 #-- create the xy plot

#-- change resource settings for regression line
res.xyMarkLineModes = "Lines"                   #-- use lines
res.xyMarker        =  1                        #-- solid line
res.xyMarkerColor   = "black"                   #-- marker color

#-- create plot of regression line
plot1 = Ngl.xy(wks,x,y,res)

#-- overlay regression line plot on xy plot
Ngl.overlay(plot0,plot1)

#-- draw plot and advance the frame
Ngl.draw(plot0)
Ngl.frame(wks)

Ngl.end()


