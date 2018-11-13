#
#  File:
#    mptick_1.py
#
#  Synopsis:
#    Demonstrates the default tickmarks of a cylindrical equidistant plot.
#
#  Categories:
#    map plot
#
#  Based on NCL example:
#    mptick_1.ncl
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
#    o  Drawing a default cylindrical equidistant map
#    o  Drawing default tickmarks on a cylindrical equidistant map
# 
#  Output:
#    A single visualization is produced.
#   
'''
  PyNGL Example: 	mptick_1.py

  -  Drawing a default cylindrical equidistant map
  -  Drawing default tickmarks on a cylindrical equidistant map
 
'''
from __future__ import print_function
import Ngl

wks = Ngl.open_wks("png","plot_mptick_1")               #-- open workstation

mpres                        =  Ngl.Resources()         #-- resource object
mpres.mpFillOn               =  True                    #-- turn map fill on
mpres.mpOutlineOn            =  False                   #-- outline map
mpres.mpOceanFillColor       = "Transparent"            #-- set ocean fill color to transparent
mpres.mpLandFillColor        = "Gray90"                 #-- set land fill color to gray
mpres.mpInlandWaterFillColor = "Gray90"                 #-- set inland water fill color to gray
mpres.mpProjection           = "CylindricalEquidistant" #-- map projection
mpres.mpGridAndLimbOn        =  False                   #-- turn off grid lines

map = Ngl.map(wks,mpres)                                #-- create the plot

Ngl.end()
