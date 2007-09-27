#
#  File:
#    wmstnm01.py
#
#  Synopsis:
#    Illustrates plotting station model data.
#
#  Category:
#    Wind barbs/Station model data.
#
#  Author:
#    Fred Clare
#
#  Date of initial publication:
#    September, 2007 
#
#  Description:
#    Draws a single large station model data plot.
#
#  Effects illustrated:
#    o  Drawing station model data.
# 
#  Output:
#    A single visualization is produced that draws the 
#    given station model data.
#
#  Notes:
#     

import Ngl
import numpy

#
#  Define a color map and open a workstation.
#
cmap = numpy.array([[1., 1., 1.], [0., 0., 0.]], 'f')
rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"wmstnm01",rlist)

#
#  Define the station model data and plot.
#
imdat="11212833201001120000300004014752028601117706086792"
Ngl.wmsetp("wbs",0.25)    #  Scale
Ngl.wmstnm(wks,0.5,0.5,imdat)

Ngl.frame(wks)
Ngl.end()
