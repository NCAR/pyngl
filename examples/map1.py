#
#  File:
#    map1.py
#
#  Synopsis:
#    Illustrates how to change a map's aspect ratio.
#
#  Categories:
#    maps only
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    February 2004
#
#  Description:
#    This example shows how to skew a map's aspect ratio.
#
#  Effects illustrated:
#    o  Using mpShapeMode to allow a free aspect ratio.
#    o  Changing the map projection.
# 
#  Output:
#    This example produces two visualizations:
#      1.) Cylindrical Equidistant projection
#      2.) Mollweide projection
#
#  Notes:
#     

import Ngl

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"map1")

projections = ["CylindricalEquidistant","Mollweide"]

mpres              = Ngl.Resources()   # Indicate you want to set some
                                       # resources.
mpres.mpShapeMode  = "FreeAspect"      # Allow the aspect ratio to be
                                       # changed.

mpres.vpWidthF     = 0.8               # Make the aspect ratio square.
mpres.vpHeightF    = 0.8

mpres.pmTitleDisplayMode = "Always"    # Force the title to appear.

for i in xrange(len(projections)):
  mpres.mpProjection = projections[i]
  mpres.tiMainString = projections[i]
  map = Ngl.map(wks,mpres)

Ngl.end()
