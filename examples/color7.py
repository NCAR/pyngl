#
#  File:
#    color7.py
#
#  Synopsis:
#    Draws all colors in user specified color maps using draw_color_palette.
#
#  Category:
#    Colors
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    July 2016
#
#  Description:
#    Loop through a list of color map names and
#    draw the maps as a table of colors and index 
#    values using a new function introduced in
#    PyNGL 1.5.0
#
#  Effects illustrated:
#    Changing color maps in a loop.
# 
#  Output:
#    By default, three color maps are drawn.
#
#  Notes:
#    If you decide to draw *all* color maps and are writing
#    to a PS, PDF, or an NCGM file, then those files will be 
#    quite large (approxmately 20MB for PS; 30MB for PDF; 
#    8 MB for NCGM).
#     

from __future__ import print_function
import Ngl
import os, sys

#----------------------------------------------------------------------
# This function returns a list of all the color maps in 
# this version of PyNGL.
#----------------------------------------------------------------------
def get_all_colormaps():
  pkgs_pth    = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
                            'site-packages')
  color_dir   = os.path.join(pkgs_pth,"PyNGL","ncarg","colormaps")
  color_files = os.listdir(color_dir)

  cmaps = []
  for cmap in color_files:
    cmaps.append(cmap.split('.')[:-1][0])
  return(set(cmaps))

#----------------------------------------------------------------------
# Main code.
#----------------------------------------------------------------------
wks_type = "png"
wks = Ngl.open_wks(wks_type,"color7") 

rlist = Ngl.Resources()

#
#  Specify a list of color map names, or uncomment the
# get_all_colormaps to draw all (200+) color maps.
#
# color_maps = get_all_colormaps()

color_maps = ["WhiteBlueGreenYellowRed","BkBlAqGrYeOrReViWh200","amwg"]

print("Will draw {} color maps".format(len(color_maps)))

for cmap in color_maps:
  print("Drawing color map '{}'...".format(cmap))
  Ngl.draw_color_palette(wks,cmap)

Ngl.end()
