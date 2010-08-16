#
#  File:
#    color1.py
#
#  Synopsis:
#    Draws all colors in user specified color maps.
#
#  Category:
#    Colors
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    December, 2004
#
#  Description:
#    Loop through a list of color map names and
#    draw the maps as a table of colors and index 
#    values.  By uncommenting designated sections 
#    of the code all available color maps can be drawn.
#    Note that PyNGL internally sets the background and
#    foreground colors to white and black respectively. If
#    you want to change this, you will need to set the
#    resources wkBackgroundColor and wkForegroundColor when
#    you create the workstation.
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

import Ngl
import os, sys

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"color1") 

rlist = Ngl.Resources()

#
#  Specify a list of color map names.
#
color_files = ["thelix","GreenYellow","gsltod"]

#
#  If you want to draw *all* color maps, uncomment the following
#  lines (delete the initial "#" only in order to preserve proper
#  indentation).
#
#pkgs_pth    = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
#                          'site-packages')
#color_dir   = os.path.join(pkgs_pth,"PyNGL","ncarg","colormaps")
#color_files = os.listdir(color_dir)

for i in xrange(len(color_files)):
  rlist.wkColorMap = color_files[i]
#
#  If you want to draw all color maps uncomment the following
#  three lines (replace the initial "#" with a space in order to 
#  preserve proper indentation).
#
# base_name = os.path.splitext(color_files[i])
# print base_name[0]
# rlist.wkColorMap = base_name[0]

#
#  Set the color map and draw it.  The draw call automatically 
#  advances the frame.
#
  Ngl.set_values(wks,rlist)
  Ngl.draw_colormap(wks)

Ngl.end()
