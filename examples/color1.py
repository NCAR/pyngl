import Ngl
import os, sys

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"color1") 

rlist = Ngl.Resources()

#
# Loop through all of the listed color maps and draw them as a 
# table of colors and index values.
#
# Originally, this example drew *all* of the color maps in the colormaps
# directory. This resulted in a large PostScript file, so just do a
# subset instead. If you want to do all of them, uncomment the following
# code:
#
#pkgs_pth    = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
#                           'site-packages')
#color_dir   = pkgs_pth + "/PyNGL/ncarg/colormaps"
#color_files = os.listdir(color_dir)

color_files = ["thelix","GreenYellow","rainbow"]

for i in xrange(len(color_files)):
#
# Uncomment this code too, if you want to draw all colormaps,
# and comment out the other wkColorMap line below.
#
#base_name = os.path.splitext(color_files[i])
#print base_name[0]
#rlist.wkColorMap = base_name[0] 

  rlist.wkColorMap = color_files[i]

  Ngl.set_values(wks,rlist)

  Ngl.draw_colormap(wks)      # This call automatically advances the frame.

Ngl.end()

