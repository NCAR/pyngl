import Ngl
import os, sys

pkgs_pth    = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
                           'site-packages')
color_dir   = pkgs_pth + "/PyNGL/ncarg/colormaps"
color_files = os.listdir(color_dir)

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"color1") 

rlist = Ngl.Resources()

#
# Loop through all of the user defined color maps and draw each one
# as a table of colors and index values. If you just want to draw
# some of the color maps, you can list them out using color_files:
#
#   color_files = ["BlueRed","thelix","cosam"]
#
for i in xrange(len(color_files)):
  base_name = os.path.splitext(color_files[i])
  print base_name[0]

  rlist.wkColorMap = base_name[0] 
  Ngl.set_values(wks,rlist)

  Ngl.draw_colormap(wks)      # This call automatically advances the frame.

Ngl.end()

