#
#  File:
#    colors_draw_colormap_from_file.py
#
#  Synopsis:
#    Demonstrates how to draw a color map (user defined) from file.
#
#  Category:
#    colors
#
#  Based on NCL example:
#    -
#
#  Author:
#    Karin Meier-Fleischer
#  
#  Date of initial publication:
#    November, 2018
#
#  Description:
#    This example shows how to draw a color map (user defined) from file.
#
#  Effects illustrated:
#    o  Drawing a color map
#
#  Output:
#     A single visualization is produced.     
#
'''
  PyNGL Example: 	colors_draw_colormap_from_file.py

  -  Drawing a color map
  
'''
from __future__ import print_function
import Ngl

wks = Ngl.open_wks("x11","plot_user_defined_colormap")

Ngl.draw_color_palette(wks,"./ccc-tool_colormap.rgb")

Ngl.end()
