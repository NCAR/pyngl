#
#  File:
#    color6.py
#
#  Synopsis:
#    Draws colormaps (by name) as a vertical or horizontal bar of colors
#
#  Category:
#    Colors
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    April 2015
#
#  Description:
#    Loop through the official list of PyNGL color maps
#    and draws each one as a vertical or horizontal bar.
#
#    This example is similar to color1.py, except it
#    uses read_colormap_file to directly read the color map.
#    as an RGBA array. Note that you don't need to use 
#    define_colormap to first set the color map.   
#
#  Effects illustrated:
#    Changing color maps in a loop.
# 
#  Output:
#    By default, three color maps are drawn. See the code
#    below if you want to draw all the color maps.
#
#  Notes:
#    There are over 230 color maps.
#     
from __future__ import print_function
import Ngl, os

#----------------------------------------------------------------------
# Returns unique values
#----------------------------------------------------------------------

def remove_duplicates(slist):
  slist_uniq = []
  for s in slist:
    if s not in slist_uniq:
      slist_uniq.append(s)
  return(slist_uniq)

#----------------------------------------------------------------------
# This function retrieves all the color map names that are part of
# the official PyNGL distribution, without the leading directory
# path or the ".xxx" suffix.
#----------------------------------------------------------------------
def get_colormap_names():
  cmap_dir   = Ngl.pynglpath("colormaps")
  cmap_names = []
  for dirname, dirnames, filenames in os.walk(cmap_dir):
    # get just the filenames without the suffix or directory path
    for filename in filenames:
        cmap_names.append(filename.split(".")[0])
  # Some color maps appear twice for some reason (3gauss.ncmap and
  # 3gauss.rgb) so remove the duplicates before returning.
  return(remove_duplicates(cmap_names))

#----------------------------------------------------------------------
# This function draws the given color map as a vertical or
# horizontal labelbar.
#----------------------------------------------------------------------
def draw_colormap(wks,cmap):

#---Read as RGBA array
  cmapr = Ngl.read_colormap_file(cmap)    

#---Set some options
  orientation = "horizontal"       #  "horizontal" or "vertical"
  lbres = Ngl.Resources()

  if(orientation == "vertical"):
    width  = 0.2
    height = 1.0             # full height of viewport
    xpos   = 0.5-width/2.    # centered
    ypos   = 1.0             # flush w/top of viewport
  else:
    height = 0.2
    width  = 1.0             # full width of viewport
    xpos   = 0.0             # flush right
    ypos   = 0.5+height/2.   # centered

  lbres.vpWidthF          = width
  lbres.vpHeightF         = height
  lbres.lbOrientation     = orientation
  lbres.lbLabelsOn        = False
  lbres.lbBoxLinesOn      = False

#---Make sure labelbar fills the viewport region we specified
  lbres.lbBoxMinorExtentF = 1.0
  lbres.lbTopMarginF      = 0.0
  lbres.lbBottomMarginF   = 0.0
  lbres.lbRightMarginF    = 0.0
  lbres.lbLeftMarginF     = 0.0

#---Necessary so we get all solid fill
  lbres.lbMonoFillPattern = True    
  lbres.lbFillColors      = cmapr

  cmaplen = cmapr.shape[0]
  labels = ["" for x in range(cmaplen)]

  lbid = Ngl.labelbar_ndc (wks,cmaplen,labels,xpos,ypos,lbres)
  Ngl.draw(lbid)

#---Draw a box around the labelbar
  xbox = [xpos,xpos+width,xpos+width,xpos,xpos]
  ybox = [ypos,ypos,ypos-height,ypos-height,ypos]

  lnres = Ngl.Resources()
  lnres.gsLineThicknessF = 1.5
  Ngl.polyline_ndc(wks,xbox,ybox,lnres)


  lnres = Ngl.Resources()
  lnres.gsLineThicknessF = 1.5
  Ngl.polyline_ndc(wks,xbox,ybox,lnres)

#---Add name of the color map
  txres = Ngl.Resources()
  txres.txFontHeightF = 0.02
  if(orientation == "vertical"):
    txres.txJust = "TopRight"
    Ngl.text_ndc(wks,cmap,xpos,1.0,txres)
  else:
    txres.txJust = "BottomCenter"
    Ngl.text_ndc(wks,cmap,0.5,ypos,txres)

  Ngl.frame(wks)      # Advance the frame


#======================================================================
# Main code
#======================================================================

draw_all_colormaps = False

#---Open PNG file for graphics
wks_type = "png"
wks = Ngl.open_wks(wks_type,"color6")

cmaps = get_colormap_names()
print("There are {} color maps".format(len(cmaps)))

if draw_all_colormaps:
  num_colormaps = len(cmaps)
else:
  num_colormaps = 3

for n in range(num_colormaps):
  print("Drawing '{}'".format(cmaps[n]))
  draw_colormap(wks,cmaps[n])
    
Ngl.end()

