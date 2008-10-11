#
#   File:
#     cn12p.py
#
#   Synopsis:
#     Draws a color filled contour map over Africa.
#
#   Category:
#     Contours over maps
#
#   Author:
#     Fred Clare (based on examples of Dave Kennison and Mary Haley).
#
#   Date of initial publication:    
#     September, 2004
#  
#   Description:
#     This example draws a map with filled contours appearing
#     only over Africa.  In order to mask Africa from the map fill,
#     the mpMaskAreaSpecifiers resource is used to mask all of
#     the countries in Africa.
#
#  Effects illustrated
#     o  Generating test data using generate_2d_array.
#     o  Drawing color filled contours over specified geographic areas.
#     o  Using a map labelbar instead of a contour labelbar
#
#  Output:
#     A single visualization is produced.
#
#  Notes: 
#     This example requires the resource file cn12p.res.
#
#     Resource files should be used carefully, because you want to
#     make sure the resources apply to the correct objects. Resources
#     set in a script override any resources set in a resource file.
#
#     The resources for this example are split between a resource 
#     file and this script. It is mostly by choice which resources
#     appear where.
# 
#     You cannot set ngl* resources in a resource file.
#
#  Import numpy
#
import numpy

#
#  Import Ngl support functions.
#
import Ngl

#
#  These are the geographical areas we want to fill.
#
mp_fill_specs = ["water","land"]
mp_fill_colrs = [7,2]              # blue, gray

#
#  These are the geographical areas we want to mask.
#
mask_specs =                                                                 \
 ["algeria","angola","angola-exclave-called-cabinda","benin","botswana",     \
  "burundi","cameroon","central-african-republic","chad","congo","djibouti", \
  "egypt","equatorial-guinea","ethiopia","gabon","gambia","ghana","guinea",  \
  "guinea-bissau","ivory-coast","kenya","lesotho","liberia","libya",         \
  "madagascar","malawi","mali","mauritania","mauritius","morocco",           \
  "mozambique","namibia","niger","nigeria","rwanda","senegal","sierra-leone",\
  "somalia","south-africa","sudan","swaziland","tanzania","togo","tunisia",  \
  "uganda","upper-volta","western-sahara","zaire","zambia","zimbabwe"]

cn_fill_clrs = [3,4,5,6,8,9,10,11,12,13,14,15]

#
# Generate color map using RGB values.
#
cmap = numpy.array([[1.00,1.00,1.00],[0.00,0.00,0.00],[0.70,0.70,0.70], \
                    [0.75,0.50,1.00],[0.50,0.00,1.00],[0.00,0.00,1.00], \
                    [0.00,0.50,1.00],[0.00,1.00,1.00],[0.00,1.00,0.60], \
                    [0.00,1.00,0.00],[0.70,1.00,0.00],[1.00,1.00,0.00], \
                    [1.00,0.75,0.00],[1.00,0.38,0.38],[1.00,0.00,0.38], \
                    [1.00,0.00,0.00]])
#
#  Open a workstation.
#
wres            = Ngl.Resources()
wres.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"cn12p",wres)

# Generate some dummy data
dirc = Ngl.pynglpath("data")
z    = Ngl.generate_2d_array ([40,40],15,15,-10.,110.)
 
# Create contour and map resource lists.
cnres = Ngl.Resources()
mpres = Ngl.Resources()
cnres.nglDraw  = False
cnres.nglFrame = False
mpres.nglDraw  = False
mpres.nglFrame = False

#
# Map fill resources.
#
# To fill map areas, you can first specify a general set
# of areas (mpFillBoundarySets) and then a specific set of
# areas. This allows you to indicate what general areas you always
# want filled, and then what specific areas on top of that you
# also want filled.
#
# Here, no general areas are being specified, but the specific
# areas of "water" and "land" are. This is because we can then
# use the mpMaskAreaSpecifiers resource to indicate which of
# the land and/or water areas you don't want to fill.
#
mpres.mpFillOn              = True               # Turn on map fill
mpres.mpFillBoundarySets    = "NoBoundaries"   
mpres.mpFillAreaSpecifiers  = mp_fill_specs      # water, land
mpres.mpSpecifiedFillColors = mp_fill_colrs      #  blue, gray

# The mpMaskAreaSpecifiers array allows you to specify which areas are
# not to be filled in the areas given by mpFillAreaSpecifiers.
mpres.mpAreaMaskingOn       = True
mpres.mpMaskAreaSpecifiers  =  mask_specs      # areas to mask (protect)

#
# This set of resources deals with a map labelbar.
#
# Note that this is different from a contour labelbar
# (which we will turn off below), and we have more 
# control over it (i.e. we can set the number of boxes,
# the fill # colors, etc) than we do with a contour labelbar.
#
mpres.pmLabelBarDisplayMode =  "Always"        # Turn on a map labelbar

# Labelbar resources.
mpres.lbLabelStrings        = ["Ocean","Land","< 0","0-10","10-20","20-30",\
                               "30-40","40-50","50-60","60-70","70-80",    \
                               "80-90","90-100","> 100"]
mpres.lbFillColors          = mp_fill_colrs + cn_fill_clrs

# Fill resources
cnres.cnFillColors          =  cn_fill_clrs
cnres.pmLabelBarDisplayMode = "Never"  # turn off, b/c map has one turned on

contour = Ngl.contour(wks,z[:,:],cnres)
map     = Ngl.map(wks,mpres)

Ngl.overlay(map,contour)
Ngl.draw(map)
Ngl.frame(wks)

Ngl.end()
