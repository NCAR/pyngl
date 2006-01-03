#
#  File:
#    skewt1.py
#
#  Synopsis:
#    Draws three skew-T background plots.
#
#  Category:
#    Skew-T
#
#  Author:
#    Fred Clare (based on a code of Dennis Shea).
#  
#  Date of original publication:
#    March, 2005
#
#  Description:
#    This example draws three skew-T background plots, first 
#    using all defaults and then setting some skew-T resources.
#
#  Effects illustrated:
#    o  Drawing skew-T backgrounds.
#    o  Using skew-T resources.
# 
#  Output:
#    This example produces three visualizations:
#      1.)  A skew-T background using all defaults.
#      2.)  A skew-T background with a height scale and background color.
#      3.)  A skew-T background with a main title.
#
#  Notes:
#    This example was updated in January 2006 to include the new
#    Skew-T resource names decided on.
#     

import Ngl

wks_type = "ps"
wks = Ngl.open_wks (wks_type, "skewt1")

#
#  First plot - default background
#
skewtOpts              = Ngl.Resources()
skewtOpts.tiMainString = "Default Skew-T"
skewt_bkgd = Ngl.skewt_bkg(wks, skewtOpts)

Ngl.draw(skewt_bkgd)
Ngl.frame(wks)
del skewtOpts

#
#  Second plot - fancier background 
#
skewtOpts                   = Ngl.Resources()
skewtOpts.sktColoredBandsOn = True    # default is False
skewtOpts.sktHeightScaleOn  = True    # default is False
skewtOpts.tiMainString      = "USAF Skew T, Log(p); DOD-WPC 9-16-1"
 
skewt_bkgd     = Ngl.skewt_bkg(wks, skewtOpts)
Ngl.draw(skewt_bkgd)
Ngl.frame(wks)
del skewtOpts

#
#  Third plot - fancier background 
#
skewtOpts                     = Ngl.Resources()
skewtOpts.sktColoredBandsOn   = True    # default is False
skewtOpts.sktHeightScaleOn    = True    # default is False
skewtOpts.sktHeightScaleUnits = "km"    # default is "feet"
skewtOpts.sktTemperatureUnits = "celsius"   # default is "fahrenheit"
skewtOpts.tiMainString        = "Centigrade - Meters"

skewt_bkgd = Ngl.skewt_bkg(wks, skewtOpts)
Ngl.draw(skewt_bkgd)
Ngl.frame(wks)
del skewtOpts

Ngl.end()
