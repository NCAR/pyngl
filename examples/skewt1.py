#
#   File:    skewt1.py
#
#   Author:  Fred Clare (based on an NCL example of Dennis Shea)
#            National Center for Atmospheric Research
#            PO 3007, Boulder, Colorado
#
#   Date:    Tue Mar  1 12:28:42 MST 2005
#
#   Description:     
#            Demonstrates three skew-T background plots: 
#              1. All defaults. 
#              2. Sets the two resources DrawColAreaFill and 
#                 DrawHeightScale to "True". 
#              3. Uses a centigrade scale [DrawFahrenheit = False] and 
#                 heights are indicated in meters [DrawHeightScale=True 
#                 and DrawHeightScaleFt=False ].
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
skewtOpts                  = Ngl.Resources()
skewtOpts.sktDrawColAreaFill  = True    # default is False
skewtOpts.sktDrawHeightScale  = True    # default is False
skewtOpts.tiMainString     = "USAF Skew T, Log(p); DOD-WPC 9-16-1"
 
skewt_bkgd     = Ngl.skewt_bkg(wks, skewtOpts)
Ngl.draw(skewt_bkgd)
Ngl.frame(wks)
del skewtOpts

#
#  Third plot - fancier background 
#
skewtOpts                  = Ngl.Resources()
skewtOpts.sktDrawColAreaFill  = True    # default is False
skewtOpts.sktDrawHeightScale  = True    # default is False
skewtOpts.sktDrawHeightScaleFt= False   # default is True
skewtOpts.sktDrawFahrenheit   = False   # default is True
skewtOpts.tiMainString     = "Centigrade - Meters"

skewt_bkgd = Ngl.skewt_bkg(wks, skewtOpts)
Ngl.draw(skewt_bkgd)
Ngl.frame(wks)
del skewtOpts

Ngl.end()
