#
#  File:
#    skewt3.py
#
#  Synopsis:
#    Draws two skew-T plots using real data.
#
#  Category:
#    Skew-T
#
#  Author:
#    Fred Clare (based on an NCL example of Dennis Shea)
#  
#  Date of original publication:
#    March, 2005
#
#  Description:
#    Produces two skew-T plots using real data.
#
#  Effects illustrated:
#    o  Reading from an ASCII file.
#    o  Skew-T plots.
#    o  Thinning wind barbs on skew-T plots.
#    o  Changing the text function code flag.
# 
#  Output:
#    This example produces two visualizations:
#      1.)  Shows the full radiosonde.
#      2.)  "thins" the number of wind barbs plotted and
#           uses a Centigrade scale.  Setting "Wthin" to 3 
#           causes the plotting of every third wind barb.
#
#  Notes:
#    This example was updated in January 2006 to include the new
#    Skew-T resource names decided on.
#     

import Ngl, os
  
nlvl = 65
ncol = 7
TestData =  Ngl.asciiread(os.path.join(Ngl.pynglpath("data"),"asc","sounding_ATS.asc"), \
                          [nlvl,ncol], "float")

#
# Order: Surface is 'bottom'  eg: 1000,950,935,897,...  
#
p    = TestData[:,0]    # pressure     [mb / hPa] 
tc   = TestData[:,1]    # temperature  [C]    
tdc  = TestData[:,2]    # dew pt temp  [C]   
z    = TestData[:,4]    # geopotential [gpm] 
wspd = TestData[:,5]    # wind speed   [knots or m/s]    
wdir = TestData[:,6]    # meteorological wind dir   

wks_type = "ps"
wks = Ngl.open_wks(wks_type, "skewt3")

#
#  Plot1 - Create background skew-T and plot sounding.
#
skewtOpts                   = Ngl.Resources()
skewtOpts.sktColoredBandsOn = True                 # default is False
skewtOpts.tiMainString      = "ATS Rawindsonde: default dataOpts" 

skewt_bkgd = Ngl.skewt_bkg(wks, skewtOpts)
skewt_data = Ngl.skewt_plt(wks, skewt_bkgd, p, tc, tdc, z, \
                                  wspd,wdir)
Ngl.draw(skewt_bkgd)
Ngl.draw(skewt_data)
Ngl.frame(wks)

#
#  Plot 2 - thin the wind barbs and use Centigrade.
#
dataOpts                           = Ngl.Resources()
dataOpts.sktPressureWindBarbStride = 3           # Plot every 3rd wind barb

skewtOpts.sktTemperatureUnits      = "celsius"   # default is "fahrenheit"
skewtOpts.tiMainString             = "ATS Rawindsonde: degC + Thin wind" 

skewt_bkgd = Ngl.skewt_bkg(wks, skewtOpts)
skewt_data = Ngl.skewt_plt(wks, skewt_bkgd, p, tc, tdc, z,  \
                            wspd,wdir, dataOpts)
Ngl.draw (skewt_bkgd)
Ngl.draw (skewt_data)
Ngl.frame(wks)
Ngl.end()
