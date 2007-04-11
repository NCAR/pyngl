#
#  File:
#    skewt2.py
#
#  Synopsis:
#    Draws skew-T visualizations using dummy data.
#
#  Category:
#    Skew-T
#
#  Author:
#   Author:  Fred Clare (based on an NCL example of Dennis Shea)
#  
#  Date of original publication:
#    March, 2005
#
#  Description:
#    This example draws two skew-T plots using real data.  The 
#    winds from a (bogus) pibal are drawn using different colors.
#
#  Effects illustrated:
#    o  Reading from an ASCII file.
#    o  Flagging missing values.
#    o  Plotting soundings and winds.
#    o  Plotting wind barbs at height levels.
# 
#  Output:
#    This example produce two visualizations:
#      1.)  A Raob sounding with no winds.
#      2,)  A Raob sounding with wind barbs at height levels and a
#           height scale in feet.
#
#  Notes:
#    This example was updated in January 2006 to include the new
#    Skew-T resource names decided on.
#     

import Ngl
import numpy

nlvl = 30  
ncol = 16
TestData = Ngl.asciiread(Ngl.pynglpath("data") + "/asc/sounding_testdata.asc", \
                   [nlvl,ncol], "float")

p    = TestData[:,1]
z    = TestData[:,2]
tc   = TestData[:,5] + 2.    # for demo purposes
tdc  = TestData[:,9]

#
#  Set winds to missing values so that they will not be plotted.
#
wspd = -999.*numpy.ones(nlvl,'f')
wdir = -999.*numpy.ones(nlvl,'f')

#
#  Plot 1 - Create background skew-T and plot sounding.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type, "skewt2")

skewtOpts                          = Ngl.Resources()
skewtOpts.sktWindSpeedMissingV     = -999.     # Missing value for 
                                               # wind speed.
skewtOpts.sktWindDirectionMissingV = -999.     # Missing value for 
                                               # wind direction.
skewtOpts.sktColoredBandsOn        = True      # Default is False
skewtOpts.tiMainString             = "Raob Data; No Winds" 

skewt_bkgd = Ngl.skewt_bkg(wks, skewtOpts)
skewt_data = Ngl.skewt_plt(wks, skewt_bkgd, p, tc, tdc, z,  \
                                wspd, wdir, skewtOpts)
Ngl.draw(skewt_bkgd)
Ngl.draw(skewt_data)
Ngl.frame(wks)

#
#  Plot 2 - Create background skew-T and plot sounding and winds.
#
wspd = Ngl.fspan(0., 150., nlvl)   # wind speed at each level.
wdir = Ngl.fspan(0., 360., nlvl)   # wind direction.

#
#  Create a few artificial "pibal" reports.
#
hght = numpy.array([1500., 6000., 10000., 15000.], 'f') # Meters
hspd = numpy.array([  50.,   27.,   123.,    13.], 'f')
hdir = numpy.array([ 315.,  225.,    45.,   135.], 'f')

dataOpts                               = Ngl.Resources()  # Options describing 
                                                          # data and plotting.
dataOpts.sktHeightWindBarbsOn          = True             # Plot wind barbs at
                                                          # height levels.
dataOpts.sktPressureWindBarbComponents = "SpeedDirection" # Wind speed and 
                                                          # dir [else: u,v].

dataOpts.sktHeightWindBarbPositions  = hght        # height of wind reports
dataOpts.sktHeightWindBarbSpeeds     = hspd        # speed
                                                   # [or u components]
dataOpts.sktHeightWindBarbDirections = hdir        # direction
                                                   # [or v components]

skewtOpts                              = Ngl.Resources()
skewtOpts.sktHeightScaleOn             = True      # default is False
skewtOpts.sktHeightScaleUnits          = "feet"    # default is "feet"
skewtOpts.sktColoredBandsOn            = True      # default is False
skewtOpts.sktGeopotentialWindBarbColor = "Red"
skewtOpts.tiMainString                 = "Raob; [Wind Reports]"

skewt_bkgd = Ngl.skewt_bkg(wks, skewtOpts)
skewt_data = Ngl.skewt_plt(wks, skewt_bkgd, p, tc, tdc, z, \
                                wspd, wdir, dataOpts)
Ngl.draw(skewt_bkgd)
Ngl.draw(skewt_data)
Ngl.frame(wks)

Ngl.end()
