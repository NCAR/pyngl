#
#   File:    skewt2.py
#
#   Author:  Fred Clare (based on an NCL example of Dennis Shea)
#            National Center for Atmospheric Research
#            PO 3007, Boulder, Colorado
#
#   Date:    Tue Mar  1 12:38:22 MST 2005
# 
#   Description:
#            Produces two skew-T plots using real data.  The winds from 
#            a (bogus) pibal are drawn using a different colors.
#    
import Ngl,Numeric

nlvl = 30  
ncol = 16
TestData = Ngl.asciiread(Ngl.ncargpath("data") + "/asc/sounding_testdata.asc", \
                   [nlvl,ncol], "float")

p    = TestData[:,1]
z    = TestData[:,2]
tc   = TestData[:,5] + 2.    # for demo purposes
tdc  = TestData[:,9]

#
#  Set winds to missing values so that they will not be plotted.
#
wspd = -999.*Numeric.ones(nlvl,Numeric.Float0)
wdir = -999.*Numeric.ones(nlvl,Numeric.Float0)

#
#  Plot 1 - Create background skew-T and plot sounding.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type, "skewt2")

skewtOpts                 = Ngl.Resources()
skewtOpts.sktWSPDmissingV    = -999.     # Missing value for wind speed.
skewtOpts.sktWDIRmissingV    = -999.     # Missing value for wind direction.
skewtOpts.sktDrawColAreaFill = True      # Default is False
skewtOpts.tiMainString    = "Raob Data; No Winds" 

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
hght = Numeric.array([1500., 6000., 10000., 15000.], Numeric.Float0) # Meters
hspd = Numeric.array([  50.,   27.,   123.,    13.], Numeric.Float0)
hdir = Numeric.array([ 315.,  225.,    45.,   135.], Numeric.Float0)

dataOpts           = Ngl.Resources()  # Options describing data and plotting.
dataOpts.sktPlotWindH = True          # Plot wind barbs at height levels.
dataOpts.sktHspdHdir  = True          # Wind speed and dir [else: u,v].

dataOpts.sktHeight    = hght          # height of wind reports
dataOpts.sktHspd      = hspd          # speed [or u component]
dataOpts.sktHdir      = hdir          # dir   [or v component]

skewtOpts                 = Ngl.Resources()
skewtOpts.sktDrawHeightScale    = True
skewtOpts.sktDrawHeightScaleFt    = True
skewtOpts.tiMainString    = "Raob; [Wind Reports]"
skewtOpts.sktDrawColAreaFill = True    # default is False

skewtOpts.sktcolWindZ = "Red"
skewt_bkgd = Ngl.skewt_bkg(wks, skewtOpts)
skewt_data = Ngl.skewt_plt(wks, skewt_bkgd, p, tc, tdc, z, \
                                wspd, wdir, dataOpts)
Ngl.draw(skewt_bkgd)
Ngl.draw(skewt_data)
Ngl.frame(wks)

Ngl.end()
