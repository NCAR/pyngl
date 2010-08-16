#
#  File:
#    conmasklc.py
#
#  Synopsis:
#    Draws contours over a masked lambert conformal map.
#
#  Category:
#    Contouring over maps.
#
#  Author:
#    Mary Haley (based on a code by Fred Clare)
#  
#  Date of initial publication:
#    December, 2009
#
#  Description:
#    This example produces two frames:
#    1) A masked Lambert Conformal map
#    2) Contours over a masked Lambert Conformal map
#
#  Effects illustrated:
#    o  Reading in a NetCDF file using Nio.
#    o  Masking a Lambert conformal map projection.
#    o  Controlling the number and spacing of contour levels.
#    o  Changing font height and color.
#    o  How to explicitly define contour levels.
# 
#  Output:
#    This examples produces contours over a masked lambert conformal map.
#
#  Notes:
#     

#
#  Import NumPy.
#
import numpy, os

#
#  Import Nio for a NetCDF reader and Ngl for plotting.
#
import Nio, Ngl

#
# Read some variables off the file.
#
dirc = Ngl.pynglpath("data")
nf   = Nio.open_file(os.path.join(dirc,"cdf","meccatemp.cdf"))
T    = nf.variables["t"][:]
lat  = nf.variables["lat"][:]
lon  = nf.variables["lon"][:]

#
# Set up a color map using RGB triplets.
#
cmap =  numpy.array(\
        [ [.000,.000,.000], [1.00,1.00,1.00], [.700,.700,.700], \
          [.650,.650,.700], [.610,.600,.700], [.550,.550,.700], \
          [.560,.500,.700], [.450,.450,.700], [.420,.400,.700], \
          [.350,.350,.700], [.300,.300,.700], [.250,.250,.700], \
          [.200,.200,.700], [.150,.150,.700], [.100,.100,.700], \
          [.050,.050,.700], [.000,.000,.700], [.000,.050,.700], \
          [.000,.100,.700], [.000,.150,.700], [.000,.200,.700], \
          [.000,.250,.700], [.000,.300,.700], [.000,.350,.700], \
          [.000,.400,.700], [.000,.450,.600], [.000,.500,.500], \
          [.000,.550,.400], [.000,.600,.300], [.000,.650,.200], \
          [.000,.700,.100], [.000,.725,.000], [.000,.690,.000], \
          [.030,.685,.000], [.060,.680,.000], [.100,.575,.000], \
          [.130,.570,.000], [.160,.565,.000], [.550,.550,.000], \
          [.555,.545,.000], [.560,.530,.000], [.565,.485,.000], \
          [.570,.420,.000], [.675,.375,.000], [.680,.330,.000], \
          [.690,.300,.000], [.700,.285,.000], [.700,.270,.000], \
          [.700,.260,.000], [.700,.240,.000], [.700,.180,.000], \
          [.700,.130,.000], [.700,.120,.000], [.700,.100,.000], \
          [.700,.090,.000], [.750,.090,.000], [.800,.090,.000], \
          [.830,.070,.000], [.870,.050,.000], [.900,.030,.000], \
          [.950,.010,.000], [.990,.000,.000], [1.00,.000,.000], \
          [1.00,.000,.000] ])


#
#  Open a workstation.
#

wks_type = "ps"
wres = Ngl.Resources()
wres.wkColorMap = cmap
wks = Ngl.open_wks(wks_type,"conmasklc",wres)

#----------- Begin first plot -----------------------------------------
 
res = Ngl.Resources()

#
# Set map resources.
#
res.mpProjection        = "LambertConformal"

res.mpLimitMode         = "LatLon"     # limit map via lat/lon
res.mpMinLatF           =  10.         # map area
res.mpMaxLatF           =  75.         # latitudes
res.mpMinLonF           =  60.         # and
res.mpMaxLonF           = 165.         # longitudes

res.nglMaskLambertConformal = True
#res.nglMaskLambertConformalOutlineOn = False

res.tiMainString         = "A masked Lambert Conformal map"
res.tiMainFontHeightF    = 0.010

map = Ngl.map(wks,res)

#
# Set some scalar field resources.
#
res.sfXArray = lon
res.sfYArray = lat

#
# Set some contour resources
#
res.cnFillOn             = True
res.cnLinesOn            = False
res.cnLineLabelsOn       = False

res.cnLevelSelectionMode = "ManualLevels"
res.cnMinLevelValF       = 195.
res.cnMaxLevelValF       = 328.
res.cnLevelSpacingF      =   5.

#
# Set title resources.
#
res.tiMainString         = "January Global Surface Temperature (K)"

#
# Set labelbar resources.
#
res.lbOrientation        = "Horizontal"

nt = 30    # Pick a time for a single plot.
res.lbTitleString = "Day " + str(nt+1)
map = Ngl.contour_map(wks,T[nt,:,:],res)

Ngl.end()
