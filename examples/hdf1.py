#
#  File:
#    hdf1.py
#
#  Synopsis:
#    Plots EOS-DIS data
#
#  Category:
#    Contours over maps
#    Labelbar
#    Maps
#
#  Author:
#    Mary Haley (based on NCL example from Dennis Shea)
#  
#  Date of initial publication:
#    July, 2011
#
#  Description:
#    This example uses Ngl.dim_gbits to extra bit information from
#    cloud mask data.
#
#  Effects illustrated:
#    o  Reading from a NetCDF file.
#    o  Setting map resources.
#    o  Setting labelbar resources.
#    o  Contouring in triangular mesh mode.
#
#  Output:
#    A single visualization is produced showing cloud mask data
#    over a satellite projection.
#    
#
#******************************************************************
#    Bit fields within each byte are numbered from the left: 
#    7, 6, 5, 4, 3, 2, 1, 0.                                                 
#    The left-most bit (bit 7) is the most significant bit.                  
#    The right-most bit (bit 0) is the least significant bit.                
#                                                                            
#    bit field    Description                        Key                     
#    ---------    -----------                        ---                     
#    2, 1         Unobstructed FOV Quality Flag    00 (0) = Cloudy           
#                                                  01 (1) = Uncertain        
#                                                  10 (2) = Probably  Clear  
#                                                  11 (3) = Clear          

#---Import support modules
import numpy, os, sys, Nio, Ngl

#---Test if file exists
flnm  = "MOD05_L2.A2002161.1830.004.2003221153410.hdf"
if(not os.path.exists(flnm)):
  print "You do not have the necessary HDF file to run this example."
  print "You can get the file from"
  print "    http://www.ncl.ucar.edu/Applications/Data/#hdf"
  sys.exit()

#---Open netCDF file
hfile = Nio.open_file(flnm)

#---Read lat/lon [5km] and Cloud_Mask [1km]
cldmsk = hfile.variables["Cloud_Mask_QA"][:]
lat2d  = hfile.variables["Latitude"][:]
lon2d  = hfile.variables["Longitude"][:]
dimll  = lat2d.shape
nlat   = dimll[0]                  # "along"  swath  [5 km]
nlon   = dimll[1]                  # "across" swath  [5 km]

#
# Latitude/Longitude arrays are every 5km. Cloud_Mask is every 1km
# It is necessary to decimate the Cloud_Mask to match lat/lon
#
dimcm  = cldmsk.shape
NY     = dimcm[0]                  # "along"  swath  [1 km]
MX     = dimcm[1]                  # "across" swath  [1 km]
cm     = cldmsk[4::5,4::5]         # decimate to match lat2d/lon2d dimsizes
del cldmsk                         # no longer needed

#
# Extract the desired bits and attach lat/lon arrays
# cmfov = 00 [0] cloudy            01 [1] uncertain 
#         10 [2] probably clear    00 [3] confidently clear
#
tmp    = Ngl.dim_gbits(numpy.ravel(cm),5,2,6,nlat*nlon)
cmfov  = numpy.reshape(tmp, [nlat,nlon])
del tmp
del cm

#---Set the colormap and open a PostScript workstation.
colors   = ["white","black","gray","red","green","white"]
wks_type = "ps"
wklist = Ngl.Resources()
wklist.wkColorMap = colors
wks = Ngl.open_wks(wks_type,"hdf1",wklist)

#---Create variable to hold list of plot resources.
res = Ngl.Resources()

res.nglDraw              = False         # Don't create plot or
res.nglFrame             = False         # advance frame yet.

res.cnFillOn             = True          # turn on color fill
res.cnLinesOn            = False         # turn off contour lines
res.cnLineLabelsOn       = False         # turn of contour labels
res.cnFillMode           = "RasterFill"  # turn on raster mode

res.pmLabelBarDisplayMode = "Never"           # Turn off labelbar

res.cnLevelSelectionMode = "ManualLevels"     # set manual contour levels
res.cnMinLevelValF       =   1                # set min contour level
res.cnMaxLevelValF       =   3                # one less than max
res.cnLevelSpacingF      =   1                # set contour spacing

res.mpDataBaseVersion    = "MediumRes"
res.mpOutlineBoundarySets= "National"

res.mpProjection         = "Satellite"        # choose map projection
res.mpFillOn             = False              # turn off map fill
res.mpGridAndLimbOn      = False              # turn off lat/lon lines
res.mpCenterLonF         = (numpy.min(lon2d)+numpy.max(lon2d))*0.5
res.mpCenterLatF         = (numpy.min(lat2d)+numpy.max(lat2d))*0.5

res.mpLimitMode          = "LatLon"
res.mpMinLatF            = numpy.min(lat2d)-1       # min lat
res.mpMaxLatF            = numpy.max(lat2d)+1       # max lat
res.mpMinLonF            = numpy.min(lon2d)-1       # min lon
res.mpMaxLonF            = numpy.max(lon2d)+1       # max lon

res.tiMainString         = "Unobstructed FOV Quality Flag"

res.trGridType           = "TriangularMesh"   # faster graphic rendering

res.sfXArray             = lon2d              # 5 km 2D lat/lon arrays
res.sfYArray             = lat2d

#---Create plot (it won't get drawn yet)
plot = Ngl.contour_map(wks,cmfov,res)

#---Resources for creating a labelbar
lbres = Ngl.Resources()

lbres.nglDraw              = False
lbres.lbPerimOn            = False            # turn off perimeter.

lbres.vpWidthF             = 0.60             # width and height
lbres.vpHeightF            = 0.15

lbres.lbOrientation        = "Horizontal"
lbres.lbLabelPosition      = "Center"         # label position
lbres.lbFillColors         = colors[2:]
lbres.lbMonoFillPattern    = True             # solid color fill
lbres.lbLabelFontHeightF   = 0.02

lbres.lbTitleOn            = True
lbres.lbTitleString        = "0=cldy, 1=uncertain, 2=prob clr, 3=clr"
lbres.lbTitlePosition      = "Bottom"
lbres.lbTitleFontHeightF   = 0.02
lbres.lbTitleOffsetF       = 0.00

#---Create the labelbar
labels  = ["0","1","2","3"]
lbid = Ngl.labelbar_ndc (wks,len(labels),labels,0.0,0.,lbres)  

#
# Create some annotation resources indicating how we want to
# attach the labelbar to the plot. The default is the center
# of the plot. Below amOrthogonalPosF is set to move the
# labelbar down and outside the plot.
#                                                                 
amres                  = Ngl.Resources()
amres.amOrthogonalPosF =  0.7
annoid = Ngl.add_annotation(plot,lbid,amres)

txres = Ngl.Resources()
txres.txFontHeightF = 0.01
txres.txJust        = "TopRight"
txres.txPerimOn     = True
txid  = Ngl.add_text(wks,plot,flnm,res.mpMaxLonF,res.mpMaxLatF,txres)

#---This will resize plot so it and the labelbar fit in the frame.
Ngl.maximize_plot(wks, plot)

#---Drawing the original map also draws the attached labelbar.
Ngl.draw(plot)
Ngl.frame(wks)

Ngl.end()
