#
#  File:
#    shapefile3.py
#
#  Synopsis:
#    Example of how to mask contour data over a map using a lat,lon area.
#
#  Categories:
#    Maps only
#    Polylines
#    Polymarkers
#
#  Author:
#    Mary Haley (based on an NCL script)
#  
#  Date of initial publication:
#    July 2011
#
#  Description:
#
#  Effects illustrated:
#      o Plotting data from a shapefile
#      o Drawing selected data based upon a database query of the shapefile
#      o Using Ngl.gc_inout to mask contours that fall outside a polygon
#      o Drawing masked lat/lon grid
# 
#  Output:
#     Three visualizations are produced showing data over the 
#     Mississippi River Basin.
#
#  Notes:
#     You must download the "mrb.xxx" files from:
#
#          http://www.ncl.ucar.edu/Applications/Data/#shp
#
import numpy,os,sys
from numpy import ma

#---Import Ngl,Nio support functions.
import Ngl, Nio


#----------------------------------------------------------------------
# This function adds the given lat/lon lines to a map
#----------------------------------------------------------------------
def add_mrb_lines(wks,plot,lat,lon):
#---Resources for polyline
  lnres                   = Ngl.Resources()
  lnres.gsLineColor       = "black"
  lnres.gsLineThicknessF  = 3.0            # 3x as thick

  return Ngl.add_polyline(wks, plot, lon, lat, lnres)

#----------------------------------------------------------------------
# This function adds lat/lon grid points outside a given masked area
#----------------------------------------------------------------------
def add_masked_grid(wks,plot,lat,lon,mask):
#
# Create 2D arrays of lat/lon so we can draw markers outside the 
# masked area
#
  nlat  = len(lat)
  nlon  = len(lon)
  lat2d = numpy.tile(lat,nlon)
  lat2d = numpy.reshape(lat2d,[nlon,nlat])
  lat2d = numpy.transpose(lat2d)
  lon2d = numpy.tile(lon,nlat)
  lon2d = numpy.reshape(lon2d,[nlat,nlon])

  lat1d_mask = ma.ravel(ma.masked_where(mask==0,lat2d))
  lon1d_mask = ma.ravel(ma.masked_where(mask==0,lon2d))

  mkres               = Ngl.Resources()
  mkres.gsMarkerIndex = 16          # filled dot
  mkres.gsMarkerSizeF = 0.003
  mkres.gsMarkerColor = "purple"

  return Ngl.add_polymarker(wks,plot,lon1d_mask,lat1d_mask,mkres)

#----------------------------------------------------------------------
# Function to generate some geographical dummy data
#----------------------------------------------------------------------
def gen_dummy_data():
  minlat =   24
  maxlat =   50
  minlon = -125
  maxlon =  -67

#---Size of dummy data. Use smaller values (16x32) for faster code speed
  nlat   =  64
  nlon   = 128

  data   = Ngl.generate_2d_array([nlat,nlon], 10, 19, 0., 100.)
  lat    = Ngl.fspan(minlat,maxlat,nlat)
  lon    = Ngl.fspan(minlon,maxlon,nlon)

  return data,lat,lon

#----------------------------------------------------------------------
# Main code
#----------------------------------------------------------------------

#---Generate dummy data
data,lat1d,lon1d = gen_dummy_data()

#---Get nice contour levels
cmin,cmax,cint = Ngl.nice_cntr_levels(numpy.min(data),numpy.max(data))

#---Start the graphics
wks_type = "ps"
wks = Ngl.open_wks (wks_type,"shapefile3")

#---Define some graphical resources
res = Ngl.Resources()

res.nglFrame              = False
res.nglDraw               = False

#---Scalar field resources
res.sfXArray              = lon1d
res.sfYArray              = lat1d

#---Contour resources
res.cnFillOn              = True
res.cnLinesOn             = False
res.cnLineLabelsOn        = False

#---Contour levels
res.cnLevelSelectionMode  = "ManualLevels"
res.cnMinLevelValF        = cmin
res.cnMaxLevelValF        = cmax
res.cnLevelSpacingF       = cint

#---Labelbar resources
res.lbOrientation         = "Horizontal"
res.lbLabelFontHeightF    = 0.015
res.pmLabelBarWidthF      = 0.5
res.pmLabelBarHeightF     = 0.2

#---Title resources
res.tiMainString          = "Contours of dummy data"
res.tiMainFontHeightF     = 0.015

#---Map resources
res.mpDataBaseVersion     = "MediumRes"     # slightly better resolution
res.mpOutlineBoundarySets = "USStates"
res.mpGridAndLimbOn       = False           # Turn off lat/lon grid
res.pmTickMarkDisplayMode = "Never"         # Turn off map tickmarks

#---Zoom in on North America.
res.mpLimitMode           = "LatLon"
res.mpMinLatF             = min(lat1d)
res.mpMaxLatF             = max(lat1d)
res.mpMinLonF             = min(lon1d)
res.mpMaxLonF             = max(lon1d)

#---Create contours over map.
map = Ngl.contour_map(wks,data,res)

#---Open shapefile with Mississippi River Basin outlines
dir      = Ngl.pynglpath("data")
filename = os.path.join(dir,"shp","mrb.shp")
if(not os.path.exists(filename)):
  print "You do not have the necessary files to run this example."
  print "The comments at the top of this script tell you how to get the files."
  sys.exit()

f = Nio.open_file(filename, "r")

#---Read data off shapefile
mrb_lon = f.variables["x"][:]
mrb_lat = f.variables["y"][:]

#---Add MRB outlines to map
lines = add_mrb_lines(wks,map,mrb_lat,mrb_lon)

#---Draw plot and advance frame. MRB outlines will be included.
Ngl.draw(map)
Ngl.frame(wks)

#
# Get the approximate index values that contain the area
# we don't want to mask.
#
# This will make our gc_inout loop below go faster, because we're
# not checking every single lat/lon point to see if it's inside
# or outside the area of interest.
#
min_mrb_lat = min(mrb_lat)
max_mrb_lat = max(mrb_lat)
min_mrb_lon = min(mrb_lon)
max_mrb_lon = max(mrb_lon)

ilt_min = numpy.greater_equal(lat1d,min_mrb_lat).tolist().index(True)
ilt_max = numpy.less_equal(lat1d,max_mrb_lat).tolist().index(False)-1
iln_min = numpy.greater_equal(lon1d,min_mrb_lon).tolist().index(True)
iln_max = numpy.less_equal(lon1d,max_mrb_lon).tolist().index(False)-1

#
# Loop through area of interest, and create a mask for values
# inside the MRB area.
#
mask = numpy.ones(data.shape) 
for ilt in range(ilt_min,ilt_max+1):
  for iln in range(iln_min,iln_max+1):
    if(Ngl.gc_inout(lat1d[ilt],lon1d[iln],mrb_lat,mrb_lon) != 0):
      mask[ilt][iln] = 0

#---Mask original data array against mask array
data_mask = ma.array(data,mask=mask)

#---Change the main title
res.tiMainString = "Contours masked by Mississippi River Basin"

#---Create contours of masked array and add MRB outline
map   = Ngl.contour_map(wks,data_mask,res)
lines = add_mrb_lines(wks,map,mrb_lat,mrb_lon)

#---Draw masked contours and advance frame
Ngl.draw(map)
Ngl.frame(wks)

#
# Last plot will be further zoomed and all map outlines 
# (except for MRB) turned off. Markers will be drawn at
# the  masked grid locations.
#
res.mpOutlineOn = False
res.mpMinLatF   = min_mrb_lat-1
res.mpMaxLatF   = max_mrb_lat+1
res.mpMinLonF   = min_mrb_lon-1
res.mpMaxLonF   = max_mrb_lon+1

map   = Ngl.contour_map(wks,data_mask,res)
lines = add_mrb_lines(wks,map,mrb_lat,mrb_lon)

#---Add markers in grid outside of masked area
grid = add_masked_grid(wks,map,lat1d,lon1d,mask)

#---Draw masked contours and grid and advance frame
Ngl.draw(map)
Ngl.frame(wks)

Ngl.end()
