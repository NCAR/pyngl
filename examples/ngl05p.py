#
#  File:
#    ngl05p.py
#
#  Synopsis:
#    Draws contours over maps.
#
#  Category:
#    Contouring over maps.
#
#  Author:
#    Fred Clare (based on a code by Mary Haley).
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    This example draws contours over maps of varying degrees
#    of complexity illustrating several map projections.
#
#  Effects illustrated:
#    o  Reading in several NetCDF files using Nio.
#    o  Using several map projections: cylindrical equidistant,
#       orthographic, Lambert equal area.
#    o  Contouring only over land.
#    o  Controlling the number and spacing of contour levels.
#    o  Changing font height and color.
#    o  How to explicitly define contour levels.
# 
#  Output:
#    This examples produces four visualizations:
#      1.)  A simple contour over a map using the default map
#           projection (cylindrical equidistant).
#      2.)  Contouring over the oceans using an orthographic
#           projection and a fill pattern while coloring the 
#           land masses with a single color.
#      3.)  Color contours over a specified map area using a
#           Lambert conformal projection.
#      4.)  Contouring over the U.S. with specified solid colors
#           for all land, U.S. water, and all other water.
#  Notes:
#     

#
#  Import NumPy.
#
import numpy, os

#
#  Import Nio for a NetCDF reader.
#
import Nio

#
#  To use the ScientificPython module to read in the netCDF file,
#  comment out the above "import" command, and uncomment the 
#  import line below.
#
# from Scientific.IO.NetCDF import NetCDFFile

#
#  Import Ngl support functions.
#
import Ngl
#
#  Open three netCDF files and get variables.
#
data_dir  = Ngl.pynglpath("data")
cdf_file1 = Nio.open_file(os.path.join(data_dir,"cdf","941110_P.cdf"),"r")
cdf_file2 = Nio.open_file(os.path.join(data_dir,"cdf","sstdata_netcdf.nc"),"r")
cdf_file3 = Nio.open_file(os.path.join(data_dir,"cdf","Pstorm.cdf"),"r")

#
#  This is the ScientificPython method for opening netCDF files.
#
# cdf_file1 = NetCDFFile(os.path.join(data_dir,"cdf","941110_P.cdf"),"r")
# cdf_file2 = NetCDFFile(os.path.join(data_dir,"cdf","sstdata_netcdf.nc"),"r")
# cdf_file3 = NetCDFFile(os.path.join(data_dir,"cdf","Pstorm.cdf"),"r")

psl = cdf_file1.variables["Psl"]   
sst = cdf_file2.variables["sst"]  
pf  = cdf_file3.variables["p"]  

psl_lon  =  cdf_file1.variables["lon"][:]
psl_lat  =  cdf_file1.variables["lat"][:]
psl_nlon =  len(psl_lon)
psl_nlat =  len(psl_lat)

sst_lon  =  cdf_file2.variables["lon"][:]
sst_lat  =  cdf_file2.variables["lat"][:]
sst_nlon =  len(sst_lon)
sst_nlat =  len(sst_lat)

pf_lon  =  cdf_file3.variables["lon"][:]
pf_lat  =  cdf_file3.variables["lat"][:]
pf_nlon =  len(pf_lon)
pf_nlat =  len(pf_lat)

#
#  Open a workstation.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl05p")

#----------- Begin first plot -----------------------------------------
 
resources = Ngl.Resources()

resources.sfXCStartV = float(min(psl_lon))
resources.sfXCEndV   = float(max(psl_lon))
resources.sfYCStartV = float(min(psl_lat))
resources.sfYCEndV   = float(max(psl_lat))

map = Ngl.contour_map(wks,psl,resources)

#----------- Begin second plot -----------------------------------------

ic = Ngl.new_color(wks,0.75,0.75,0.75)   # Add gray to the color map

resources.mpProjection = "Orthographic" # Change the map projection.
resources.mpCenterLonF = 180.           # Rotate the projection.
resources.mpFillOn     = True           # Turn on map fill.
resources.mpFillColors = [0,-1,ic,-1]   # Fill land and leave oceans
                                        # and inland water transparent.

resources.vpXF      = 0.1    # Change the size and location of the
resources.vpYF      = 0.9    # plot on the viewport.
resources.vpWidthF  = 0.7
resources.vpHeightF = 0.7

mnlvl = 0                        # Minimum contour level.
mxlvl = 28                       # Maximum contour level.
spcng = 2                        # Contour level spacing.
ncn   = (mxlvl-mnlvl)/spcng + 1  # Number of contour levels.

resources.cnLevelSelectionMode = "ManualLevels" # Define your own
resources.cnMinLevelValF       = mnlvl          # contour levels.
resources.cnMaxLevelValF       = mxlvl
resources.cnLevelSpacingF      = spcng

resources.cnLineThicknessF     = 2.0   # Double the line thickness.

resources.cnFillOn           = True  # Turn on contour level fill.
resources.cnMonoFillColor    = True  # Use one fill color.
resources.cnMonoFillPattern  = False # Use multiple fill patterns.
FillPatterns = numpy.zeros([ncn+1],'i')-1
FillPatterns[ncn-1:ncn+1] =  17
resources.cnFillPatterns     = FillPatterns
resources.cnLineDrawOrder      = "Predraw" # Draw lines and filled
resources.cnFillDrawOrder      = "Predraw" # areas before map gets
                                             # drawn.

resources.tiMainString = "~F26~" + cdf_file2.title

resources.sfXCStartV = float(min(sst_lon))   # Define where contour plot
resources.sfXCEndV   = float(max(sst_lon))   # should lie on the map plot.
resources.sfYCStartV = float(min(sst_lat))
resources.sfYCEndV   = float(max(sst_lat))

resources.pmLabelBarDisplayMode = "Never"  # Turn off the label bar.

map = Ngl.contour_map(wks,sst[0,:,:],resources) # Draw contours over a map.

#----------- Begin third plot -----------------------------------------

del resources
resources = Ngl.Resources()

resources.tiXAxisString = "~F25~longitude"
resources.tiYAxisString = "~F25~latitude"

resources.cnFillOn              = True     # Turn on contour fill.
resources.cnLineLabelsOn        = False    # Turn off line labels.
resources.cnInfoLabelOn         = False    # Turn off info label.

resources.nglSpreadColorEnd     = -2       # Don't include gray in contours.

resources.sfXCStartV = float(min(pf_lon))   # Define where contour plot
resources.sfXCEndV   = float(max(pf_lon))   # should lie on the map plot.
resources.sfYCStartV = float(min(pf_lat))
resources.sfYCEndV   = float(max(pf_lat))

resources.mpProjection = "LambertEqualArea"  # Change the map projection.
resources.mpCenterLonF = (pf_lon[pf_nlon-1] + pf_lon[0])/2
resources.mpCenterLatF = (pf_lat[pf_nlat-1] + pf_lat[0])/2

resources.mpLimitMode = "LatLon"    # Limit the map view.
resources.mpMinLonF   = float(min(pf_lon))
resources.mpMaxLonF   = float(max(pf_lon))
resources.mpMinLatF   = float(min(pf_lat))
resources.mpMaxLatF   = float(max(pf_lat))
resources.mpPerimOn   = True        # Turn on map perimeter.

resources.pmTickMarkDisplayMode = "Never"  # Turn off map tickmarks.

resources.tiMainString = "~F26~January 1996 storm" # Set a title.

resources.vpXF      = 0.1    # Change the size and location of the
resources.vpYF      = 0.9    # plot on the viewport.
resources.vpWidthF  = 0.7
resources.vpHeightF = 0.7

resources.nglFrame = False # Don't advance frame.

#
#  Extract the dataset from pf and scale by 0.01. 
#
pfa = 0.01 * pf[0,:,:]

#
# draw contours over map.
#
map = Ngl.contour_map(wks,pfa,resources) # Convert pf to "mb" and

txres = Ngl.Resources()
txres.txFontHeightF = 0.025  # for a text string.
txres.txFontColor   = 4
Ngl.text_ndc(wks,"~F25~Pressure (mb)",.41,.185,txres)
Ngl.frame(wks)   # Advance the frame.

#---------- Begin fourth plot ------------------------------------------

del resources.tiXAxisString  # Delete some resources you don't
del resources.tiYAxisString  # need anymore.
del resources.nglFrame

cmap = numpy.array([[1.00, 1.00, 1.00], [0.00, 0.00, 0.00], \
                    [.560, .500, .700], [.300, .300, .700], \
                    [.100, .100, .700], [.000, .100, .700], \
                    [.000, .300, .700], [.000, .500, .500], \
                    [.000, .400, .200], [.000, .600, .000], \
                    [.000, 1.00, .000], [.550, .550, .000], \
                    [.570, .420, .000], [.700, .285, .000], \
                    [.700, .180, .000], [.870, .050, .000], \
                    [1.00, .000, .000], [0.00, 1.00, 1.00], \
                    [.700, .700, .700]],'f')

rlist = Ngl.Resources()
rlist.wkColorMap = cmap
Ngl.set_values(wks,rlist)

resources.mpFillOn              = True         # Turn on map fill.
resources.mpFillAreaSpecifiers  = ["Water","Land","USStatesWater"]
resources.mpSpecifiedFillColors = [17,18,17]
resources.mpAreaMaskingOn       = True            # Indicate we want to
resources.mpMaskAreaSpecifiers  = "USStatesLand"  # mask land.
resources.mpPerimOn             = True            # Turn on a perimeter.
resources.mpGridMaskMode        = "MaskLand"      # Mask grid over land.
resources.cnFillDrawOrder       = "Predraw"       # Draw contours first.

resources.cnLevelSelectionMode = "ExplicitLevels" # Define own levels.
resources.cnLevels             = numpy.arange(985.,1046.,5.)

resources.lbTitleString  = "~F25~pressure (mb)" # Title for label bar.
resources.cnLinesOn      = False         # Turn off contour lines.
resources.lbOrientation  = "Horizontal"  # Label bar orientation.

#
#  Extract the dataset from pf and scale by 0.01.
#
pfa = 0.01*pf[1,:,:]

map = Ngl.contour_map(wks,pfa,resources)

del map
del resources
del rlist
del txres

Ngl.end()
