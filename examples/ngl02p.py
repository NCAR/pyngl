#
#  File:
#    ngl02p.py
#
#  Synopsis:
#    Draws a sequence of contour visualizations from simple to more complex.
#
#  Category:
#    Contouring
#
#  Author:
#    Fred Clare (based on a code of Mary Haley)
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    This is the second of the tutorial examples and is meant to
#    show PyNGL contourng basics by progressing through successively more
#    complicated contouring visualizations.
#
#  Effects illustrated:
#    o  Reading data from a NetCDF file.
#    o  Changing the color map.
#    o  Contours with colored lines.
#    o  Contours with hatch pattern fill.
#    o  Turning off the label bar on a coutour plot.
#    o  Color filled coutours.
#    o  Using a long_name in a NetCDF file for a title.
#    o  Writing data to an ASCII file.
#    o  Contouring using grayscale contour levels.
# 
#  Output:
#    Five contouring visualizations are produced:
#      1.)  Contour with contour lines drawn in the foreground color.
#      2.)  Contour with colored contour lines.
#      3.)  Contour with hatch pattern fill.
#      4.)  Contour with color fill.
#      5.)  Contour with grayscale fill.
#
#  Notes:
#     

#
#  Import Python modules to be used.
#
import numpy
import sys,os

#
#  Import Nio (for reading netCDF files).
#
import Nio

#
#  Import Ngl support functions.
#
import Ngl

#  
#  Open the netCDF file.
#  
cdf_file = Nio.open_file(os.path.join(Ngl.pynglpath("data"),"cdf","contour.cdf"),"r")

#
#  This is the ScientificPython method for opening a netCDF file.
#
# cdf_file = NetCDFFile(os.path.join(Ngl.pynglpath("data"),"cdf","contour.cdf"),"r")

#
#  Associate Python variables with NetCDF variables.
#  These variables have associated attributes.
#
temp = cdf_file.variables["T"]    # temperature
Z    = cdf_file.variables["Z"]    # geopotential height
pres = cdf_file.variables["Psl"]  # pressure at mean sea level
lat  = cdf_file.variables["lat"]  # latitude
lon  = cdf_file.variables["lon"]  # longitude

#
#  Open a workstation and specify a different color map.
#
wkres = Ngl.Resources()
wkres.wkColorMap = "default"
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl02p",wkres)

resources = Ngl.Resources()
#
#  Define a NumPy data array containing the temperature for the
#  first time step and first level. This array does not have the
#  attributes that are associated with the variable temp.
#
tempa = temp[0,0,:,:]
tempa = tempa - 273.15

#
#  Specify the main title base on the long_name attribute of temp.
#
if hasattr(temp,"long_name"):
  resources.tiMainString = temp.long_name

plot = Ngl.contour(wks,tempa,resources)

#----------- Begin second plot -----------------------------------------

resources.cnMonoLineColor = False  # Allow multiple colors for contour lines.
resources.tiMainString    = "Temperature (C)"

plot = Ngl.contour(wks,tempa,resources)  # Draw a contour plot.

#----------- Begin third plot -----------------------------------------

resources.cnFillOn          = True    # Turn on contour line fill.
resources.cnMonoFillPattern = False   # Turn off using a single fill pattern.
resources.cnMonoFillColor   = True
resources.cnMonoLineColor   = True

if hasattr(lon,"long_name"):
  resources.tiXAxisString = lon.long_name
if hasattr(lat,"long_name"):
  resources.tiYAxisString = lat.long_name
resources.sfXArray        = lon[:]
resources.sfYArray        = lat[:]
resources.pmLabelBarDisplayMode = "Never" # Turn off label bar.

plot = Ngl.contour(wks,tempa,resources)   # Draw a contour plot.

#---------- Begin fourth plot ------------------------------------------

#
#  Specify a new color map.
#
rlist = Ngl.Resources()
rlist.wkColorMap = "BlGrYeOrReVi200"
Ngl.set_values(wks,rlist)

resources.cnMonoFillPattern     = True     # Turn solid fill back on.
resources.cnMonoFillColor       = False    # Use multiple colors.
resources.cnLineLabelsOn        = False    # Turn off line labels.
resources.cnInfoLabelOn         = False    # Turn off informational
                                             # label.
resources.pmLabelBarDisplayMode = "Always" # Turn on label bar.
resources.cnLinesOn             = False    # Turn off contour lines.

resources.tiMainFont      = "Helvetica-bold"
resources.tiXAxisFont     = "Helvetica-bold"
resources.tiYAxisFont     = "Helvetica-bold"

if hasattr(Z,"long_name"):
  resources.tiMainString = Z.long_name
plot = Ngl.contour(wks,Z[0,0,:,:],resources)    # Draw a contour plot.

#---------- Begin fifth plot ------------------------------------------

cmap = numpy.array([[0.00, 0.00, 0.00], [1.00, 1.00, 1.00], \
                    [0.10, 0.10, 0.10], [0.15, 0.15, 0.15], \
                    [0.20, 0.20, 0.20], [0.25, 0.25, 0.25], \
                    [0.30, 0.30, 0.30], [0.35, 0.35, 0.35], \
                    [0.40, 0.40, 0.40], [0.45, 0.45, 0.45], \
                    [0.50, 0.50, 0.50], [0.55, 0.55, 0.55], \
                    [0.60, 0.60, 0.60], [0.65, 0.65, 0.65], \
                    [0.70, 0.70, 0.70], [0.75, 0.75, 0.75], \
                    [0.80, 0.80, 0.80], [0.85, 0.85, 0.85]],'f')

rlist.wkColorMap = cmap       #  Specify a new color map.
Ngl.set_values(wks,rlist)

#
#  If the pressure field has a long_name attribute, use it for a title.
#
if hasattr(pres,"long_name"):
  resources.tiMainString = pres.long_name

presa = 0.01*pres[0,:,:]

plot = Ngl.contour(wks,presa,resources)  # Draw a contour plot.

print "\nSubset [2:6,7:10] of temp array:" # Print subset of "temp" variable.
print tempa[2:6,7:10]
print "\nDimensions of temp array:"        # Print dimension names of T.
print temp.dimensions
print "\nThe long_name attribute of T:"    # Print the long_name attribute of T.
print temp.long_name 
print "\nThe nlat data:"                   # Print the lat data.
print lat[:]           
print "\nThe nlon data:"                   # Print the lon data.
print lon[:]          

#
#  Write a subsection of tempa to an ASCII file.
#
os.system("/bin/rm -f data.asc")
sys.stdout = open("data.asc","w")
for i in range(7,2,-2):
  for j in range(0,5):
    print "%9.5f" % (tempa[i,j])

# Clean up (not really necessary, but a good practice).

sys.stdout.close()
del plot 
del resources
del temp

Ngl.end()
