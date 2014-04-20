#
#  File:
#    ngl03.py
#
#  Synopsis:
#    Vector visualizations from simple to complex.
#
#  Categories:
#    Vectors
#
#  Author:
#    Fred Clare (based on a code of Mary Haley)
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    This is the third of the tutorial examples and is meant to
#    show PyNGL vector plot  basics by progressing through successively 
#    more complicated examples.
#
#  Effects illustrated:
#    o  Reading from a NetCDF file.
#    o  Specifying vector reference size.
#    o  Specifying missing values.
#    o  Coloring vectors based on an associated array of colors.
#    o  Coloring vectors based on scalar values.
# 
#  Output:
#    Four visualizations are produced:
#      1.)  Basic vector plot with no vectors where there are missing values.
#      2.)  Vectors using a reference magnitude and length.
#      3.)  Colored vectors.
#      4.)  Color filled vector arrows colored using a scalar field.
#
#  Notes:
#     

#
#  Import numpy.
#
import numpy
import os

#
#  Import Nio read functions (for netCDF files).
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
#  Open the netCDF files.
#
dirc  = Ngl.pynglpath("data")
ufile = Nio.open_file(os.path.join(dirc,"cdf","Ustorm.cdf"),"r")  # Open two netCDF files.
vfile = Nio.open_file(os.path.join(dirc,"cdf","Vstorm.cdf"),"r")

#
#  This is the ScientificPython method for opening netCDF files.
#
# ufile = NetCDFFile(os.path.join(dirc,"cdf","Ustorm.cdf")"r")  # Open two netCDF files.
# vfile = NetCDFFile(os.path.join(dirc,"cdf","Vstorm.cdf"),"r")

#
#  Get the u/v variables.
#
u = ufile.variables["u"]
v = vfile.variables["v"]
lat = ufile.variables["lat"]
lon = ufile.variables["lon"]
ua = u[0,:,:]
va = v[0,:,:]

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl03p")

vc = Ngl.vector(wks,ua,va)

#----------- Begin second plot -----------------------------------------

resources = Ngl.Resources()

resources.vcMinFracLengthF = 0.33
resources.vcRefMagnitudeF  = 20.0
resources.vcRefLengthF     = 0.045
resources.vcMonoLineArrowColor  = False   # Draw vectors in color.

vc = Ngl.vector(wks,ua,va,resources)

#----------- Begin third plot -----------------------------------------

resources.tiMainString  = "~F22~wind velocity vectors - January 1996"
resources.tiXAxisString = "longitude"
resources.tiYAxisString = "latitude"
 
resources.vfXCStartV  = float(lon[0])             # Define X/Y axes range
resources.vfXCEndV    = float(lon[len(lon[:])-1]) # for vector plot.
resources.vfYCStartV  = float(lat[0])
resources.vfYCEndV    = float(lat[len(lat[:])-1])



vc = Ngl.vector(wks,ua,va,resources)

#---------- Begin fourth plot ------------------------------------------

tfile = Nio.open_file(os.path.join(dirc,"cdf","Tstorm.cdf"),"r")    # Open a netCDF file.
#
#  This is the ScientificPython method for opening netCDF files.
#
# tfile = NetCDFFile(os.path.join(dirc,"cdf","Tstorm.cdf"),"r")    # Open a netCDF file.

temp = tfile.variables["t"]

tempa = temp[0,:,:]
#
#  Convert from degrees Kelvin to degrees F.
#
# Missing values are automatically handled here, because
# tempa is a masked array.
#
tempa      = (tempa-273.15)*9.0/5.0+32.0
temp_units = "(deg F)"

cmap = numpy.array([[.560, .500, .700], [.300, .300, .700], \
                    [.100, .100, .700], [.000, .100, .700], \
                    [.000, .300, .700], [.000, .500, .500], \
                    [.000, .700, .100], [.060, .680, .000], \
                    [.550, .550, .000], [.570, .420, .000], \
                    [.700, .285, .000], [.700, .180, .000], \
                    [.870, .050, .000], [1.00, .000, .000], \
                    [.700, .700, .700]],'f')

resources.vcFillArrowsOn           = True  # Fill the vector arrows
resources.vcMonoFillArrowFillColor = False # in different colors
resources.vcFillArrowEdgeColor     = 1     # Draw the edges in black.
resources.vcFillArrowWidthF        = 0.055 # Make vectors thinner.
resources.vcLevelPalette           = cmap

resources.tiMainString      = "~F22~wind velocity vectors colored by temperature " + temp_units
resources.tiMainFontHeightF = 0.02  # Make font slightly smaller.

vc = Ngl.vector_scalar(wks,ua,va,tempa,resources) # Draw a vector plot of

del vc
del u
del v
del temp
del tempa

Ngl.end()
