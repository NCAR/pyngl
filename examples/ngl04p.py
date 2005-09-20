#
#  Import NumPy and types.
#
import Numeric, types

#
#  Import the GRIB1 reader.
#
import Nio

#
#  Import Ngl support functions.
#
import Ngl

#
#  Open the GRIB file.
#
file = Nio.open_file(Ngl.ncargpath("data") + "/grb/ced1.lf00.t00z.eta.grb","r")

names = file.variables.keys()  #  Get the variable names
print "\nVariable names:"      #  and print them out.
print names                    


#
#  For variable in names[1], retrieve and print all attributes
#  and their values (exclude printing the built-in functions).
#
print "\nThe attributes and their values for variable " + names[1] + ":"
for attrib in dir(file.variables[names[1]]):
  t = getattr(file.variables[names[1]],attrib)
  if (type(t) != types.BuiltinFunctionType):
   print "Attribute " + "'" + attrib + "' has value:", t

#
#  For variable in names[1], retrieve and print the dimension names.
#
print "\nFor variable " + names[1] + " the dimension names are:"
print file.variables[names[1]].dimensions

#
#  Open a workstation.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl04p",None)

#----------- Begin first plot -----------------------------------------
resources = Ngl.Resources()
#
#  Get the u/v variables.
#

uvar = file.variables["U_GRD_6_ISBL"]
vvar = file.variables["V_GRD_6_ISBL"]

#
#  Set resources and plot.
#
if hasattr(uvar,"units"):
  resources.tiMainString = "GRD_6_ISBL (u,v " + uvar.units + ")"
else:
  resources.tiMainString = "GRD_6_ISBL"

if hasattr(uvar,"_FillValue"):
  resources.vfMissingUValueV = uvar._FillValue
if hasattr(vvar,"_FillValue"):
  resources.vfMissingVValueV = vvar._FillValue

resources.tiMainFont    = "Times-Roman"
resources.tiXAxisString = "streamlines"

plot = Ngl.streamline(wks,uvar[0,::2,::2],vvar[0,::2,::2],resources) 

#----------- Begin second plot -----------------------------------------

del uvar
del vvar

uvar = file.variables["U_GRD_6_TRO"]
vvar = file.variables["V_GRD_6_TRO"]

#
#  Set resources and plot.
#
if hasattr(uvar,"units"):
  resources.tiMainString = "GRD_6_TRO (u,v " + uvar.units + ")"
else:
  resources.tiMainString = "GRD_6_TRO"

if hasattr(uvar,"_FillValue"):
  resources.vfMissingUValueV = uvar._FillValue
if hasattr(vvar,"_FillValue"):
  resources.vfMissingVValueV = vvar._FillValue

resources.tiXAxisFont   = "Times-Roman"  # Change the default font used.
resources.tmXBLabelFont = "Times-Roman"
resources.tmYLLabelFont = "Times-Roman"

resources.stLineColor = "green"  # Change streamlines to green.

plot = Ngl.streamline(wks,uvar[::2,::2],vvar[::2,::2],resources) 

#----------- Begin third plot -----------------------------------------

arrowlength = Ngl.get_float(plot,"stArrowLengthF")
spacing = Ngl.get_float(plot,"stMinLineSpacingF")

resources.stMinLineSpacingF = spacing * 2.0     # Set some resources based
resources.stArrowLengthF    = arrowlength * 2.0 # on resources you retrieved.
resources.stLineColor       = "red"             # Change line color to red
resources.stLineThicknessF  = 1.5

uvar = file.variables["U_GRD_6_GPML"]
vvar = file.variables["V_GRD_6_GPML"]

#
#  Set resources and plot.
#
if hasattr(uvar,"units"):
  resources.tiMainString = "GRD_6_GPML (u,v " + uvar.units + ")"
else:
  resources.tiMainString = "GRD_6_GPML"

if hasattr(uvar,"_FillValue"):
  resources.vfMissingUValueV = uvar._FillValue
if hasattr(vvar,"_FillValue"):
  resources.vfMissingVValueV = vvar._FillValue

plot = Ngl.streamline(wks,uvar[0,::2,::2],vvar[0,::2,::2],resources)

del uvar
del vvar
del plot

Ngl.end()
