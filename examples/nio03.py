#
#  File:
#    nio03.py
#
#  Synopsis:
#    Demonstrates PyNIO reading GRIB2/ writing NetCDF
#
#  Category:
#    Processing.
#
#  Author:
#    Dave Brown 
#  
#  Date of original publication:
#    June, 2006
#
#  Description:
#    This example reads a GRIB2 file and copies its contents
#    to a NetCDF file, setting some options for efficient
#    copying along the way.
#
#  Effects illustrated:
#    o  Reading a GRIB2 file, setting options, and programatically
#       writing a NetCDF file
# 
#  Output:
#    NetCDF file
#
#  Notes:
#     

import numpy 
import Nio
import Ngl
import time,os


#
#  Read a GRIB2 file from the example data directory
#
dirc = Ngl.pynglpath("data")
fname = "wafsgfs_L_t06z_intdsk60.grib2"
f = Nio.open_file(os.path.join(dirc,"grb",fname + ".grb"))

#
# Print the input file contents
#
# print f

#
# If the output file already exists, remove it
#
os.system("rm -f " + fname + ".nc")

#
# Set the PreFill option to False to improve writing performance
#
opt = Nio.options()
opt.PreFill = False

#
# Options for writing NetCDF4 "classic" file.
#
# If Nio wasn't built with netcdf 4 support, you will get a
# warning here, and the code will use netcdf 3 instead.
#
opt.Format           = "netcdf4classic"
opt.CompressionLevel = 5                  # Can go up to 9

#
# Set the history attribute
#
hatt = "Converted from GRIB2: " + time.ctime(time.time())

#
# Create the output file
#
fout = Nio.open_file(fname + ".nc","c",opt,hatt)

#
# Note that it is much more efficient if all dimensions, variables,
# and attributes are defined before any actual data is written
#

# create dimensions

dims = f.dimensions.keys()
dims.sort()
for dim in dims:
    length = f.dimensions[dim]
    fout.create_dimension(dim,length)

# create variables and attributes

vars = f.variables.keys()
vars.sort()

for var in vars:
    v = f.variables[var]
    type = v.typecode()
    vdims = v.dimensions
    fout.create_variable(var,type,vdims)
    varAtts = v.__dict__.keys()
    varAtts.sort()
    for att in varAtts:
        value = getattr(v,att)
        setattr(fout.variables[var],att,value)

# Write data contents.
# Since we are always writing the complete contents
# and the get_value/assign_value interface is the only
# way to assign scalar values if they are encountered,
# use get_value and assign_value.
# Write coordinate dimension variables first, because the library
# code gives an error message if they are not set when the
# variables are written

for var in vars:
    if dims.count(var) > 0:
        v = f.variables[var].get_value()
        fout.variables[var].assign_value(v)
        print "finished writing " + var

for var in vars:
    if dims.count(var) == 0:
        v = f.variables[var].get_value()
        fout.variables[var].assign_value(v)
        print "finished writing " + var
            
#
# print the output file contents
#
# print fout    

f.close()

# Print file size.
print "File size of",fname+ ".nc:"
os.system("wc -c " + fname + ".nc")
