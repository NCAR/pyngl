#
#  File:
#    nio04.py
#
#  Synopsis:
#    Shows how to use Nio to open an existing NetCDF file and append
#    data to a variable with an unlimited dimension.
#
#  Category:
#    Processing.
#
#  Author:
#    Dave Brown
#  
#  Date of original publication:
#    August, 2007
#
#  Description:
#    This example creates a NetCDF file "nio04.nc" that has a
#    variable with an unlimited dimension. It then reads this
#    file back in and appends data to the unlimited dimension.
#
#  Effects illustrated:
#    o  Creating, reading, and appending to NetCDF files using Nio.
# 
#  Output:
#    This example produces no graphic output, but it does
#    write to standard out.
#
#  Notes:
#     

import numpy
import Nio 
import os

#
#  Creating a NetCDF file named "nio04.nc".  If there is already
#  a file with that name, delete it first.
#
filename = "nio04.nc"

if (os.path.exists(filename)):
  os.remove(filename)

#
#  Open a NetCDF file for writing file and specify a 
#  global history attribute.
#
file       = Nio.open_file(filename, "w")
file.title = "Unlimited dimension test file"

#
#  Create some dimensions.
#
file.create_dimension("lon",     3)
file.create_dimension("lat",     3)
file.create_dimension("time", None)     # unlimited dimension

print file

#
#  Create a variable of type double with three dimemsions.
#
var = file.create_variable("var", 'd', ("time", "lat","lon"))
var._FillValue = -999.0

for i in xrange(10):
# Initialize lat/lon grid to the timestep number for 10 timesteps 
  var[i] = i

print "After assigning elements 0-9 of unlimited dimension:"
print file

print "Closing '" + filename + "' file...\n"
file.close()

#
#  Reopen the file for writing
#
print "Reopening '" + filename + "' file..."
file = Nio.open_file(filename, "w")

print file

var = file.variables['var']

for i in xrange(10,20):
  var[i] = i  # add ten more elements to the unlimited dimension

print "After assigning elements 10-19 of unlimited dimension:"
print file

var[25] = 25  # add ten more elements to the unlimited dimension

print "After assigning element 25 of unlimited dimension:"
print file

print "'print var'"
print var

print "'print var[23:]'"
print var[23:]

file.close()
