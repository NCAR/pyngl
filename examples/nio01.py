#
#  File:
#    nio01.py
#
#  Synopsis:
#    Shows how to use Nio to create a NetCDF file and then to read it.
#
#  Category:
#    Processing.
#
#  Author:
#    Dave Brown (modelled after an example of Konrad Hinsen).
#  
#  Date of original publication:
#    November, 2005
#
#  Description:
#    This example creates a NetCDF file "nio01.nc" that has two
#    variables and several global attributes; it prints out some
#    information before writing the file.  The example then reads
#    "nio01.nc" and prints out some information.
#
#  Effects illustrated:
#    o  Creating and reading NetCDF files using Nio.
# 
#  Output:
#    This example produces no graphic output, but it does
#    write to standard out.
#
#  Notes:
#     

import numpy 
import Nio 
import time, os

#
#  Function to retrieve the user's name.
#
def getUserName():
  try:
    import os, pwd, string
  except ImportError:
    return "unknown user"
  pwd_entry = pwd.getpwuid(os.getuid())
  name = string.strip(string.splitfields(pwd_entry[4], ",")[0])
  if name == "":
    name = pwd_entry[0]
  return name

#
#  Creating a NetCDF file named "nio01.nc".  If there is already
#  a file with that name, delete it first.
#
if (os.path.exists("nio01.nc")):
  os.system("/bin/rm -f nio01.nc")

#
#  Specify a global history attribute and open a NetCDF file
#  for writing.
#
hatt = "Created " + time.ctime(time.time()) + " by " + getUserName()
file = Nio.open_file("nio01.nc", "w", None, hatt)

#
#  Create some global attributes.
#
file.title   = "Nio test NetCDF file"
file.series  = [ 1, 2, 3, 4, 5,6 ]
file.version = 45

#
#  Create some dimensions.
#
file.create_dimension("xyz",    3)
file.create_dimension(  "n",   20)
file.create_dimension(  "t", None) # unlimited dimension

#
#  Create some variables.
#
foo = file.create_variable("foo", "i", ("n", "xyz"))
foo[:,:] = 0.                    #  Initialize foo to zeros.
foo[0,:] = [42., 42.1, 42.2]     #  Specify the first row.
foo[:,1] = 1.                    #  Column one is "1." everywhere.

#
#  Specify the "units" attribute.
#
foo.units = "arbitrary"

#
#  Print out the first row of "foo" and the dimensions of "foo".
#
print "On write, foo first row and foo dimensions:"
print "  " + str(foo[0])
print "  " + str(foo.dimensions)

#
#  Create a second integer variable.
#
bar = file.create_variable("bar","i", ("t", "n"))
for i in range(10):
  bar[i] = i            #  Each row of "n" integers equals the row number.

print "On write, bar.shape:"
print "  " + str(bar.shape)

#
#  Print out the file dimesions and variables and write the file.
#
print "On write, file dimensions:"
print "  " + str(file.dimensions)
print "On write, file variables:"
print "  " + str(file.variables)

file.close()

#
#  Read the file we just created.
#
file = Nio.open_file("nio01.nc", "r")

print "\nOn read, a summary of the file contents:"
print file
print "\nOn read, file dimensions:"
print "  " + str(file.dimensions)
print "On read, file variables:"
print "  " + str(file.variables)

#
#  Read the first row of "foo".
#
foo = file.variables["foo"]
print "\nOn read, a summary of variable 'foo':'"
print foo
foo_array = foo[:]
foo_units = foo.units

#
#  Print out some of the same things we did on the write.
#
print "On read, foo first row:"
print "  " + str(foo[0])
print "On read, foo units:"
print "  " + str(foo.units)

file.close()
