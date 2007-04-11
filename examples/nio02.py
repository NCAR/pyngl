#
#  File:
#    nio02.py
#
#  Synopsis:
#    Demonstrates PyNIO docstrings
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
#    This example reads NetCDF file 'pop.nc' only to provide
#    an instance of the NioFile and NioVariable classes.
#    It prints all available docstrings in a systematic way,
#
#  Effects illustrated:
#    o  Reading a NetCDF file, and learning to use PyNIO through
#       its self-documenting docstrings.
# 
#  Output:
#    None
#
#  Notes:
#     

import numpy 
import Nio
import Ngl

print """
This example prints all Nio docstrings. For clarity each docstring
is bracketed by a line of equal signs, and preceded by a comment naming the
particular docstring.
"""
#
# print the Nio summary documentation
#

print "The Nio docstring (Nio.__doc__):"

print "======================================================================="
print Nio.__doc__
print "======================================================================="


print "The NioOptions constructor options docstring (Nio.options.__doc__):"

print "======================================================================="
print Nio.options.__doc__
print "======================================================================="

# create an NioOptions object

opt = Nio.options()

print "The NioOptions class docstring (opt.__doc__):"

print "======================================================================="
print opt.__doc__
print "======================================================================="


print "The NioFile constructor open_file docstring (Nio.open_file.__doc__):"

print "======================================================================="
print Nio.open_file.__doc__
print "======================================================================="

#
#  Read the file pop.nc
#
dirc = Ngl.pynglpath("data")
f = Nio.open_file(dirc + "/cdf/pop.nc")

print "The NioFile object docstring (f.__doc__):"
print "======================================================================="
print f.__doc__
print "======================================================================="

print "The close method docstring (f.close.__doc__):"
print "======================================================================="
print f.close.__doc__
print "======================================================================="

print "The create_dimension method docstring (f.create_dimension.__doc__):"
print "======================================================================="
print f.create_dimension.__doc__
print "======================================================================="

print "The create_variable method docstring (f.create_variable.__doc__):"
print "======================================================================="
print f.create_variable.__doc__
print "======================================================================="

v = f.variables['t']

print "The NioVariable object docstring (f.variables['varname'].__doc__):"
print "======================================================================="
print v.__doc__
print "======================================================================="

print "The assign_value method docstring (f.variables['varname'].assign_value.__doc__):"
print "======================================================================="
print v.assign_value.__doc__
print "======================================================================="

print "The get_value method docstring (f.variables['varname'].get_value.__doc__):"
print "======================================================================="
print v.get_value.__doc__
print "======================================================================="

print "The typecode method docstring (f.variables['varname'].typecode.__doc__):"
print "======================================================================="
print v.typecode.__doc__
print "======================================================================="


f.close()
