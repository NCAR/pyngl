# To test with the Numeric module, be sure to uncomment these three
# lines, and comment the NumPy lines after this.
#
# import PAF_numeric.Ngl as Ngl
# import MA
# default_type = "numeric"

import sys

# To test with the NumPy module, be sure to uncomment these two lines,
# and comment the three Numeric lines above.
import Ngl
default_type = "numpy"

#
# Leave these both alone, regardless of what module you are testing.
#
import numpy
import Numeric
from utils import *

#
# Begin linmsg tests.
#

#
# Correct value for first set of tests.
#
npts = 101
x     = Ngl.fspan(0.,npts-1,101)
xorig = Ngl.fspan(0.,npts-1,101)

print "linmsg (float input)"
print "--------------------"
xnew = Ngl.linmsg(x)
print xnew,type(xnew),"\n"
test_values("linmsg",xnew,xorig)
check_type(xnew,default_type)
del xnew

print "linmsg (float, int input)"
print "-------------------------"
xnew = Ngl.linmsg(x,0)
print xnew,type(xnew),"\n"
test_values("linmsg",xnew,xorig)
check_type(xnew,default_type)
del xnew

print "linmsg (float, int, int input)"
print "------------------------------"
xnew = Ngl.linmsg(x,0,0)
print xnew,type(xnew),"\n"
test_values("linmsg",xnew,xorig)
check_type(xnew,default_type)
del xnew

fill_value = -999
x[3]  = fill_value
x[7]  = fill_value
x[21] = fill_value
x[50] = fill_value
x[80] = fill_value
x[99] = fill_value

print "linmsg (float, int, fill_value input)"
print "-------------------------------------"
xnew = Ngl.linmsg(x,0,fill_value)
print xnew,type(xnew),"\n"
test_values("linmsg",xnew,xorig)
check_type(xnew,default_type)

x[0]       = fill_value
x[100]     = fill_value
xorig[0]   = fill_value
xorig[99]  = fill_value
xorig[100] = fill_value

print "linmsg (float, int, fill_value input)"
print "-------------------------------------"
xnew = Ngl.linmsg(x,1,fill_value)
print xorig,type(xorig),"\n"
print xnew,type(xnew),"\n"
test_values("linmsg",xnew,xorig)
check_type(xnew,default_type)
