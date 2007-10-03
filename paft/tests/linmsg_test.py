# To test with the Numeric module, be sure to uncomment these three
# lines, and comment the NumPy lines after this.
#

import sys

# To test with the NumPy module, be sure to uncomment these two lines,
# and comment the three Numeric lines above.
import Ngl
default_type = "numpy"

#
# Leave these both alone, regardless of what module you are testing.
#
import numpy
import numpy.core.ma as ma
from utils import *

#
# Begin linmsg tests.
#

#
# Correct value for first set of tests.
#
npts = 101
x     = Ngl.fspan(0.,npts-1,101)
y     = Ngl.fspan(0.,npts-1,101)
xorig = Ngl.fspan(0.,npts-1,101)

xnew = Ngl.linmsg(x)
test_values("linmsg (float)",xnew,xorig)
check_type(xnew,default_type)
del xnew

xnew = Ngl.linmsg(x,0)
test_values("linmsg (float,int)",xnew,xorig)
check_type(xnew,default_type)
del xnew

xnew = Ngl.linmsg(x,0,0)
test_values("linmsg (float,int,int)",xnew,xorig)
check_type(xnew,default_type)
del xnew

fill_value = -999
x[3]  = fill_value
x[7]  = fill_value
x[21] = fill_value
x[50] = fill_value
x[80] = fill_value
x[99] = fill_value

xma = ma.masked_array(x,fill_value=fill_value)

xnew = Ngl.linmsg(x,0,fill_value=fill_value)
xnewma = Ngl.linmsg(xma,0)
test_values("linmsg (some fill_values)",xnew,xorig)
test_values("linmsg (some fill_values)",xnewma,xorig)

sys.exit()

check_type(xnew,default_type)
del xnew

x[0]       = fill_value
x[100]     = fill_value
xorig[0]   = fill_value
xorig[99]  = fill_value
xorig[100] = fill_value

xnew = Ngl.linmsg(x,1,fill_value=fill_value)
test_values("linmsg (end fill_values)",xnew,xorig)
check_type(xnew,default_type)
del xnew

xnew = Ngl.linmsg(x,-1,fill_value=fill_value)
test_values("linmsg (interpolated end fill_values)",xnew[1:99],y[1:99])
test_values("linmsg",[xnew[0],xnew[100]],[xnew[1],xnew[99]])
check_type(xnew,default_type)

del x
del y
del xnew

x = [[[1.,2.,3.],[4.,5.,6.]],[[7.,8.,9.],[10.,11.,12.]]]
y = [[[1.,2.,3.],[4.,5.,6.]],[[7.,8.,9.],[10.,11.,12.]]]

xnew = Ngl.linmsg(x,fill_value=-999)
test_values("linmsg (multi-d, no missing values)",xnew,y)
del xnew

x[1][1][1] = -999
x[1][0][2] = -999
x[0][0][2] = -999
y = [[[1.,2.,2.],[4.,5.,6.] ],[ [7.,8.,8.],[10.,11.,12.]]]
xnew = Ngl.linmsg(x,-1,fill_value=-999)
test_values("linmsg (multi-d, missing values)",xnew,y)


fv = -999
x[0][0][0] = fv
x[1][1][2] = fv
y = [[[fv,2.,fv],[4.,5.,6.] ],[ [7.,8.,fv],[10.,fv,fv]]]
xnew = Ngl.linmsg(x,1,fill_value=fv)
test_values("linmsg (multi-d, missing values)",xnew,y)
test_values("linmsg (multi-d, msg)",[xnew[0][0][0],xnew[0][0][2], \
                                     xnew[1][0][2],xnew[1][1][1], \
                                     xnew[1][1][2]],[fv,fv,fv,fv,fv])

del x
x = [ 1190.,1455.,1550.,-999.,1745.,1770., \
      1900.,-999.,-999.,-999.,2335.,2490., \
      2720.,2710.,2530.,2900.,2760.,-999. ]

xout = [ 1190.,1455.  ,1550. ,1647.5 ,1745.,1770., \
         1900.,2008.75,2117.5,2226.25,2335.,2490., \
         2720.,2710.  ,2530. ,2900.  ,2760.,-999.]

x = Ngl.linmsg(x,0,fill_value=-999)   # missing end point(s) unaltered
test_values("linmsg (same array used for output)",x,xout)


fv = 1.e10
y = [ 1115.,fv ,1515.,1794.,fv ,1710., \
      1830.,1920.,1970.,2300.,2280.,2520., \
      2630.,fv ,fv ,2800.,fv ,fv]


# endPoint= -1 missing end point(s) set to nearest non-missing value
ynew = Ngl.linmsg(y,-1,fill_value=fv)

yout = [1115.,1315.,1515.,1794.,1752.,1710., \
        1830.,1920.,1970.,2300.,2280.,2520., \
        2630.,2686.66666667, 2743.33333333,2800.,2800.,2800.]

test_values("linmsg (yet another test)",ynew,yout,delta=1e-1)



