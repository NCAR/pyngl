import Ngl, numpy
from numpy import ma
from utils import *

npts = 400
x = Ngl.fspan(100.,npts-1,npts)
y = 500.+ x * numpy.sin(0.031415926535898*x)

wks = Ngl.open_wks("ps","datandc")

xy = Ngl.xy(wks,x,y)

#
# Test with all in-range values and no missing values present.
#
x_dat = x
y_dat = numpy.absolute(y)

x_ndc, y_ndc   = Ngl.datatondc(xy, x_dat, y_dat)
x_dat2, y_dat2 = Ngl.ndctodata(xy, x_ndc, y_ndc)

check_type(x_ndc)
check_type(y_ndc)
check_type(x_dat2)
check_type(y_dat2)

test_values("no oor or msg data: datatondc/ndctodata",x_dat,x_dat2,delta=1e-4)
test_values("no oor or msg data: datatondc/ndctodata",y_dat,y_dat2,delta=1e-4)

#
# Test with some values outside the range, and hence you should
# get masked arrays returned.
#
x_dat = numpy.array([100.,300.,200.,350.,425.,200.])
y_dat = numpy.array([100.,500.,100.,900.,600.,500.])
x_ndc, y_ndc   = Ngl.datatondc(xy, x_dat, y_dat)
x_dat2, y_dat2 = Ngl.ndctodata(xy, x_ndc, y_ndc)

check_type(x_ndc,"nma")
check_type(y_ndc,"nma")
check_type(x_dat2,"nma")
check_type(y_dat2,"nma")

# At index locations 3 and 4, values outside the range were input,
# so these should have be set to missing in the output.
x_dat = ma.array(x_dat,mask=[0,0,0,1,1,0])
y_dat = ma.array(y_dat,mask=[0,0,0,1,1,0])

test_values("oor data: datatondc/ndctodata",x_dat,x_dat2)
test_values("oor data: datatondc/ndctodata",y_dat,y_dat2,delta=1e-4)

#
# Test with missing value (masked arrays) input.
#
x_dat          = ma.masked_values([100.,300.,-999.,350., 400.,200.],-999.)
y_dat          = ma.masked_values([100.,500., 100.,200.,-999.,500.],-999.)
x_ndc, y_ndc   = Ngl.datatondc(xy, x_dat, y_dat)
x_dat2, y_dat2 = Ngl.ndctodata(xy, x_ndc, y_ndc)
check_type(x_ndc,"nma")
check_type(y_ndc,"nma")
check_type(x_dat2,"nma")
check_type(y_dat2,"nma")

# At index locations 2 and 4, missing values were input,
# so these should have be set to missing in the output.
x_dat = ma.array(x_dat,mask=[0,0,1,0,1,0])
y_dat = ma.array(y_dat,mask=[0,0,1,0,1,0])
test_values("msg data: datatondc/ndctodata",x_dat,x_dat2)
test_values("msg data: datatondc/ndctodata",y_dat,y_dat2,delta=1e-4)

#
# Test with a mix of out-of-range and missing values.
#
x_dat          = ma.masked_values([900.,300.,-999.,350.,400,200.],-999.)
y_dat          = ma.masked_values([100.,500., 100.,900.,600,500.],-999.)
x_ndc, y_ndc   = Ngl.datatondc(xy, x_dat, y_dat)
x_dat2, y_dat2 = Ngl.ndctodata(xy, x_ndc, y_ndc)
check_type(x_ndc,"nma")
check_type(y_ndc,"nma")
check_type(x_dat2,"nma")
check_type(y_dat2,"nma")

# At index locations 0, 3, and 3, either missing values or values
# outside the range were input, so these should have be set to
# missing in the output.
x_dat = ma.array(x_dat,mask=[1,0,1,1,0,0])
y_dat = ma.array(y_dat,mask=[1,0,1,1,0,0])
test_values("oor and msg data: datatondc/ndctodata",x_dat,x_dat2)
test_values("oor and msg data: datatondc/ndctodata",y_dat,y_dat2,delta=1e-4)

Ngl.end()
