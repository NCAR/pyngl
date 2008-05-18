import Ngl, numpy
from numpy import ma
from utils import *

npts = 400
x = Ngl.fspan(100.,npts-1,npts)
y = 500.+ x * numpy.sin(0.031415926535898*x)

wks = Ngl.open_wks("ps","datandc")

xy = Ngl.xy(wks,x,y)

x_in = x
y_in = numpy.absolute(y)

x_out, y_out = Ngl.datatondc(xy,x_in,y_in)
x_in2,y_in2  = Ngl.ndctodata(xy,x_out,y_out)
test_values("datatondc/ndctodata",x_in,x_in2,delta=1e-4)
test_values("datatondc/ndctodata",y_in,y_in2,delta=1e-4)

x_dat          = ma.masked_values([100.,300.,-999.,350.,400,200.],-999.)
y_dat          = ma.masked_values([100.,500., 100.,900.,600,500.],-999.)
x_ndc, y_ndc   = Ngl.datatondc(xy, x_dat, y_dat)
x_dat2, y_dat2 = Ngl.ndctodata(xy, x_ndc, y_ndc)

x_dat = ma.array(x_dat,mask=[0,0,1,1,0,0])
y_dat = ma.array(y_dat,mask=[0,0,1,1,0,0])
test_values("datatondc/ndctodata",x_dat,x_dat2,delta=1e-2)
test_values("datatondc/ndctodata",y_dat,y_dat2,delta=1e-2)

Ngl.end()
