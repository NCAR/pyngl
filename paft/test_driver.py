import NglA, MA

#
#  chiinv
#
print "chiinv"
print "------"
print NglA.chiinv(0.99,2.0)

#
#  linmsg
#
x = MA.array([
     [ 1190., 1455., 1550., -999., 1745., 1770.,  \
      1900.,  -999., -999., -999., 2335., 2490.,  \
      2720.,  2710., 2530., 2900., 2760., -999.], \
     [ 1115., -999., 1515., 1794., -999. ,1710.,  \
       1830., 1920., 1970., 2300., 2280., 2520.,  \
       2630., -999. ,-999. ,2800., -999.,-999. ]
    ],fill_value=-999.)

#
#  linmsg
#
print "\nlinmsg"
print "------"
intp = NglA.linmsg(x)
print intp
print "Type of intp = ",type(intp),", fill_value = ",intp.fill_value()
