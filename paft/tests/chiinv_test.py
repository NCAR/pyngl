import Ngl
default_type = "numpy"

import numpy
from utils import *

#
# Begin chiinv tests.
#

#
# Correct value for first set of tests.
#
value = 9.21034037

scalar1 = 0.99
scalar2 = 2
numpy_s1  = numpy.array(scalar1)
numpy_s2  = numpy.array(scalar2)
numpy_l1  = numpy.array([scalar1])
numpy_l2  = numpy.array([scalar2])
numpy_f1  = numpy.float32(scalar1)
numpy_u2  = numpy.uint32(scalar2)

print "chiinv (float,int input)"
print "-----------------------"
chi = Ngl.chiinv(scalar1,scalar2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,default_type)
print type(numpy.array([0,1]))

print "\nchiinv (list,scalar input)"
print "---------------------------"
chi = Ngl.chiinv([scalar1],scalar2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,default_type)

print "\nchiinv (scalar,list input)"
print "---------------------------"
chi = Ngl.chiinv(scalar1,[scalar2])
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,default_type)

print "\nchiinv (numpy array,scalar input)"
print "----------------------------------"
chi = Ngl.chiinv(numpy_s1,scalar2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,"numpy")

print "\nchiinv (scalar,numpy uint32 input)"
print "-----------------------------------"
chi = Ngl.chiinv(scalar1,numpy_u2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,"numpy")

