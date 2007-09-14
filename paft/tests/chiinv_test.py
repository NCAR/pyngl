# To test with the Numeric module, be sure to uncomment these three
# lines, and comment the NumPy lines after this.
#
# import PAF_numeric.NglA as NglA
# import MA
# default_type = "numeric"

# To test with the NumPy module, be sure to uncomment these two lines,
# and comment the three Numeric lines above.
import NglA
default_type = "numpy"

#
# Leave these both alone, regardless of what module you are testing.
#
import numpy
import Numeric
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
numer_s1  = Numeric.array(scalar1)
numer_s2  = Numeric.array(scalar2)
numer_l1  = Numeric.array([scalar1])
numer_l2  = Numeric.array([scalar2])
numpy_s1  = numpy.array(scalar1)
numpy_s2  = numpy.array(scalar2)
numpy_l1  = numpy.array([scalar1])
numpy_l2  = numpy.array([scalar2])
numpy_f1  = numpy.float32(scalar1)
numpy_u2  = numpy.uint32(scalar2)

print "chiinv (float,int input)"
print "-----------------------"
chi = NglA.chiinv(scalar1,scalar2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,default_type)

print "\nchiinv (list,scalar input)"
print "---------------------------"
chi = NglA.chiinv([scalar1],scalar2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,default_type)

print "\nchiinv (scalar,list input)"
print "---------------------------"
chi = NglA.chiinv(scalar1,[scalar2])
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,default_type)

print "\nchiinv (numeric array,scalar input)"
print "------------------------------------"
chi = NglA.chiinv(numer_s1,scalar2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,"numeric")

print "\nchiinv (numpy array,scalar input)"
print "----------------------------------"
chi = NglA.chiinv(numpy_s1,scalar2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,"numpy")

print "\nchiinv (scalar,numpy uint32 input)"
print "-----------------------------------"
chi = NglA.chiinv(scalar1,numpy_u2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,"numpy")

print "\nchiinv (numpy array,numeric array)"
print "-----------------------------------"
chi = NglA.chiinv(numpy_s1,numer_s2)
print chi,type(chi)
test_value("chiinv",chi,value)
check_type(chi,"numpy")
