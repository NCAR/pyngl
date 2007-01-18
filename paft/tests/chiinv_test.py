# Numeric module. Be sure to uncomment these two lines together.
#import NglA, MA
#default_type = "numeric"

# Numpy module. Be sure to uncomment these two lines together.
import PAF_numpy.NglA as NglA
default_type = "numpy"

import numpy
import Numeric

#
# Check type of value.
#
def check_type(value,type_expected):
  value_type = type(value)
  if type_expected == "numpy":
    try:
      import numpy
      if value_type == type(numpy.array([0])):
        print "Type test successful."
      else:
        print "Type test unsuccessful."
    except:
        print "Type test unsuccessful."
  elif type_expected == "numeric":
    try:
      import Numeric
      if value_type == type(Numeric.array([0])):
        print "Type test successful."
      else:
        print "Type test unsuccessful."
    except:
        print "Type test unsuccessful."
  
#
# Compare two values for equality.
#
def test_value(title,value1,value2,delta=None):
  if delta == None:
    delta = 1e-8
  diff = abs(value1 - value2)
  if diff[0] < delta:
    print title + " test successful."
  else:
    print title + " test unsuccessful."
  
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
