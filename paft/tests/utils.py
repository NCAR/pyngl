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
  
