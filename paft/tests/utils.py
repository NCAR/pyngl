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
# Compare two single values for equality.
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
# Compare two numpy values for equality.
#
def test_values(title,values1,values2,delta=None):
  if numpy.all(values1 == values2):
    print title + " test successful."
  else:
    print title + " test unsuccessful."
    if isinstance(values1,numpy.ndarray):
      print "min/max values1 =" + str(numpy.min(values1)) + "/" + \
                                  str(numpy.max(values1))
    else:
      print "min/max values1 =" + str(numpy.min(numpy.array(values1))) + \
                            "/" + str(numpy.max(numpy.array(values1)))

    if isinstance(values2,numpy.ndarray):
      print "min/max values2 =" + str(numpy.min(values2)) + "/" + \
                                  str(numpy.max(values2))
    else:
      print "min/max values2 =" + str(numpy.min(numpy.array(values2))) + \
                            "/" + str(numpy.max(numpy.array(values2)))
  
