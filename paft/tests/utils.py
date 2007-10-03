import numpy
import numpy.core.ma as ma
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
  if diff < delta:
    print title + " test successful."
  else:
    print title + " test unsuccessful."

#
# Compare two numpy values for equality.
#
def test_values(title,values1,values2,delta=None):
  if delta == None:
    delta = 1e-8

  if ma.isMaskedArray(values1):
    values11 = values1.filled()
  else:
    values11 = values1
  if ma.isMaskedArray(values2):
    values22 = values2.filled()
  else:
    values22 = values2
  if numpy.all(values11 == values22):
    print title + " test successful."
  else:
    diff = numpy.max(numpy.abs(values11 - values22))

    if diff < delta:
      print title + " test successful."
    else:
      print title + " test unsuccessful."
      if isinstance(values11,numpy.ndarray):
        print "min/max values11 =" + str(numpy.min(values11)) + "/" + \
                                     str(numpy.max(values11))
      else:
        print "min/max values11 =" + str(numpy.min(numpy.array(values11))) + \
                               "/" + str(numpy.max(numpy.array(values11)))

      if isinstance(values22,numpy.ndarray):
        print "min/max values22 =" + str(numpy.min(values22)) + "/" + \
                                     str(numpy.max(values22))
      else:
        print "min/max values22 =" + str(numpy.min(numpy.array(values22))) + \
                               "/" + str(numpy.max(numpy.array(values22)))
