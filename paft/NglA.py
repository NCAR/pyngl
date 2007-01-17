import paf_version
__version__              = paf_version.version
__array_module__         = paf_version.array_module
__array_module_version__ = paf_version.array_module_version
HAS_NUM                  = paf_version.HAS_NUM

import fplib

#
# Test to make sure we can actually load numpy or Numeric.
# If HAS_NUM = 2, then user has imported numpy version of module.
# If HAS_NUM = 1, then user has imported Numeric version of module.
#
if HAS_NUM == 2:
  try:
    import numpy
  except ImportError:
    print 'Cannot find numpy.'
    print 'Will try Numeric, even though you have loaded numpy'
    print 'version of this module.'
    try:
      import Numeric
    except ImportError:
      print 'Cannot find Numeric either, cannot proceed.'
      exit
else:
  try:
    import Numeric
  except ImportError:
    print 'Cannot find Numeric.'
    print 'Will try numpy, even though you have loaded Numeric'
    print 'version of this module.'
    try:
      import numpy
    except ImportError:
      print 'Cannot find numpy either, cannot proceed.'
      exit

#
#  get_ma_fill_value(arr)
#     input: 
#       arr - any Python object
#     output:
#       Two values: type, fill_value
#         if arr is a Numeric masked array:
#            type = "MA" and fill_value is the fill value
#         if arr is a numpy masked array:
#            type = "num" and fill_value is the fill value
#         if arr is not a masked array
#            type and fill_value are returned as None.
#
def get_ma_fill_value(arr):
  try:
    import MA
    fv = None
#
#  If arr is a Numeric masked array, return its fill value
#
    if (type(arr) == type(MA.array([0]))):
      fv = arr.fill_value()
#
#  For Numeric 2.4 or later, the fill value is returned as
#  a single-element array.  For Numeric releases prior to
#  2.4, the fill value is returned as a numeric value, so
#  return it.
#
      try:
        len(fv)
        return "MA",fv[0]
      except:
        return "MA",fv
  except:
    pass
#
#  If not a Numeric masked array, try for NumPy masked array.
#
  try:
    "Try numpy"
    import numpy.core.ma
    if (type(arr) == type(numpy.core.ma.array([0]))):
      return "num",arr.fill_value()
  except:
    pass
#
#  Neither a Numeric nor a NumPy masked array.
#
  return None, None

def chiinv(x,y):
  return fplib.chiinv(x,y)

def linmsg(x, end_pts_msg=0, max_msg=None, fill_value=1.e20):
#
#  If max_msg is the default, then set it to "0" which will
#  be interpreted in the extension module to be the maximum.
#
  if (max_msg == None):
    max_msg = 0
#
#  If end_pts_msg is 0, then end points that are missing values 
#  are returned as missing; if 1, then nearest non-missing value
#  is used.  If end_pts_msg is set to 1 here, the value passed
#  to the extension module is -1, since that is what the Fortran
#  function wants.
#
  if (end_pts_msg == 1):
    end_pts_msg = -1

#
#  If input array is a Numeric masked array then return a Numeric
#  masked array; if numpy masked array, return a numpy masked array.
#  Otherwise missing values are dealt with using the fill_value.
#
  type, fv = get_ma_fill_value(x)
  if (fv != None):
    aret = fplib.linmsg(x.filled(fv), end_pts_msg, max_msg, fv)
    if (type == "MA"):
      import MA
      return MA.array(aret, fill_value=fv)
    elif (type == "num"):
      import numpy.core.ma
      return numpy.core.ma.array(aret, fill_value=fv)
  else:
    return fplib.linmsg(x,end_pts_msg,max_msg,fill_value)

def regline(x, y, fill_value_x=1.e20, fill_value_y=1.e20, 
                    return_info=True):
  type_x, fv_x = get_ma_fill_value(x)
  type_y, fv_y = get_ma_fill_value(y)
  if (fv_x != None or fv_y != None):
    aret = fplib.regline(x.filled(fv_x), y.filled(fv_y), fv_x, fv_y)
    if (type_x == "MA"):
      import MA
      if (return_info == True):
       return MA.array(aret, fill_value=fv_y)
      else:
       return aret[0]
    elif (type_x == "num"):
      import numpy.core.ma
      if (return_info == True):
        return numpy.core.ma.array(aret, fill_value=fv_y)
      else:
        return aret[0]
  else:
    if (return_info == True): 
      return fplib.regline(x,y,fill_value_x,fill_value_y)
    else:
      return aret[0]
