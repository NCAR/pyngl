import sys
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
# However, loading a Numeric module doesn't preclude the user from
# using numpy arrays (and vice versa). We don't recommend this, but
# we believe that it can be allowed.
#
# Keep track of which array module we were successful in loading,
# as we will use this information later for determining whether
# to create Numeric or numpy arrays to pass to the extension module
# routiines.
#
recommend_numeric = False
if HAS_NUM == 2:
  try:
    import numpy
  except ImportError:
    print 'Cannot find numpy.'
    print 'Will try Numeric, even though you have loaded numpy'
    print 'version of this module.'
    try:
      import Numeric
      recommend_numeric = True
    except ImportError:
      print 'Cannot find Numeric either, cannot proceed.'
      sys.exit()
else:
  try:
    import Numeric
    recommend_numeric = True
  except ImportError:
    print 'Cannot find Numeric.'
    print 'Will try numpy, even though you have loaded Numeric'
    print 'version of this module.'
    try:
      import numpy
    except ImportError:
      print 'Cannot find numpy either, cannot proceed.'
      sys.exit()

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

#
# This function returns True if it encounters a Python scalar.
#
def is_python_scalar(arg):
  import types
  if (type(arg)==types.IntType or type(arg)==types.LongType or \
      type(arg)==types.FloatType):
    return True
  else:
    return False

# 
# This function returns True if we have a Numeric array.
#
def is_numeric(arg):
  try:
    import Numeric
    if (type(arg) == type(Numeric.array([0]))):
      return True
    else:
      return False
  except:
    return False

# 
# This function returns True if we have a numpy scalar or array.
#
def is_numpy(arg):
  try:
    import numpy
    if (type(arg) == type(numpy.array([0])) or isinstance(arg,numpy.generic)):
      return True
    else:
      return False
  except:
    return False

#
# This function returns True if it encounters a numeric scalar.
# A numeric scalar is a numeric array with 0 dimensions.
#
def is_numeric_scalar(arg):
  if (is_numeric(arg) and (len(arg.shape) == 0)):
    return True
  else:
    return False

#
# This function returns True if it encounters a numpy scalar.
# A numpy scalar can either be a numpy array with 0 dimensions,
# or a numpy scalar, which is a new thing that didn't exist in
# numeric.
#
def is_numpy_scalar(arg):
  try:
    import numpy
    if (type(arg) == type(numpy.array([0]))) and (len(arg.shape) == 0):
      return True
#
# Test for numpy scalar.
#
    elif isinstance(arg,numpy.generic):
      return True
    else:
      return False
  except:
    return False

#
# This function returns True if it's a Python scalar, a
# numpy scalar, or a numeric scalar.
#
def is_scalar(arg):
  return is_numeric_scalar(arg) or is_numpy_scalar(arg) or \
         is_python_scalar(arg)
#
# Special function to deal with values that may come in as
# a scalar (as defined by the "is_scalar" function above)
# that need to be converted to something that won't
# register as having 0 dimensions.  We do this by 
# promoting it to a Numeric array.  Note that "Numeric"
# may actually be "numpy", depending on whether the
# Numeric or numpy module was imported. 
#
def promote_scalar(x):
  if is_scalar(x):
    if recommend_numeric:
      import Numeric
      return Numeric.array([x])
    else:
      import numpy
      return numpy.array([x])
  else:
    return x

def chiinv(x,y):
#
# Promote x and y to Numeric (or numpy) arrays that have at least
# a dimension of 1.
#
  x2 = promote_scalar(x)
  y2 = promote_scalar(y)
#
# Determine what kind of array to return. This is dependent on the
# types of the arguments passed to chiinv, and not which fplib
# module was loaded. Note that numpy is favored over Numeric.
#
  if is_numpy(x) or is_numpy(y):
    import numpy
    return numpy.array(fplib.chiinv(x2,y2))
  elif is_numeric(x) or is_numeric(y):
    import Numeric
    return Numeric.array(fplib.chiinv(x2,y2))
  else:
    return fplib.chiinv(x2,y2)
    
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
    return fplib.linmsg(promote_scalar(x),end_pts_msg,max_msg,fill_value)

def regline(x, y, fill_value_x=1.e20, fill_value_y=1.e20, 
                    return_info=True):
  type_x, fv_x = get_ma_fill_value(x)
  type_y, fv_y = get_ma_fill_value(y)
#
#  x and y both masked arrays.
#
  if (fv_x != None and fv_y != None):
    result = fplib.regline(x.filled(fv_x), y.filled(fv_y), fv_x, fv_y)
    if (return_info == True): 
      return result
    else:
      return result[0]
#
#  x is a masked array, y is not.
#
  elif (fv_x != None and fv_y == None):
    result = fplib.regline(x.filled(fv_x), y, fv_x, fill_value_y)
    if (return_info == True): 
      return result
    else:
      return result[0]
#
#  x is not a masked array, y is.
#
  elif (fv_x == None and fv_y != None):
    result = fplib.regline(x, y.filled(fv_y), fill_value_x, fv_y)
    if (return_info == True): 
      return result
    else:
      return result[0]
#
#  Neither x nor y is a masked array.
#
  else:
    result = fplib.regline(x,y,fill_value_x,fill_value_y)
    if (return_info == True): 
      return result
    else:
      return result[0]
