import paf_version
__version__              = paf_version.version
__array_module__         = paf_version.array_module
__array_module_version__ = paf_version.array_module_version
HAS_NUM                  = paf_version.HAS_NUM

import fplib

#
# Test to make sure we can actually load numpy or Numeric, and
# that we are dealing with a reasonable version.
#
if HAS_NUM == 2:
  try:
    import numpy as Numeric
# 
# If we are dealing with a numpy version that is less than 1.0.0, then
# check the version that PAF was built with against this version.
#
    if Numeric.__version__[0] == '0' and \
       Numeric.__version__ < __array_module_version__:
      print 'Warning: your version of numpy may be older than what PAF'
      print 'was built with. You could have compatibility problems.'
      print 'PAF was built with numpy version',__array_module_version__,'and you are'
      print 'importing version',Numeric.__version__
  except ImportError:
    print 'Cannot find numpy, cannot proceed.'
    print 'Perhaps you need to install the Numeric version of PAF instead.'
    exit
else:
  try:
    import Numeric
#
# I decided to comment this section out, because a Numeric 24 version
# of PyNGL seems to work okay with Numeric 23.x (23.8 anyway), and I'm
# assuming this carries over for PAF as well.
#
#    if Numeric.__version__[0] != __array_module_version__[0]:
#      print 'Warning: your version of Numeric is different from what PyNGL'
#      print 'was built with. You may have compatibility problems.'
#      print 'PyNGL was built with Numeric version',__array_module_version__
#      print 'and you are importing version',Numeric.__version__
  except ImportError:
    print 'Cannot find Numeric, cannot proceed'
    print 'Perhaps you need to install the numpy version of PAF instead.'
    exit

#
#  get_ma_fill_value(arr)
#     input: 
#       arr - any Python object
#     output:
#       Two values: type, fill_value
#         if HAS_NUM == 1:
#           if arr is a Numeric masked array:
#             type = "MA" and fill_value is the fill value
#           else arr is a numpy masked array:
#             type = "num", fill_value is the fill value,
#                     warning is printed
#         elif HAS_NUM == 2:
#           if arr is a numpy masked array:
#             type = "num" and fill_value is the fill value
#           else arr is a Numeric masked array:
#             type = "MA", fill_value is the fill value,
#                     warning is printed
#
def get_ma_fill_value(arr):
#
# If HAS_NUM = 1, this means the Numeric version of this module
# was imported, and hence we should try the Numeric MA module first.
#
  if HAS_NUM == 1:
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
#
#  If not a Numeric masked array, try for NumPy masked array.
#  The user really shouldn't be doing this, however, b/c
#  they've imported the Numeric version of the module.
#  We will print a warning in this case.
#
      try:
        import numpy.core.ma
        if (type(arr) == type(numpy.core.ma.array([0]))):
          "Warning: you've imported the Numeric version of PAF, but are using numpy arrays"
          return "num",arr.fill_value()
      except:
        return None,None
#
# If HAS_NUM=2, this means the NumPy version of this module
# was imported, and hence we should try the NumPy MA module first.
#
  elif HAS_NUM == 2:
    "Try numpy"
    try:
      import numpy.core.ma
      if (type(arr) == type(numpy.core.ma.array([0]))):
        return "num",arr.fill_value()
    except:
#
#  If not a NumPy masked array, try for Numeric masked array.
#  The user shouldn't be doing this, however, b/c they've
#  imported the numpy version of the module. We will print a
#  warning.
#
      try:
        import MA
        fv = None
#
#  If arr is a Numeric masked array, return its fill value
#
        if (type(arr) == type(MA.array([0]))):
          "Warning: you've imported the NumPy version of PAF, but are using Numric arrays"
          fv = arr.fill_value()
#
#  For Numeric 24 or later, the fill value is returned as
#  a single-element array.  For Numeric releases prior to
#  24, the fill value is returned as a numeric value, so
#  return it.
#
          try:
            len(fv)
            return "MA",fv[0]
          except:
            return "MA",fv
        else:
          return None,None
      except:
        return None,None
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

