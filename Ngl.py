from hlu import *
import hlu
import sys, os
import site
import types
import string
import Numeric

first_call_to_open_wks = 0

class Resources:
  pass

class PlotIds:
  pass

def int_id(plot_id):
#
#  Convert PlotIds objects to integer plot ids.
#
  if (type(plot_id) == type(1)):
#  Is an integer.
    return plot_id
  elif (type(plot_id) == type([1])):
#  Is a list.
    return plot_id[0]
  elif (isinstance(plot_id,PlotIds)):
#  Is a PlotIds class instance.
    if (type(plot_id.base[0]) != type(1)):
      print "plot id is not valid"
      return None
    return plot_id.base[0]
  else:
    print "plot id is not valid"
    return None

def overlay(plot_id1,plot_id2):
  NhlAddOverlay(int_id(plot_id1),int_id(plot_id2),-1)

def lst2pobj(lst):
#
#  Converts a list of object ids returned from a plotting function
#  to a PlotIds object with attrubutes.
#
#  A Python list of PlotIds is in the order:
#
#    base
#    contour
#    vector
#    streamline
#    map
#    xy
#    xydspec
#    text
#    primitive
#    cafield
#    sffield
#    vffield
#
  rval = PlotIds()

  if (lst[0] == None):
    rval.nbase = 0
  else:
    rval.nbase = len(lst[0])
  rval.base    = lst[0]

  if (lst[1] == None):
    rval.ncontour = 0
  else:
    rval.ncontour = len(lst[1])
  rval.contour    = lst[1]

  if (lst[2] == None):
    rval.nvector = 0
  else:
    rval.nvector = len(lst[2])
  rval.vector    = lst[2]

  if (lst[3] == None):
    rval.nstreamline = 0
  else:
    rval.nstreamline = len(lst[3])
  rval.streamline    = lst[3]

  if (lst[4] == None):
    rval.nmap = 0
  else:
    rval.nmap = len(lst[4])
  rval.map    = lst[4]

  if (lst[5] == None):
    rval.nxy = 0
  else:
    rval.nxy = len(lst[5])
  rval.xy     = lst[5]

  if (lst[6] == None):
    rval.nxydspec = 0
  else:
    rval.nxydspec = len(lst[6])
  rval.xydspec    = lst[6]

  if (lst[7] == None):
    rval.ntext = 0
  else:
    rval.ntext = len(lst[7])
  rval.text    = lst[7]

  if (lst[8] == None):
    rval.nprimitive = 0
  else:
    rval.nprimitive = len(lst[8])
  rval.primitive    = lst[8]

  if (lst[9] == None):
    rval.ncafield = 0
  else:
    rval.ncafield = len(lst[9])
  rval.cafield    = lst[9]

  if (lst[10] == None):
    rval.nsffield = 0
  else:
    rval.nsffield = len(lst[10])
  rval.sffield    = lst[10]

  if (lst[11] == None):
    rval.nvffield = 0
  else:
    rval.nvffield = len(lst[11])
  rval.vffield    = lst[11]

  return rval

def pobj2lst(pobj):
#  
#  A Python list of PlotIds is in the order:
#
#    base
#    contour
#    vector
#    streamline
#    map
#    xy
#    xydspec
#    text
#    primitive
#    cafield
#    sffield
#    vffield
#
#  Converts the attributes of a PlotId object to a Python list.
#
  if (pobj == 0):
    return [None,None,None,None,None,None,None,None,None,None,None,None]
  else:
    return [pobj.base,pobj.contour,pobj.vector,pobj.streamline,pobj.map, \
           pobj.xy,pobj.xydspec,pobj.text,pobj.primitive,pobj.cafield,   \
           pobj.sffield,pobj.vffield]

def pseq2lst(pseq):
#
#  Takes a list of Python plot objects and converts it to
#  a list of lists that will be converted to a list of PlotId
#  structures in the panel argument.
#
  lst = []
  for i in range(len(pseq)):
    lst.append(pobj2lst(pseq[i]))
  return lst

def set_spc_res(resource_name,value):
#
#  Change True and False values to 1 and 0 and leave all other
#  values unchaged.
#
  lval = value
  if (value == True):
    lval = 1
  elif (value == False):
    lval = 0

#
#  Set the special resource values.
#
  if (resource_name   == "Maximize"):
    set_nglRes_i(0, lval) 
  elif (resource_name == "Draw"):
    set_nglRes_i(1, lval) 
  elif (resource_name == "Frame"):
    set_nglRes_i(2, lval) 
  elif (resource_name == "Scale"):
    set_nglRes_i(3, lval) 
  elif (resource_name == "Debug"):
    set_nglRes_i(4, lval) 
  elif (resource_name == "PaperOrientation"):
    if(type(lval) == types.StringType):
      if(string.lower(lval) == "portrait"):
        set_nglRes_i(5, 0) 
      elif(string.lower(lval) == "landscape"):
        set_nglRes_i(5, 6) 
      elif(string.lower(lval) == "auto"):
        set_nglRes_i(5, 3) 
      else:
        print "set_spc_res: Unknown value for " + resource_name
    else:
      set_nglRes_i(5, lval) 
  elif (resource_name == "PaperWidth"):
    set_nglRes_f(6, lval) 
  elif (resource_name == "PaperHeight"):
    set_nglRes_f(7, lval) 
  elif (resource_name == "PaperMargin"):
    set_nglRes_f(8, lval) 
  elif (resource_name == "PanelCenter"):
    set_nglRes_i(9, lval) 
  elif (resource_name == "PanelRowSpec"):
    set_nglRes_i(10, lval) 
  elif (resource_name == "PanelXWhiteSpacePercent"):
    set_nglRes_f(11, lval) 
  elif (resource_name == "PanelYWhiteSpacePercent"):
    set_nglRes_f(12, lval) 
  elif (resource_name == "PanelBoxes"):
    set_nglRes_i(13, lval) 
  elif (resource_name == "PanelLeft"):
    set_nglRes_f(14, lval) 
  elif (resource_name == "PanelRight"):
    set_nglRes_f(15, lval) 
  elif (resource_name == "PanelBottom"):
    set_nglRes_f(16, lval) 
  elif (resource_name == "PanelTop"):
    set_nglRes_f(17, lval) 
  elif (resource_name == "PanelInvsblTop"):
    set_nglRes_f(18, lval) 
  elif (resource_name == "PanelInvsblLeft"):
    set_nglRes_f(19, lval) 
  elif (resource_name == "PanelInvsblRight"):
    set_nglRes_f(20, lval) 
  elif (resource_name == "PanelInvsblBottom"):
    set_nglRes_f(21, lval) 
  elif (resource_name == "PanelSave"):
    set_nglRes_i(22, lval) 
  elif (resource_name == "SpreadColors"):
    set_nglRes_i(23, lval) 
  elif (resource_name == "SpreadColorStart"):
    set_nglRes_i(24, lval) 
  elif (resource_name == "SpreadColorEnd"):
    set_nglRes_i(25, lval) 
  elif (resource_name == "PanelLabelBarOrientation"):
    set_nglRes_i(26, lval) 
  elif (resource_name == "PanelLabelBar" and len(resource_name) == 13):
    set_nglRes_i(27, lval) 
  elif (resource_name == "PanelLabelBarXF"):
    set_nglRes_f(28, lval) 
  elif (resource_name == "PanelLabelBarYF"):
    set_nglRes_f(29, lval) 
  elif (resource_name == "PanelLabelBarLabelFontHeightF"):
    set_nglRes_f(30, lval) 
  elif (resource_name == "PanelLabelBarWidthF"):
    set_nglRes_f(31, lval) 
  elif (resource_name == "PanelLabelBarHeightF"):
    set_nglRes_f(32, lval) 
  elif (resource_name == "PanelLabelBarOrthogonalPosF"):
    set_nglRes_f(33, lval) 
  elif (resource_name == "PanelLabelBarParallelPosF"):
    set_nglRes_f(34, lval) 
  elif (resource_name == "PanelLabelBarPerimOn"):
    set_nglRes_i(35, lval) 
  elif (resource_name == "PanelLabelBarAlignment"):
    set_nglRes_i(36, lval) 
  elif (resource_name == "PanelLabelBarLabelAutoStride"):
    set_nglRes_i(37, lval) 
  elif (resource_name == "PanelFigureStrings" and len(resource_name) == 18):
    set_nglRes_c(38, lval) 
  elif (resource_name == "PanelFigureStringsCount"):
    set_nglRes_i(39, lval) 
  elif (resource_name == "PanelFigureStringsJust"):
    set_nglRes_i(40, lval) 
  elif (resource_name == "PanelFigureStringsOrthogonalPosF"):
    set_nglRes_f(41, lval) 
  elif (resource_name == "PanelFigureStringsParallelPosF"):
    set_nglRes_f(42, lval) 
  elif (resource_name == "PanelFigureStringsPerimOn"):
    set_nglRes_i(43, lval) 
  elif (resource_name == "PanelFigureStringsBackgroundFillColor"):
    set_nglRes_i(44, lval) 
  elif (resource_name == "PanelFigureStringsFontHeightF"):
    set_nglRes_f(45, lval) 
  else:
    print "set_spc_res: Unknown special resource ngl" + resource_name

def check_res_value(resvalue,strvalue,intvalue):
#
#  Function for checking a resource value that can either be of
#  type string or integer.
#
  if( (type(resvalue) == types.StringType and \
     string.lower(resvalue) == string.lower(strvalue)) or \
     (type(resvalue) == types.IntType and resvalue == intvalue)):
    return(True)
  else:
    return(False)

def set_contour_res(reslist,reslist1):
#
#  Set some contour resources of which we either don't like the NCL
#  defaults, or we want to set something on behalf of the user.
#
  if(reslist.has_key("cnFillOn") and reslist["cnFillOn"] > 0):
    if ( not (reslist.has_key("cnInfoLabelOn"))):
      reslist1["cnInfoLabelOn"] = False
    if ( not (reslist.has_key("pmLabelBarDisplayMode"))):
      reslist1["pmLabelBarDisplayMode"] = "Always"
#
#  Check for "plural" resources that only take effect if their
#  corresponding "Mono" resource is set to False, and set the
#  Mono resource on behalf of the user.
#
  if(reslist.has_key("cnLineDashPatterns")):
    if ( not (reslist.has_key("cnMonoLineDashPattern"))):
      reslist1["cnMonoLineDashPattern"] = False
  if(reslist.has_key("cnLineColors")):
    if (not (reslist.has_key("cnMonoLineColor"))):
      reslist1["cnMonoLineColor"] = False
  if(reslist.has_key("cnLineThicknesss")):
    if (not (reslist.has_key("cnMonoLineThickness"))):
      reslist1["cnMonoLineThickness"] = False
  if(reslist.has_key("cnLevelFlags")):
    if (not (reslist.has_key("cnMonoLevelFlag"))):
      reslist1["cnMonoLevelFlag"] = False
  if(reslist.has_key("cnFillPatterns")):
    if (not (reslist.has_key("cnMonoFillPattern"))):
      reslist1["cnMonoFillPattern"] = False
  if(reslist.has_key("cnFillScales")):
    if (not (reslist.has_key("cnMonoFillScale"))):
      reslist1["cnMonoFillScale"] = False
  if(reslist.has_key("cnLineLabelFontColors")):
    if (not (reslist.has_key("cnMonoLineLabelFontColor"))):
      reslist1["cnMonoLineLabelFontColor"] = False

def set_vector_res(reslist,reslist1):
#
#  Set some vector resources of which we either don't like the NCL
#  defaults, or we want to set something on behalf of the user.
#
# Vectors can be colored one of two ways, either with colored line
# vectors or filled colored vectors, or wind barbs.  Any one of these
# would warrant a labelbar.
#
# Don't bother setting the the vcMonoLineArrowColor,
# vcMonoFillArrowEdgeColor, vcMonoFillArrowFillColor, or 
# vcMonoWindBarbColor resources to False if vcLevelColors is
# set, because it all depends on if vcGlyphStyle is set a certain way.
# Put the responsibility on the user
#
  if( (reslist.has_key("vcMonoLineArrowColor") and
       check_res_value(reslist["vcMonoLineArrowColor"],"False",0)) or
      (reslist.has_key("vcMonoFillArrowFillColor") and
       check_res_value(reslist["vcMonoFillArrowFillColor"],"False",0)) or
      (reslist.has_key("vcMonoFillArrowEdgeColor") and
       check_res_value(reslist["vcMonoFillArrowEdgeColor"],"False",0)) or
      (reslist.has_key("vcMonoWindBarbColor") and
       check_res_value(reslist["vcMonoWindBarbColor"],"False",0))):
    if ( not (reslist.has_key("pmLabelBarDisplayMode"))):
      reslist1["pmLabelBarDisplayMode"] = "Always"


def set_streamline_res(reslist,reslist1):
#
#  Set some streamline resources of which we either don't like the NCL
#  defaults, or we want to set something on behalf of the user.
#
# stMonoLineColor is different, because there is no stLineColors resource.
# Instead, there's an stLevelColors resource, and this is the one we
# need to set here.
#
  if(reslist.has_key("stLevelColors")):
    if (not (reslist.has_key("stMonoLineColor"))):
      reslist1["stMonoLineColor"] = False

def set_map_res(reslist,reslist1):
#
# Turn on map tickmarks.
#
  if ( not (reslist.has_key("pmTickMarkDisplayMode"))):
    reslist1["pmTickMarkDisplayMode"] = "Always"
  if(reslist.has_key("mpFillPatterns")):
    if (not (reslist.has_key("mpMonoFillPattern"))):
      reslist1["mpMonoFillPattern"] = False
  if(reslist.has_key("mpFillScales")):
    if (not (reslist.has_key("mpMonoFillScale"))):
      reslist1["mpMonoFillScale"] = False
  

def set_labelbar_res(reslist,reslist1):
#
# Set some labelbar resources of which we don't like the NCL
# defaults.
# 
  if ( not (reslist.has_key("lbPerimOn"))):
    reslist1["lbPerimOn"] = False
  if ( not (reslist.has_key("lbLabelAutoStride"))):
    reslist1["lbLabelAutoStride"] = True
  if(reslist.has_key("lbLabelFontHeightF")):
    if ( not (reslist.has_key("lbAutoManage"))):
      reslist1["lbAutoManage"] = False
  if(reslist.has_key("lbFillScales")):
    if (not (reslist.has_key("lbMonoFillScale"))):
      reslist1["lbMonoFillScale"] = False
  if(reslist.has_key("lbOrientation")):
    if ( not (reslist.has_key("pmLabelBarSide"))):
      if(check_res_value(reslist["lbOrientation"],"Horizontal",0)):
        reslist1["pmLabelBarSide"] = "Bottom"
      if(check_res_value(reslist["lbOrientation"],"Vertical",1)):
        reslist1["pmLabelBarSide"] = "Right"

def set_legend_res(reslist,reslist1):
#
# Set some legend resources of which we don't like the NCL
# defaults.  These are mostly the Mono resources which we default to
# False if the corresponding parallel version of this resource is set.
#
# We may not be doing anything with these yet, since
# we don't have a legend function yet.
# 
  if(reslist.has_key("lgLabelFontHeightF")):
    if ( not (reslist.has_key("lgAutoManage"))):
      reslist1["lgAutoManage"] = False
  if(reslist.has_key("lgItemTypes")):
    if (not (reslist.has_key("lgMonoItemType"))):
      reslist1["lgMonoItemType"] = False
  if(reslist.has_key("lgLineDashSegLens")):
    if (not (reslist.has_key("lgMonoLineDashSegLen"))):
      reslist1["lgMonoLineDashSegLen"] = False
  if(reslist.has_key("lgLineThicknesses")):
    if (not (reslist.has_key("lgMonoLineThickness"))):
      reslist1["lgMonoLineThickness"] = False
  if(reslist.has_key("lgMarkerThicknesses")):
    if (not (reslist.has_key("lgMonoMarkerThickness"))):
      reslist1["lgMonoMarkerThickness"] = False
  if(reslist.has_key("lgLineLabelFontHeights")):
    if (not (reslist.has_key("lgMonoLineLabelFontHeight"))):
      reslist1["lgMonoLineLabelFontHeight"] = False
  if(reslist.has_key("lgMarkerSizes")):
    if (not (reslist.has_key("lgMonoMarkerSize"))):
      reslist1["lgMonoMarkerSize"] = False

def change_workstation(obj,wks):
  return NhlChangeWorkstation(int_id(obj),wks)

def end():
  NhlClose()

#
#  Get indices of a list where the list values are true.
#
def ind(seq):
  inds = []
  for i in xrange(len(seq)):
    if (seq[i] != 0):
      inds.append(i)
  return(inds)

#
# Add a cyclic point in the x dimension (longitude dimension) to
# a 2D array. If there is also a 1D lon coordinate array, add 360 to 
# create the cyclic point.
#
def add_cyclic(data,lon_coord=None):
#
# Check input data to make sure it is 2D.
#
  dims = data.shape
  if (len(dims) != 2):
    print "add_cyclic: input must be a 2-dimensional array."
    sys.exit()

  ny  = dims[0]
  nx  = dims[1]
  nx1 = nx + 1

#
# Test longitude array, if it exists.
#
  if(lon_coord != None):
    lon_coord_dims = lon_coord.shape
    lon_coord_rank = len(lon_coord_dims)
#
# Longitude coordindate array must be 1D.
#
    if (lon_coord_rank != 1):
      print "add_cyclic: longitude coordinate array must be a 1-dimensional."
      sys.exit()
#
# Check dimension size against data array.
#
    nlon = lon_coord_dims[0]
    if (nlon != nx):
      print "add_cyclic: longitude coordinate array must be the same length as the rightmost dimension of the data array."
      sys.exit()

#
# Create the new data array with one extra value in the X direction.
#
  newdata         = Numeric.zeros((ny,nx1),data.typecode())
  newdata[:,0:nx] = data
  newdata[:,nx]   = data[:,0]

#
# Add 360 to the longitude value in order to make it cyclic.
#
  if(lon_coord != None):
    newloncoord       = Numeric.zeros(nx1,lon_coord.typecode())
    newloncoord[0:nx] = lon_coord
    newloncoord[nx]   = lon_coord[0] + 360

    return newdata,newloncoord
  else:
    return newdata

def ftcurv(x,y,xo):
  if ( ((type(x) == types.ListType) or (type(x) == types.TupleType)) ):
    dsizes_x = len(x)
  elif (type(x) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_x = x.shape[0]
  else:
    print \
     "ftcurv: type of argument 1 must be one of: list, tuple, or Numeric array"
    return None
  if ( ((type(y) == types.ListType) or (type(y) == types.TupleType)) ):
    dsizes_y = len(y)
  elif (type(y) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_y = y.shape[0]
  else:
    print \
     "ftcurv: type of argument 2 must be one of: list, tuple, or Numeric array"
    return None
  if (dsizes_x != dsizes_y):
    print "ftcurv: first and second arguments must be the same length."
    return None

  if ( ((type(xo) == types.ListType) or (type(xo) == types.TupleType)) ):
    dsizes_xo = len(xo)
  elif (type(xo) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_xo = xo.shape[0]

  status,yo = ftcurvc(dsizes_x,x,y,dsizes_xo,xo)
  if (status == 1):
    print "ftcurv: input array must have at least three elements."
    return None
  elif (status == 2): 
    print "ftcurv: input array values must be strictly increasing."
    return None
  else:
    del status
    return yo

def ftcurvp(x,y,p,xo):
  if ( ((type(x) == types.ListType) or (type(x) == types.TupleType)) ):
    dsizes_x = len(x)
  elif (type(x) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_x = x.shape[0]
  else:
    print \
     "ftcurvp: type of argument 1 must be one of: list, tuple, or Numeric array"
    return None
  if ( ((type(y) == types.ListType) or (type(y) == types.TupleType)) ):
    dsizes_y = len(y)
  elif (type(y) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_y = y.shape[0]
  else:
    print \
     "ftcurvp: type of argument 2 must be one of: list, tuple, or Numeric array"
    return None
  if (dsizes_x != dsizes_y):
    print "ftcurvp: first and second arguments must be the same length."
    return None

  if ( ((type(xo) == types.ListType) or (type(xo) == types.TupleType)) ):
    dsizes_xo = len(xo)
  elif (type(xo) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_xo = xo.shape[0]

  status,yo = ftcurvpc(dsizes_x,x,y,p,dsizes_xo,xo)
  if (status == 1):
    print "ftcurvp: input array must have at least three elements."
    return None
  elif (status == 2):
    print "ftcurvp: the period is strictly less than the span of the abscissae."
    return None
  else:
    del status
    return yo

def ftcurvpi(xl, xr, p, x, y):
  if ( ((type(x) == types.ListType) or (type(x) == types.TupleType)) ):
    dsizes_x = len(x)
  elif (type(x) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_x = x.shape[0]
  else:
    print \
     "ftcurvpi: type of argument 4 must be one of: list, tuple, or Numeric array"
    return None
  if ( ((type(y) == types.ListType) or (type(y) == types.TupleType)) ):
    dsizes_y = len(y)
  elif (type(y) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_y = y.shape[0]
  else:
    print \
     "ftcurvpi: type of argument 5 must be one of: list, tuple, or Numeric array"
    return None
  if (dsizes_x != dsizes_y):
    print "ftcurvpi: fourth and fifth arguments must be the same length."
    return None

  return (ftcurvpic(xl,xr,p,dsizes_x,x,y)[1])

def natgrid(x,y,z,xo,yo):
  if ( ((type(x) == types.ListType) or (type(x) == types.TupleType)) ):
    dsizes_x = len(x)
  elif (type(x) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_x = x.shape[0]
  else:
    print \
     "natgrid: type of argument 1 must be one of: list, tuple, or Numeric array"
    return None

  if ( ((type(xo) == types.ListType) or (type(xo) == types.TupleType)) ):
    dsizes_xo = len(xo)
  elif (type(xo) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_xo = xo.shape[0]
  else:
    print \
     "natgrid: type of argument 4 must be one of: list, tuple, or Numeric array"
    return None

  if ( ((type(yo) == types.ListType) or (type(yo) == types.TupleType)) ):
    dsizes_yo = len(yo)
  elif (type(yo) == type(Numeric.array([0],Numeric.Int0))):
    dsizes_yo = yo.shape[0]
  else:
    print \
     "natgrid: type of argument 5 must be one of: list, tuple, or Numeric array"
    return None

  ier,zo = \
     natgridc(dsizes_x,x,y,z,dsizes_xo,dsizes_yo,xo,yo,dsizes_xo,dsizes_yo)

  if (ier != 0):
    print "natgrid: error number %d returned, see error table." % (ier)
    del ier
    return None
  else:
    return zo

def ncargpath(type):
  return NGGetNCARGEnv(type)

def crt_dict(resource_i):
  dic = {}
  if (resource_i == None):
    return(dic)
  for t in dir(resource_i):
    if (t[0:2] != '__'):        # skip any Python-supplied attributes.
      dic[t] = getattr(resource_i,t)
  return(dic)

def set_spc_defaults(type):
#
#  Type = 1 sets the ngl special resources for plotting functions.
#  Type = 0 sets the ngl special resources for text/poly resources
#
  if (type == 1):
    set_nglRes_i(0, 1)  # nglMaximize
    set_nglRes_i(1, 1)  # nglDraw
    set_nglRes_i(2, 1)  # nglFrame
  elif (type == 0):
    set_nglRes_i(0, 0)  # nglMaximize
    set_nglRes_i(1, 1)  # nglDraw
    set_nglRes_i(2, 0)  # nglFrame
#
  set_nglRes_i( 3, 1)   # nglScale
  set_nglRes_i( 4, 0)   # nglDebug
  set_nglRes_i( 5, -1)  # nglPaperOrientation
  set_nglRes_f( 6, 8.5) # nglPaperWidth
  set_nglRes_f( 7, 11.) # nglPaperHeight
  set_nglRes_f( 8, 0.5) # nglPaperMargin
  set_nglRes_i( 9, 1)   # nglPanelCenter
  set_nglRes_i(10, 0)   # nglPanelRowSpec
  set_nglRes_f(11, 1.)  # nglPanelXWhiteSpacePercent
  set_nglRes_f(12, 1.)  # nglPanelYWhiteSpacePercent
  set_nglRes_i(13, 0)   # nglPanelBoxes
  set_nglRes_f(14, 0.)  # nglPanelLeft
  set_nglRes_f(15, 1.)  # nglPanelRight
  set_nglRes_f(16, 0.)  # nglPanelBottom
  set_nglRes_f(17, 1.)  # nglPanelTop
  set_nglRes_f(18, -999.)  # nglPanelInvsblTop
  set_nglRes_f(19, -999.)  # nglPanelInvsblLeft
  set_nglRes_f(20, -999.)  # nglPanelInvsblRight
  set_nglRes_f(21, -999.)  # nglPanelInvsblBottom
  set_nglRes_i(22, 0)   # nglPanelSave
  set_nglRes_i(23, 1)   # nglSpreadColors
  set_nglRes_i(24, 2)  # nglSpreadColorStart
  set_nglRes_i(25, -1)   # nglSpreadColorEnd
  set_nglRes_i(26, 0)   # nglPanelLabelBarOrientation
  set_nglRes_i(27, 0)   # nglPanelLabelBar
  set_nglRes_f(28, -999.)   # nglPanelLabelBarXF
  set_nglRes_f(29, -999.)   # nglPanelLabelBarYF
  set_nglRes_f(30, -999.)   # nglPanelLabelBarLabelFontHeightF
  set_nglRes_f(31, -999.)   # nglPanelLabelBarWidthF
  set_nglRes_f(32, -999.)   # nglPanelLabelBarHeightF
  set_nglRes_f(33, -999.)   # nglPanelLabelBarOrthogonalPosF
  set_nglRes_f(34, -999.)   # nglPanelLabelBarParallelPosF
  set_nglRes_i(35, 0)   # nglPanelLabelBarPerimOn
  set_nglRes_i(36, 1)   # nglPanelLabelBarAlignment
  set_nglRes_i(37, 1)   # nglPanelLabelBarLabelAutoStride
  set_nglRes_c(38, [])  # nglPanelFigureStrings
  set_nglRes_i(39, 0)   # nglPanelFigureStringsCount
  set_nglRes_i(40, 8)   # nglPanelFigureStringsJust
  set_nglRes_f(41, -999.)   # nglPanelFigureStringsOrthogonalPosF
  set_nglRes_f(42, -999.)   # nglPanelFigureStringsParallelPosF
  set_nglRes_i(43, 1)   # nglPanelFigureStringsPerimOn
  set_nglRes_i(44, 0)   # nglPanelFigureStringsBackgroundFillColor
  set_nglRes_f(45, -999.)   # nglPanelFigureStringsFontHeightF

def open_wks(wk_type,wk_name,wk_rlist=None):
  global first_call_to_open_wks
  rlist = crt_dict(wk_rlist)

# 
# Initialize the special resource values, and make sure 
# NCARG_NCARG environment variable is set.
#
  if (first_call_to_open_wks == 0):
#
#  Find the root directory that contains the supplemental PyNGL files,
#  like fontcaps, colormaps, and map databases. The default is to look
#  in site-packages/PyNGL/ncarg. Otherwise, check the NCARG_PYNGL
#  environment variable, and then ncargpath("ncarg").
#
    pkgs_pth    = site.sitedirs[0]
    pyngl1_dir  = pkgs_pth + "/PyNGL/ncarg"
    pyngl2_dir  = os.environ.get("NCARG_PYNGL")
    pyngl3_dir  = ncargpath("ncarg")
    ncarg_ncarg = None

    if (os.path.exists(pyngl1_dir)):
      ncarg_ncarg = pyngl1_dir
    else:
      if (pyngl2_dir != None and os.path.exists(pyngl2_dir)):
        ncarg_ncarg = pyngl2_dir
      else:
        if (pyngl3_dir != None and os.path.exists(pyngl3_dir)):
          ncarg_ncarg = pyngl3_dir

#
# Only print out a message about pyngl1_dir, because the other two
# directories are just shots in the dark.
#
    if (ncarg_ncarg == None):
      print pyngl1_dir + " does not exist and cannot"
      print "find alternative directory for PyNGL supplemental files."
      sys.exit()
    else:
#
#  Make sure NCARG_NCARG is set.
#
      os.environ["NCARG_NCARG"] = ncarg_ncarg

    set_spc_defaults(1)
    first_call_to_open_wks = first_call_to_open_wks + 1

#
#  Lists of triplets for color tables must be numeric arrays.
#
  if (rlist.has_key("wkColorMap")):
#
#  Type of the elements of the color map must be array and not list.
#
    if type(rlist["wkColorMap"][0]) == type([0]):
      print "opn_wks: lists of triplets for color tables must be Numeric arrays"
      return None

#
#  Call the wrapped function and return.
#
  iopn = open_wks_wrap(wk_type,wk_name,rlist)
  del rlist
  return(iopn)

def draw_colormap(wks):
  draw_colormap_wrap(wks)

def panel(wks,plots,dims,rlistc=None):
  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
  rlist1 = {}
  rlist2 = {}
  for key in rlist.keys():
    if (key[0:3] == "ngl"):
      if (key[0:21] == "nglPanelFigureStrings" and len(key) == 21):
        set_spc_res(key[3:],rlist[key])
        set_spc_res("PanelFigureStringsCount",len(rlist[key]))
      elif (key[0:25] == "nglPanelFigureStringsJust"):
        if(check_res_value(rlist[key],"TopLeft",0)):
          set_spc_res(key[3:],0)
        elif(check_res_value(rlist[key],"CenterLeft",1)): 
          set_spc_res(key[3:],1)
        elif(check_res_value(rlist[key],"BottomLeft",2)): 
          set_spc_res(key[3:],2)
        elif(check_res_value(rlist[key],"TopCenter",3)): 
          set_spc_res(key[3:],3)
        elif(check_res_value(rlist[key],"CenterCenter",4)): 
          set_spc_res(key[3:],4)
        elif(check_res_value(rlist[key],"BottomCenter",5)): 
          set_spc_res(key[3:],5)
        elif(check_res_value(rlist[key],"TopRight",6)):
          set_spc_res(key[3:],6)
        elif(check_res_value(rlist[key],"CenterRight",7)): 
          set_spc_res(key[3:],7)
        elif(check_res_value(rlist[key],"BottomRight",8)): 
          set_spc_res(key[3:],8)
      else:
        set_spc_res(key[3:],rlist[key])
    elif (key[0:2] == "lb"):
      if (key == "lbLabelAlignment"):
        if (check_res_value(rlist[key],"BoxCenters",0)):
          set_spc_res("PanelLabelBarAlignment",0)
        elif (check_res_value(rlist[key],"InteriorEdges",1)):
          set_spc_res("PanelLabelBarAlignment",1)
        elif (check_res_value(rlist[key],"ExternalEdges",2)):
          set_spc_res("PanelLabelBarAlignment",2)
        else:
          set_spc_res("PanelLabelBarAlignment",rlist[key])
      elif (key == "lbPerimOn"):
        if (rlist[key] == 1):
          set_spc_res("PanelLabelBarPerimOn",1)
        elif (rlist[key] == 0):
          set_spc_res("PanelLabelBarPerimOn",0)
        else:
          set_spc_res("PanelLabelBarPerimOn",rlist[key])
      elif (key == "lbLabelAutoStride"):
        if (rlist[key] == 1):
          set_spc_res("PanelLabelBarAutoStride",1)
        elif (rlist[key] == 0):
          set_spc_res("PanelLabelBarAutoStride",0)
        else:
          set_spc_res("PanelLabelBarAutoStride",rlist[key])
      elif (key == "lbLabelFontHeightF"):
        set_spc_res("PanelLabelBarFontHeightF",rlist[key])
      elif (key == "lbOrientation"):
        if (check_res_value(rlist[key],"Vertical",1)):
          set_spc_res("PanelLabelBarOrientation",1)
        elif (check_res_value(rlist[key],"Horizontal",0)):
          set_spc_res("PanelLabelBarOrientation",0)
        else:
          set_spc_res("PanelLabelBarOrientation",rlist[key])
        

      rlist1[key] = rlist[key]
  panel_wrap(wks,pseq2lst(plots),len(plots),dims,len(dims),rlist1,rlist2,pvoid())
  del rlist
  del rlist1

def map(wks,rlistc=None):
  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
  rlist1 = {}
  for key in rlist.keys():
    if (key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]

  set_map_res(rlist,rlist1)           # Set some map resources

  imp = map_wrap(wks,rlist1,pvoid())
  del rlist
  del rlist1
  return(lst2pobj(imp))

def poly(wks,plot,x,y,ptype,is_ndc,rlistc=None):
  set_spc_defaults(0)
  rlist = crt_dict(rlistc)  
  rlist1 = {}
  for key in rlist.keys():
    if (key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]
  ply = poly_wrap(wks,pobj2lst(plot),x,y,"double","double",len(x),0,0,pvoid(), \
                      pvoid(),ptype,rlist1,pvoid())
  del rlist
  del rlist1
  return None

def polymarker_ndc(wks,x,y,rlistc=None):
  return(poly(wks,0,x,y,NhlPOLYMARKER,1,rlistc))

def polygon_ndc(wks,x,y,rlistc=None):
  return(poly(wks,0,x,y,NhlPOLYGON,1,rlistc))

def polyline_ndc(wks,x,y,rlistc=None):
  return(poly(wks,0,x,y,NhlPOLYLINE,1,rlistc))

def polymarker(wks,plot,x,y,rlistc=None):  # plot converted in poly
  return(poly(wks,plot,x,y,NhlPOLYMARKER,0,rlistc))

def polygon(wks,plot,x,y,rlistc=None):
  return(poly(wks,plot,x,y,NhlPOLYGON,0,rlistc))

def polyline(wks,plot,x,y,rlistc=None):
  return(poly(wks,plot,x,y,NhlPOLYLINE,0,rlistc))

def add_poly(wks,plot,x,y,ptype,rlistc=None):
  rlist = crt_dict(rlistc)  
  rlist1 = {}
  for key in rlist.keys():
    if (key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]
  ply = add_poly_wrap(wks,pobj2lst(plot), x,y, "double","double",
            len(x),0,0,pvoid(), pvoid(),ptype,rlist1,pvoid())
  del rlist
  del rlist1
  return(lst2pobj(ply))

def add_polymarker(wks,plot,x,y,rlistc=None):
  return(add_poly(wks,plot,x,y,NhlPOLYMARKER,rlistc))

def add_polygon(wks,plot,x,y,rlistc=None):
  return(add_poly(wks,plot,x,y,NhlPOLYGON,rlistc))

def add_polyline(wks,plot,x,y,rlistc=None):
  return(add_poly(wks,plot,x,y,NhlPOLYLINE,rlistc))

def contour_map(wks,array,rlistc=None):
#
#  Make sure the array is 2D.
#
  if (len(array.shape) != 1 and len(array.shape) != 2):
    print "contour_map - array must be 1D or 2D"
    return None

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField, MapPlot, and ContourPlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "sf"):
      rlist1[key] = rlist[key]
    elif( (key[0:2] == "mp") or (key[0:2] == "vp") or (key[0:3] == "pmA") or \
          (key[0:3] == "pmO") or (key[0:3] == "pmT") or (key[0:2] == "tm") ):
      rlist2[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist3[key] = rlist[key]

  set_map_res(rlist,rlist2)           # Set some map resources
  set_contour_res(rlist,rlist3)       # Set some contour resources
  set_labelbar_res(rlist,rlist3)      # Set some labelbar resources

#
#  Call the wrapped function and return.
#
  if (len(array.shape) == 2):
        icm = contour_map_wrap(wks,array,"double", \
                                array.shape[0],array.shape[1],0, \
                                pvoid(),"",0,pvoid(),"", 0, pvoid(), \
                                rlist1,rlist3,rlist2,pvoid())
  else:
        icm = contour_map_wrap(wks,array,"double", \
                                array.shape[0],-1,0, \
                                pvoid(),"",0,pvoid(),"", 0, pvoid(), \
                                rlist1,rlist3,rlist2,pvoid())

  del rlist
  del rlist1
  del rlist2
  del rlist3
  return(lst2pobj(icm))

def contour(wks,array,rlistc=None):

#
#  Make sure the array is 2D.
#
  if (len(array.shape) != 1 and len(array.shape) != 2):
    print "contour - array must be 1D or 2D"
    return None

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField and those that apply to
#  ContourPlot.
#
  rlist1 = {}
  rlist2 = {}
  for key in rlist.keys():
    if (key[0:2] == "sf"):
      rlist1[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]

  set_contour_res(rlist,rlist2)       # Set some contour resources
  set_labelbar_res(rlist,rlist2)      # Set some labelbar resources
    
#
#  Call the wrapped function and return.
#
  if (len(array.shape) == 2):
    icn = contour_wrap(wks,array,"double",array.shape[0],array.shape[1], \
                           0, pvoid(),"",0,pvoid(),"", 0, pvoid(), rlist1, \
                          rlist2,pvoid())
  else:
    icn = contour_wrap(wks,array,"double",array.shape[0],-1, \
                           0, pvoid(),"",0,pvoid(),"", 0, pvoid(), rlist1, \
                          rlist2,pvoid())

  del rlist
  del rlist1
  del rlist2
  return(lst2pobj(icn))

def xy(wks,xar,yar,rlistc=None):
  set_spc_defaults(1)
#
#  Get input array dimension information.
#
  if ( ((type(xar) == types.ListType) or (type(xar) == types.TupleType)) ):
    ndims_x = 1
    dsizes_x = (len(xar),)
  elif (type(xar) == type(Numeric.array([0],Numeric.Int0))):
    ndims_x = (len(xar.shape))
    dsizes_x = xar.shape
  else:
    print \
      "xy: type of argument 2 must be one of: list, tuple, or Numeric array"
    return None

  if ( ((type(yar) == types.ListType) or (type(yar) == types.TupleType)) ):
    ndims_y = 1
    dsizes_y = (len(yar),)
  elif (type(yar) == type(Numeric.array([0],Numeric.Int0))):
    ndims_y = (len(yar.shape))
    dsizes_y = yar.shape
  else:
    print \
      "xy: type of argument 3 must be one of: list, tuple, or Numeric array"
    return None

  rlist = crt_dict(rlistc)  
 
#
#  Separate the resource dictionary into those resources
#  that apply to various lists.
#
  ca_rlist  = {}
  xy_rlist  = {}
  xyd_rlist = {}
  for key in rlist.keys():
    if (key[0:2] == "ca"):
      ca_rlist[key] = rlist[key]
    elif (key[0:2] == "xy"):
      if (key[0:4] == "xyCo"):
        xy_rlist[key] = rlist[key]
      elif (key[0:3] == "xyX"):
        xy_rlist[key] = rlist[key]
      elif (key[0:3] == "xyY"):
        xy_rlist[key] = rlist[key]
      else:
        xyd_rlist[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      xy_rlist[key] = rlist[key]

#
#  Call the wrapped function and return.
#
  ixy = xy_wrap(wks,xar,yar,"double","double",ndims_x,dsizes_x,ndims_y, \
                    dsizes_y,0,0,pvoid(),pvoid(),ca_rlist,xy_rlist,xyd_rlist,
                    pvoid())

  del rlist
  del ca_rlist
  del xy_rlist
  del xyd_rlist
  return(lst2pobj(ixy))

def y(wks,yar,rlistc=None):
  
#
#  Get input array dimension information.
#
  if ( ((type(yar) == types.ListType) or (type(yar) == types.TupleType)) ):
    ndims_y = 1
    dsizes_y = (len(yar),)
  elif (type(yar) == type(Numeric.array([0],Numeric.Int0))):
    ndims_y = (len(yar.shape))
    dsizes_y = yar.shape
  else:
    print \
      "xy: type of argument 3 must be one of: list, tuple, or Numeric array"
    return None

  if (len(dsizes_y) == 1):
    npts = dsizes_y
  elif (len(dsizes_y) == 2):
    npts = dsizes_y[1]
  else:
    print \
      "y: array can have at most two dimensions"
    return None
    
  return xy(wks,range(0,npts),yar,rlistc)

def streamline(wks,uarray,varray,rlistc=None):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to VectorField and those that apply to
#  StreamlinePlot.
#
  rlist1 = {}
  rlist2 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
    
  set_streamline_res(rlist,rlist2)        # Set some streamline resources

#
#  Call the wrapped function and return.
#
  strm = streamline_wrap(wks,uarray,varray,"double","double",         \
                         uarray.shape[0],uarray.shape[1],0,               \
                         pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(), \
                         rlist1,rlist2,pvoid())
  del rlist
  del rlist1
  del rlist2
  return(lst2pobj(strm))

def streamline_map(wks,uarray,varray,rlistc=None):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to VectorField, MapPlot, and StreamlinePlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif( (key[0:2] == "mp") or (key[0:2] == "vp") or (key[0:3] == "pmA") or \
          (key[0:3] == "pmO") or (key[0:3] == "pmT") or (key[0:2] == "tm")):
      rlist3[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]

  set_map_res(rlist,rlist3)               # Set some map resources
  set_streamline_res(rlist,rlist2)        # Set some streamline resources
    
#
#  Call the wrapped function and return.
#
  strm = streamline_map_wrap(wks,uarray,varray,"double","double",         \
                         uarray.shape[0],uarray.shape[1],0,               \
                         pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(), \
                         rlist1,rlist2,rlist3,pvoid())
  del rlist
  del rlist1
  del rlist2
  del rlist3
  return(lst2pobj(strm))

def vector(wks,uarray,varray,rlistc=None):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to VectorField and those that apply to
#  VectorPlot.
#
  rlist1 = {}
  rlist2 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
    
  set_vector_res(rlist,rlist2)        # Set some vector resources
  set_labelbar_res(rlist,rlist2)      # Set some labelbar resources

#
#  Call the wrapped function and return.
#
  ivct = vector_wrap(wks,uarray,varray,"double","double",         \
                     uarray.shape[0],uarray.shape[1],0,               \
                     pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(), \
                     rlist1,rlist2,pvoid())
  del rlist
  del rlist1
  del rlist2
  return lst2pobj(ivct)

def vector_map(wks,uarray,varray,rlistc=None):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to VectorField, MapPlot, and VectorPlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif( (key[0:2] == "mp") or (key[0:2] == "vp") or (key[0:3] == "pmA") or \
          (key[0:3] == "pmO") or (key[0:3] == "pmT") or (key[0:2] == "tm") ):
      rlist3[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]

  set_map_res(rlist,rlist3)           # Set some map resources
  set_vector_res(rlist,rlist2)        # Set some vector resources
  set_labelbar_res(rlist,rlist2)      # Set some labelbar resources
    
#
#  Call the wrapped function and return.
#
  ivct = vector_map_wrap(wks,uarray,varray,"double","double",         \
                     uarray.shape[0],uarray.shape[1],0,               \
                     pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(), \
                     rlist1,rlist2,rlist3,pvoid())

  del rlist
  del rlist1
  del rlist2
  del rlist3
  return lst2pobj(ivct)

def vector_scalar(wks,uarray,varray,tarray,rlistc=None):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to VectorField, ScalarField, and VectorPlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif(key[0:2] == "sf"):
      rlist2[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist3[key] = rlist[key]
    
  set_vector_res(rlist,rlist3)        # Set some vector resources
  set_labelbar_res(rlist,rlist3)      # Set some labelbar resources

#
#  Call the wrapped function and return.
#
  ivct = vector_scalar_wrap(wks,uarray,varray,tarray,  \
                     "double","double","double",         \
                     uarray.shape[0],uarray.shape[1],0,               \
                     pvoid(),"",0,pvoid(),"", 0, 0, 0, pvoid(), pvoid(), \
                     pvoid(),rlist1,rlist2,rlist3,pvoid())

  del rlist
  del rlist1
  del rlist2
  del rlist3
  return lst2pobj(ivct)

def vector_scalar_map(wks,uarray,varray,tarray,rlistc=None):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to VectorField, ScalarField, MapPlot, and 
#  VectorPlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  rlist4 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif(key[0:2] == "sf"):
      rlist2[key] = rlist[key]
    elif( (key[0:2] == "mp") or (key[0:2] == "vp") or (key[0:3] == "pmA") or \
          (key[0:3] == "pmO") or (key[0:3] == "pmT") or (key[0:2] == "tm") ):
      rlist4[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist3[key] = rlist[key]
    
  set_map_res(rlist,rlist4)           # Set some map resources
  set_vector_res(rlist,rlist3)        # Set some vector resources
  set_labelbar_res(rlist,rlist3)      # Set some labelbar resources

#
#  Call the wrapped function and return.
#
  ivct = vector_scalar_map_wrap(wks,uarray,varray,tarray,  \
                     "double","double","double",         \
                     uarray.shape[0],uarray.shape[1],0,               \
                     pvoid(),"",0,pvoid(),"", 0, 0, 0, pvoid(), pvoid(), \
                     pvoid(),rlist1,rlist2,rlist3,rlist4,pvoid())

  del rlist
  del rlist1
  del rlist2
  del rlist3
  del rlist4
  return lst2pobj(ivct)

def text_ndc(wks, text, x, y, rlistc=None):
  set_spc_defaults(0)
  rlist = crt_dict(rlistc)
  rlist1 = {}
  for key in rlist.keys():
    if(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]

  itxt = text_ndc_wrap(wks,text,x,y,"double","double",rlist1,pvoid())
  del rlist
  del rlist1
  return (lst2pobj(itxt))

def add_text(wks,plot,text,x,y,rlistc=None):
  rlist = crt_dict(rlistc)  
 
#
#  Separate the resource dictionary into those resources
#  that apply to am or tx.
#
  am_rlist  = {}
  tx_rlist  = {}
  for key in rlist.keys():
    if (key[0:2] == "tx"):
      tx_rlist[key] = rlist[key]
    if (key[0:2] == "am"):
      am_rlist[key] = rlist[key]

#
#  Call the wrapped function and return.
#
  atx = add_text_wrap(wks,pobj2lst(plot),text,x,y,"double","double",  \
                          tx_rlist,am_rlist, pvoid())
  del rlist
  del tx_rlist
  del am_rlist
  return(lst2pobj(atx))

def text(wks, plot, text, x, y, rlistc=None):
  set_spc_defaults(0)
  rlist = crt_dict(rlistc)
  rlist1 = {}
  for key in rlist.keys():
    if(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]
  itxt = text_wrap(wks,pobj2lst(plot),text,x,y,"double","double",rlist1,pvoid())
  del rlist
  del rlist1
  return(lst2pobj(itxt))

def set_values(obj,rlistc=None):
  rlist = crt_dict(rlistc)
  values = NhlSetValues(int_id(obj),rlist)
  del rlist
  return values

def retrieve_colormap(wks):
  return get_MDfloat_array(wks,"wkColorMap")

def get_values(obj,rlistc):
  rlist = crt_dict(rlistc)
  values = NhlGetValues(int_id(obj),rlist)
  del rlist
  return (values)

def destroy(obj):
  return(NhlDestroy(int_id(obj)))

def clear_workstation(obj):
  NhlClearWorkstation(int_id(obj))

def update_workstation(obj):
  NhlUpdateWorkstation(int_id(obj))

def get_float(obj,name):
  return(NhlGetFloat(int_id(obj),name))

def get_integer(obj,name):
  return(NhlGetInteger(int_id(obj),name))

def get_string(obj,name):
  return(NhlGetString(int_id(obj),name))

def get_double(obj,name):
  return(NhlGetDouble(int_id(obj),name))

def get_integer_array(obj,name):
  return(NhlGetIntegerArray(int_id(obj),name))

def get_float_array(obj,name):
  return(NhlGetFloatArray(int_id(obj),name))

def get_double_array(obj,name):
  return(NhlGetDoubleArray(int_id(obj),name))

def get_string_array(obj,name):
  return(NhlGetStringArray(int_id(obj),name))

def get_MDfloat_array(obj,name):
  rval = NhlGetMDFloatArray(int_id(obj),name)
  if (rval[0] != -1):
    print "get_MDfloat_array: error number %d" % (rval[0])
    return None
  return(rval[1])

def get_MDdouble_array(obj,name):
  return(NhlGetMDDoubleArray(int_id(obj),name))

def get_MDinteger_array(obj,name):
  return(NhlGetMDIntegerArray(int_id(obj),name))

def frame(wks):
  return(NhlFrame(wks))

def draw(obj):
  return(NhlDraw(int_id(obj)))

def get_tokens(line):
  tstart = []
  tend   = []
  tokens = []
  llen = len(line)
  if (line[0] != " "):
    tstart.append(0)
  for chr_num in xrange(0,llen-1):
    if (line[chr_num] == " " and line[chr_num+1] != " "): 
      tstart.append(chr_num+1)
    if (line[chr_num] != " " and line[chr_num+1] == " "): 
      tend.append(chr_num+1)
  if (line[llen-1] != " "):
    tend.append(llen)

  for i in xrange(len(tstart)):
    tokens.append(line[tstart[i]:tend[i]])
    
  return tokens


def asciiread(filename,dims,type):
  file = open(filename)
  nnum = 1
  for m in xrange(len(dims)):
    nnum = nnum*dims[m]
 
  if (type == "integer"):
    ar = Numeric.zeros(nnum,Numeric.int) 
  elif (type == "float"):
    ar = Numeric.zeros(nnum,Numeric.Float0) 
  elif (type == "double"):
    ar = Numeric.zeros(nnum,Numeric.Float) 
  else:
    print 'asciiread: type must be one of: "integer", "float", or "double".'
    sys.exit()

  count = 0
  while (1):
    line = file.readline()[0:-1]
    if len(line) == 0:
      break
    toks = get_tokens(line)
    for str in toks:
      if (type == "integer"):
        try:
          ar[count] = string.atoi(str)
        except:
          print "asciiread: data must be integer"
          return None
      elif ((type == "float") or (type == "double")):
        try:
          ar[count] = string.atof(str)
        except:
          print "asciiread: data must be floating point"
          return None
      count = count+1
      if (count >= nnum):
        file.close()
        return Numeric.reshape(ar,dims)

  if (count < nnum):
    print "asciiread: Warning, fewer data items than specified array size."

  file.close()
  return Numeric.reshape(ar,dims)

def get_workspace_id():
  return NhlGetWorkspaceObjectId()

def rgbhls(r,g,b):
  return(c_rgbhls(r,g,b))
def hlsrgb(r,g,b):
  return(c_hlsrgb(r,g,b))
def rgbhsv(r,g,b):
  return(c_rgbhsv(r,g,b))
def hsvrgb(r,g,b):
  return(c_hsvrgb(r,g,b))
def rgbyiq(r,g,b):
  return(c_rgbyiq(r,g,b))
def yiqrgb(r,g,b):
  return(c_yiqrgb(r,g,b))
