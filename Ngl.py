
from hlu import *
import hlu
import sys
import types
import string
import Numeric
from Scientific.IO.NetCDF import *

first_call_to_open_wks = 0

class Resources:
  pass

class PlotIds:
  pass

def ngl_int_id(plot_id):
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

def ngl_overlay(plot_id1,plot_id2):
  NhlAddOverlay(ngl_int_id(plot_id1),ngl_int_id(plot_id2),-1)

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

def set_ngl_spc_res(resource_name,value):
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
  if (resource_name[0:8]   == "Maximize"):
    set_nglRes_i(0, lval) 
  elif (resource_name[0:4] == "Draw"):
    set_nglRes_i(1, lval) 
  elif (resource_name[0:5] == "Frame"):
    set_nglRes_i(2, lval) 
  elif (resource_name[0:5] == "Scale"):
    set_nglRes_i(3, lval) 
  elif (resource_name[0:5] == "Debug"):
    set_nglRes_i(4, lval) 
  elif (resource_name[0:16] == "PaperOrientation"):
    set_nglRes_i(5, lval) 
  elif (resource_name[0:10] == "PaperWidth"):
    set_nglRes_f(6, lval) 
  elif (resource_name[0:11] == "PaperHeight"):
    set_nglRes_f(7, lval) 
  elif (resource_name[0:11] == "PaperMargin"):
    set_nglRes_f(8, lval) 
  elif (resource_name[0:11] == "PanelCenter"):
    set_nglRes_i(9, lval) 
  elif (resource_name[0:12] == "PanelRowSpec"):
    set_nglRes_i(10, lval) 
  elif (resource_name[0:23] == "PanelXWhiteSpacePercent"):
    set_nglRes_f(11, lval) 
  elif (resource_name[0:23] == "PanelYWhiteSpacePercent"):
    set_nglRes_f(12, lval) 
  elif (resource_name[0:10] == "PanelBoxes"):
    set_nglRes_i(13, lval) 
  elif (resource_name[0:9] == "PanelLeft"):
    set_nglRes_f(14, lval) 
  elif (resource_name[0:10] == "PanelRight"):
    set_nglRes_f(15, lval) 
  elif (resource_name[0:11] == "PanelBottom"):
    set_nglRes_f(16, lval) 
  elif (resource_name[0:8] == "PanelTop"):
    set_nglRes_f(17, lval) 
  elif (resource_name[0:14] == "PanelInvsblTop"):
    set_nglRes_f(18, lval) 
  elif (resource_name[0:15] == "PanelInvsblLeft"):
    set_nglRes_f(19, lval) 
  elif (resource_name[0:16] == "PanelInvsblRight"):
    set_nglRes_f(20, lval) 
  elif (resource_name[0:17] == "PanelInvsblBottom"):
    set_nglRes_f(21, lval) 
  elif (resource_name[0:9] == "PanelSave"):
    set_nglRes_i(22, lval) 
  elif (resource_name[0:12] == "SpreadColors"):
    set_nglRes_i(23, lval) 
  elif (resource_name[0:16] == "SpreadColorStart"):
    set_nglRes_i(24, lval) 
  elif (resource_name[0:14] == "SpreadColorEnd"):
    set_nglRes_i(25, lval) 
  elif (resource_name[0:24] == "PanelLabelBarOrientation"):
    set_nglRes_i(26, lval) 
  elif (resource_name[0:13] == "PanelLabelBar" and len(resource_name) == 13):
    set_nglRes_i(27, lval) 
  elif (resource_name[0:15] == "PanelLabelBarXF"):
    set_nglRes_f(28, lval) 
  elif (resource_name[0:15] == "PanelLabelBarYF"):
    set_nglRes_f(29, lval) 
  elif (resource_name[0:29] == "PanelLabelBarLabelFontHeightF"):
    set_nglRes_f(30, lval) 
  elif (resource_name[0:19] == "PanelLabelBarWidthF"):
    set_nglRes_f(31, lval) 
  elif (resource_name[0:20] == "PanelLabelBarHeightF"):
    set_nglRes_f(32, lval) 
  elif (resource_name[0:27] == "PanelLabelBarOrthogonalPosF"):
    set_nglRes_f(33, lval) 
  elif (resource_name[0:25] == "PanelLabelBarParallelPosF"):
    set_nglRes_f(34, lval) 
  elif (resource_name[0:20] == "PanelLabelBarPerimOn"):
    set_nglRes_i(35, lval) 
  elif (resource_name[0:22] == "PanelLabelBarAlignment"):
    set_nglRes_i(36, lval) 
  elif (resource_name[0:28] == "PanelLabelBarLabelAutoStride"):
    set_nglRes_i(37, lval) 
  elif (resource_name[0:18] == "PanelFigureStrings" and len(resource_name) == 18):
    set_nglRes_c(38, lval) 
  elif (resource_name[0:23] == "PanelFigureStringsCount"):
    set_nglRes_i(39, lval) 
  elif (resource_name[0:22] == "PanelFigureStringsJust"):
    set_nglRes_i(40, lval) 
  elif (resource_name[0:32] == "PanelFigureStringsOrthogonalPosF"):
    set_nglRes_f(41, lval) 
  elif (resource_name[0:30] == "PanelFigureStringsParallelPosF"):
    set_nglRes_f(42, lval) 
  elif (resource_name[0:25] == "PanelFigureStringsPerimOn"):
    set_nglRes_i(43, lval) 
  elif (resource_name[0:37] == "PanelFigureStringsBackgroundFillColor"):
    set_nglRes_i(44, lval) 
  elif (resource_name[0:29] == "PanelFigureStringsFontHeightF"):
    set_nglRes_f(45, lval) 
  else:
    print "set_ngl_spc_res: Unknown special resource " + resource_name

def ngl_change_workstation(obj,wks):
  return NhlChangeWorkstation(ngl_int_id(obj),wks)

def ngl_end():
  NhlClose()

#
#  Get indices of a list where the list values are true.
#
def ngl_ind(seq):
  inds = []
  for i in xrange(len(seq)):
    if (seq[i] != 0):
      inds.append(i)
  return(inds)

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

def coord_variables(nfile):
  if (type(nfile).__name__ != "NetCDFFile"):
    print "coord_variables: input file must be a NetCDF file"
    return None

  coord_vars = []
  for var in nfile.variables.keys():
    dim_names = nfile.variables[var].dimensions
    if (var == dim_names[0]) and (len(dim_names) == 1):
      coord_vars.append(var)
  return coord_vars

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
  set_nglRes_i( 3, 0)   # nglScale
  set_nglRes_i( 4, 0)   # nglDebug
  set_nglRes_i( 5, -1)  # nglPaperOrientation
  set_nglRes_f( 6, 8.5) # nglPaperWidth
  set_nglRes_f( 7, 11.) # nglPaperHeight
  set_nglRes_f( 8, 0.5) # nglPaperMargin
  set_nglRes_i( 9, 0)   # nglPanelCenter
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

def ngl_open_wks(wk_type,wk_name,wk_rlist):
  global first_call_to_open_wks
  rlist = crt_dict(wk_rlist)

# 
#  Initialize the special resource values.
#
  if (first_call_to_open_wks == 0):
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
  iopn = ngl_open_wks_wrap(wk_type,wk_name,rlist)
  del rlist
  return(iopn)

def ngl_draw_colormap(wks):
  ngl_draw_colormap_wrap(wks)

def ngl_panel(wks,plots,dims,rlistc):
  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
  rlist1 = {}
  rlist2 = {}
  for key in rlist.keys():
    if (key[0:3] == "ngl"):
      if (key[0:21] == "nglPanelFigureStrings" and len(key) == 21):
        set_ngl_spc_res(key[3:],rlist[key])
        set_ngl_spc_res("PanelFigureStringsCount",len(rlist[key]))
      elif (key[0:25] == "nglPanelFigureStringsJust"):
        if(rlist[key] == "TopLeft"):
          set_ngl_spc_res(key[3:],0)
        elif(rlist[key] == "CenterLeft"): 
          set_ngl_spc_res(key[3:],1)
        elif(rlist[key] == "BottomLeft"): 
          set_ngl_spc_res(key[3:],2)
        elif(rlist[key] == "TopCenter"): 
          set_ngl_spc_res(key[3:],3)
        elif(rlist[key] == "CenterCenter"): 
          set_ngl_spc_res(key[3:],4)
        elif(rlist[key] == "BottomCenter"): 
          set_ngl_spc_res(key[3:],5)
        elif(rlist[key] == "TopRight"):
          set_ngl_spc_res(key[3:],6)
        elif(rlist[key] == "CenterRight"): 
          set_ngl_spc_res(key[3:],7)
        elif(rlist[key] == "BottomRight"): 
          set_ngl_spc_res(key[3:],8)
      else:
        set_ngl_spc_res(key[3:],rlist[key])
    elif (key[0:2] == "lb"):
      if (key == "lbLabelAlignment"):
        if (rlist[key] == "BoxCenters"):
          set_ngl_spc_res("PanelLabelBarAlignment",0)
        elif (rlist[key] == "InteriorEdges"):
          set_ngl_spc_res("PanelLabelBarAlignment",1)
        elif (rlist[key] == "ExternalEdges"):
          set_ngl_spc_res("PanelLabelBarAlignment",2)
        else:
          set_ngl_spc_res("PanelLabelBarAlignment",rlist[key])
      elif (key == "lbPerimOn"):
        if (rlist[key] == 1):
          set_ngl_spc_res("PanelLabelBarPerimOn",1)
        elif (rlist[key] == 0):
          set_ngl_spc_res("PanelLabelBarPerimOn",0)
        else:
          set_ngl_spc_res("PanelLabelBarPerimOn",rlist[key])
      elif (key == "lbLabelAutoStride"):
        if (rlist[key] == 1):
          set_ngl_spc_res("PanelLabelBarAutoStride",1)
        elif (rlist[key] == 0):
          set_ngl_spc_res("PanelLabelBarAutoStride",0)
        else:
          set_ngl_spc_res("PanelLabelBarAutoStride",rlist[key])
      elif (key == "lbLabelFontHeightF"):
        set_ngl_spc_res("PanelLabelBarFontHeightF",rlist[key])
      elif (key == "lbOrientation"):
        if (rlist[key] == "Vertical"):
          set_ngl_spc_res("PanelLabelBarOrientation",1)
        elif (rlist[key] == "Horizontal"):
          set_ngl_spc_res("PanelLabelBarOrientation",0)
        else:
          set_ngl_spc_res("PanelLabelBarOrientation",rlist[key])
        

      rlist1[key] = rlist[key]
  ngl_panel_wrap(wks,pseq2lst(plots),len(plots),dims,len(dims),rlist1,rlist2,pvoid())
  del rlist
  del rlist1

def ngl_map(wks,rlistc):
  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
  rlist1 = {}
  for key in rlist.keys():
    if (key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]
  imp = ngl_map_wrap(wks,rlist1,pvoid())
  del rlist
  del rlist1
  return(lst2pobj(imp))

def ngl_poly(wks,plot,x,y,ptype,is_ndc,rlistc):
  set_spc_defaults(0)
  rlist = crt_dict(rlistc)  
  rlist1 = {}
  for key in rlist.keys():
    if (key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]
  ply = ngl_poly_wrap(wks,pobj2lst(plot),x,y,"double","double",len(x),0,0,pvoid(), \
                      pvoid(),ptype,rlist1,pvoid())
  del rlist
  del rlist1
  return None

def ngl_polymarker_ndc(wks,x,y,rlistc):
  return(ngl_poly(wks,0,x,y,NhlPOLYMARKER,1,rlistc))

def ngl_polygon_ndc(wks,x,y,rlistc):
  return(ngl_poly(wks,0,x,y,NhlPOLYGON,1,rlistc))

def ngl_polyline_ndc(wks,x,y,rlistc):
  return(ngl_poly(wks,0,x,y,NhlPOLYLINE,1,rlistc))

def ngl_polymarker(wks,plot,x,y,rlistc):  # plot converted in ngl_poly
  return(ngl_poly(wks,plot,x,y,NhlPOLYMARKER,0,rlistc))

def ngl_polygon(wks,plot,x,y,rlistc):
  return(ngl_poly(wks,plot,x,y,NhlPOLYGON,0,rlistc))

def ngl_polyline(wks,plot,x,y,rlistc):
  return(ngl_poly(wks,plot,x,y,NhlPOLYLINE,0,rlistc))

def ngl_add_poly(wks,plot,x,y,ptype,rlistc):
  rlist = crt_dict(rlistc)  
  rlist1 = {}
  for key in rlist.keys():
    if (key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]
  ply = ngl_add_poly_wrap(wks,pobj2lst(plot), x,y, "double","double",
            len(x),0,0,pvoid(), pvoid(),ptype,rlist1,pvoid())
  del rlist
  del rlist1
  return(lst2pobj(ply))

def ngl_add_polymarker(wks,plot,x,y,rlistc):
  return(ngl_add_poly(wks,plot,x,y,NhlPOLYMARKER,rlistc))

def ngl_add_polygon(wks,plot,x,y,rlistc):
  return(ngl_add_poly(wks,plot,x,y,NhlPOLYGON,rlistc))

def ngl_add_polyline(wks,plot,x,y,rlistc):
  return(ngl_add_poly(wks,plot,x,y,NhlPOLYLINE,rlistc))

def ngl_contour_map(wks,array,rlistc):
#
#  Make sure the array is 2D.
#
  if (len(array.shape) != 2):
    print "ngl_contour - array must be 2D."
    return NULL

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField and those that apply to
#  ContourPlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "sf"):
      rlist1[key] = rlist[key]
    elif( (key[0:2] == "mp") or (key[0:2] == "vp") or (key[0:3] == "pmA") or \
          (key[0:3] == "pmO") or (key[0:3] == "pmT") or (key[0:3] == "tm") ):
      rlist2[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist3[key] = rlist[key]
    
#
#  Call the wrapped function and return.
#
  icm = ngl_contour_map_wrap(wks,array,"double", \
                          array.shape[0],array.shape[1],0, \
                          pvoid(),"",0,pvoid(),"", 0, pvoid(), \
                          rlist1,rlist3,rlist2,pvoid())
  del rlist
  del rlist1
  del rlist2
  del rlist3
  return(lst2pobj(icm))

def ngl_contour(wks,array,rlistc):

#
#  Make sure the array is 2D.
#
  if (len(array.shape) != 2):
    print "ngl_contour - array must be 2D."
    return NULL

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField and those that apply to
#  ContourPlot.
#
  rlist1 = {}
  rlist2 = {}
  for key in rlist.keys():
#
#  Turn label bars on if "cnFillOn" is set and pmLabelBarDisplayMode
#  is not in the resource list.
#
    if(key[0:8] == "cnFillOn" and rlist[key] > 0):
      if ( not (rlist.has_key("pmLabelBarDisplayMode"))):
        rlist2["pmLabelBarDisplayMode"] = "Always"
        rlist2["lbPerimOn"] = 0
    if (key[0:2] == "sf"):
      rlist1[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
    
#
#  Call the wrapped function and return.
#
  icn = ngl_contour_wrap(wks,array,"double",array.shape[0],array.shape[1],0, \
                          pvoid(),"",0,pvoid(),"", 0, pvoid(), rlist1,rlist2,
                          pvoid())
  del rlist
  del rlist1
  del rlist2
  return(lst2pobj(icn))

def ngl_xy(wks,xar,yar,rlistc):
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
      "ngl_xy: type of argument 2 must be one of: list, tuple, or Numeric array"
    return None

  if ( ((type(yar) == types.ListType) or (type(yar) == types.TupleType)) ):
    ndims_y = 1
    dsizes_y = (len(yar),)
  elif (type(yar) == type(Numeric.array([0],Numeric.Int0))):
    ndims_y = (len(yar.shape))
    dsizes_y = yar.shape
  else:
    print \
      "ngl_xy: type of argument 3 must be one of: list, tuple, or Numeric array"
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
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      xy_rlist[key] = rlist[key]

#
#  Call the wrapped function and return.
#
  ixy = ngl_xy_wrap(wks,xar,yar,"double","double",ndims_x,dsizes_x,ndims_y, \
                    dsizes_y,0,0,pvoid(),pvoid(),ca_rlist,xy_rlist,xyd_rlist,
                    pvoid())

  del rlist
  del ca_rlist
  del xy_rlist
  del xyd_rlist
  return(lst2pobj(ixy))

def ngl_y(wks,yar,rlistc):
  
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
      "ngl_xy: type of argument 3 must be one of: list, tuple, or Numeric array"
    return None

  if (len(dsizes_y) == 1):
    npts = dsizes_y
  elif (len(dsizes_y) == 2):
    npts = dsizes_y[1]
  else:
    print \
      "ngl_y: array can have at most two dimensions"
    return None
    
  return ngl_xy(wks,range(0,npts),yar,rlistc)

def ngl_streamline(wks,uarray,varray,rlistc):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField and those that apply to
#  ContourPlot.
#
  rlist1 = {}
  rlist2 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
    
#
#  Call the wrapped function and return.
#
  strm = ngl_streamline_wrap(wks,uarray,varray,"double","double",         \
                         uarray.shape[0],uarray.shape[1],0,               \
                         pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(), \
                         rlist1,rlist2,pvoid())
  del rlist
  del rlist1
  del rlist2
  return(lst2pobj(strm))

def ngl_streamline_map(wks,uarray,varray,rlistc):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField and those that apply to
#  ContourPlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif( (key[0:2] == "mp") or (key[0:2] == "vp") or (key[0:3] == "pmA") or \
          (key[0:3] == "pmO") or (key[0:3] == "pmT") or (key[0:3] == "tm")):
      rlist3[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
    
#
#  Call the wrapped function and return.
#
  strm = ngl_streamline_map_wrap(wks,uarray,varray,"double","double",         \
                         uarray.shape[0],uarray.shape[1],0,               \
                         pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(), \
                         rlist1,rlist2,rlist3,pvoid())
  del rlist
  del rlist1
  del rlist2
  del rlist3
  return(lst2pobj(strm))

def ngl_vector(wks,uarray,varray,rlistc):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField and those that apply to
#  ContourPlot.
#
  rlist1 = {}
  rlist2 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
    
#
#  Call the wrapped function and return.
#
  ivct = ngl_vector_wrap(wks,uarray,varray,"double","double",         \
                     uarray.shape[0],uarray.shape[1],0,               \
                     pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(), \
                     rlist1,rlist2,pvoid())
  del rlist
  del rlist1
  del rlist2
  return lst2pobj(ivct)

def ngl_vector_map(wks,uarray,varray,rlistc):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField and those that apply to
#  ContourPlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif( (key[0:2] == "mp") or (key[0:2] == "vp") or (key[0:3] == "pmA") or \
          (key[0:3] == "pmO") or (key[0:3] == "pmT") or (key[0:3] == "tm") ):
      rlist3[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
    
#
#  Call the wrapped function and return.
#
  ivct = ngl_vector_map_wrap(wks,uarray,varray,"double","double",         \
                     uarray.shape[0],uarray.shape[1],0,               \
                     pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(), \
                     rlist1,rlist2,rlist3,pvoid())

  del rlist
  del rlist1
  del rlist2
  del rlist3
  return lst2pobj(ivct)

def ngl_vector_scalar(wks,uarray,varray,tarray,rlistc):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField and those that apply to
#  ContourPlot.
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
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist3[key] = rlist[key]
    
#
#  Call the wrapped function and return.
#
  ivct = ngl_vector_scalar_wrap(wks,uarray,varray,tarray,  \
                     "double","double","double",         \
                     uarray.shape[0],uarray.shape[1],0,               \
                     pvoid(),"",0,pvoid(),"", 0, 0, 0, pvoid(), pvoid(), \
                     pvoid(),rlist1,rlist2,rlist3,pvoid())

  del rlist
  del rlist1
  del rlist2
  del rlist3
  return lst2pobj(ivct)

def ngl_vector_scalar_map(wks,uarray,varray,tarray,rlistc):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to ScalarField and those that apply to
#  ContourPlot.
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
          (key[0:3] == "pmO") or (key[0:3] == "pmT") or (key[0:3] == "tm") ):
      rlist4[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist3[key] = rlist[key]
    
#
#  Call the wrapped function and return.
#
  ivct = ngl_vector_scalar_map_wrap(wks,uarray,varray,tarray,  \
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

def ngl_text_ndc(wks, text, x, y, rlistc):
  set_spc_defaults(0)
  rlist = crt_dict(rlistc)
  rlist1 = {}
  for key in rlist.keys():
    if(key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]

  itxt = ngl_text_ndc_wrap(wks,text,x,y,"double","double",rlist1,pvoid())
  del rlist
  del rlist1
  return (lst2pobj(itxt))

def ngl_add_text(wks,plot,text,x,y,rlistc):
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
  atx = ngl_add_text_wrap(wks,pobj2lst(plot),text,x,y,"double","double",  \
                          tx_rlist,am_rlist, pvoid())
  del rlist
  del tx_rlist
  del am_rlist
  return(lst2pobj(atx))

def ngl_text(wks, plot, text, x, y, rlistc):
  set_spc_defaults(0)
  rlist = crt_dict(rlistc)
  rlist1 = {}
  for key in rlist.keys():
    if(key[0:3] == "ngl"):
      set_ngl_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]
  itxt = ngl_text_wrap(wks,pobj2lst(plot),text,x,y,"double","double",rlist1,pvoid())
  del rlist
  del rlist1
  return(lst2pobj(itxt))

def ngl_set_values(obj,rlistc):
  rlist = crt_dict(rlistc)
  values = NhlSetValues(ngl_int_id(obj),rlist)
  del rlist
  return values

def ngl_retrieve_colormap(wks):
  return ngl_get_MDfloat_array(wks,"wkColorMap")

def ngl_get_values(obj,rlistc):
  rlist = crt_dict(rlistc)
  values = NhlGetValues(ngl_int_id(obj),rlist)
  del rlist
  return (values)

def ngl_destroy(obj):
  NhlDestroy(ngl_int_id(obj))

def ngl_get_float(obj,name):
  return(NhlGetFloat(ngl_int_id(obj),name))

def ngl_get_integer(obj,name):
  return(NhlGetInteger(ngl_int_id(obj),name))

def ngl_get_string(obj,name):
  return(NhlGetString(ngl_int_id(obj),name))

def ngl_get_double(obj,name):
  return(NhlGetDouble(ngl_int_id(obj),name))

def ngl_get_integer_array(obj,name):
  return(NhlGetIntegerArray(ngl_int_id(obj),name))

def ngl_get_float_array(obj,name):
  return(NhlGetFloatArray(ngl_int_id(obj),name))

def ngl_get_double_array(obj,name):
  return(NhlGetDoubleArray(ngl_int_id(obj),name))

def ngl_get_string_array(obj,name):
  return(NhlGetStringArray(ngl_int_id(obj),name))

def ngl_get_MDfloat_array(obj,name):
  rval = NhlGetMDFloatArray(ngl_int_id(obj),name)
  if (rval[0] != -1):
    print "ngl_get_MDfloat_array: error number %d" % (rval[0])
    return None
  return(rval[1])

def ngl_get_MDdouble_array(obj,name):
  return(NhlGetMDDoubleArray(ngl_int_id(obj),name))

def ngl_get_MDinteger_array(obj,name):
  return(NhlGetMDIntegerArray(ngl_int_id(obj),name))

def ngl_frame(wks):
  return(NhlFrame(wks))

def ngl_draw(obj):
  return(NhlDraw(ngl_int_id(obj)))

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


def ngl_asciiread(filename,dims,type):
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

def ngl_get_workspace_id():
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
