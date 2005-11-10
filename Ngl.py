from hlu import *
import hlu
import sys, os
import site
import types
import string
import Numeric
import commands

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

def ck_for_rangs(dir):
#
#  This function checks that the appropriate data files for
#  the high-res database exist.
#
  file_names = (                                    \
                "gshhs(0).rim", "gshhs(1).rim",     \
                "gshhs(2).rim", "gshhs(3).rim",     \
                "gshhs(4).rim",                     \
                "rangs(0).cat", "rangs(0).cel",     \
                "rangs(1).cat", "rangs(1).cel",     \
                "rangs(2).cat", "rangs(2).cel",     \
                "rangs(3).cat", "rangs(3).cel",     \
                "rangs(4).cat", "rangs(4).cel"      \
               )
  for file in file_names:
    fp_file = dir + "/" + file
    if (not (os.path.exists(fp_file))):
      print '\nInfo message: The environment variable PYNGL_RANGS has '
      print '   been set, but the required high-res database file'
      print '   "' + fp_file + '"'
      print '   is not in that directory.\n'
      return None

def ismissing(arg,mval):
#
#  Returns an array of the same shape as "arg" that
#  has True values in all places where "arg" has 
#  missing values.
#
    if (type(arg) == type(Numeric.array([0],Numeric.Int))):
      pass
    elif (type(arg)==types.IntType or type(arg)==types.LongType or \
          type(arg)==types.FloatType):
      pass
    else:
      print "ismissing: first argument must be a Numeric array."
      return None
    return(Numeric.equal(arg,mval))

def get_values(obj,rlistc):
  rlist = crt_dict(rlistc)
  values = NhlGetValues(int_id(obj),rlist)
  del rlist
  return (values)

def pynglpath_ncarg():
#
#  Find the root directory that contains the supplemental PyNGL files,
#  like fontcaps, colormaps, and map databases. The default is to look
#  in site-packages/PyNGL/ncarg. Otherwise, check the PYNGL_NCARG
#  environment variable.
#
  pkgs_pth    = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
                               'site-packages')
  pyngl1_dir  = pkgs_pth + "/PyNGL/ncarg"
  pyngl2_dir  = os.environ.get("PYNGL_NCARG")
  ncarg_ncarg = None

  if (pyngl2_dir != None and os.path.exists(pyngl2_dir)):
    pyngl_ncarg = pyngl2_dir
  elif (os.path.exists(pyngl1_dir)):
    pyngl_ncarg = pyngl1_dir
  else:
    print "pynglpath: directory" + pyngl1_dir + \
          "\n           does not exist and " + \
          "environment variable PYNGL_NCARG is not set." 
    sys.exit()

  return pyngl_ncarg

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
  elif (resource_name == "AppResFileName"):
    set_nglRes_s(46, lval) 
  elif (resource_name == "XAxisType"):
    if(type(lval) == types.StringType):
      if(string.lower(lval) == "irregularaxis"):
        set_nglRes_i(47, 0) 
      elif(string.lower(lval) == "linearaxis"):
        set_nglRes_i(47, 1) 
      elif(string.lower(lval) == "logaxis"):
        set_nglRes_i(47, 2) 
      else:
        print "set_spc_res: Unknown value for " + resource_name
    else:
      set_nglRes_i(47, lval) 
  elif (resource_name == "YAxisType"):
    if(type(lval) == types.StringType):
      if(string.lower(lval) == "irregularaxis"):
        set_nglRes_i(48, 0) 
      elif(string.lower(lval) == "linearaxis"):
        set_nglRes_i(48, 1) 
      elif(string.lower(lval) == "logaxis"):
        set_nglRes_i(48, 2) 
      else:
        print "set_spc_res: Unknown value for " + resource_name
    else:
      set_nglRes_i(48, lval) 
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

def set_tickmark_res(reslist,reslist1):
#
# Set tmEqualizeXYSizes to True so that tickmark lengths and font
# heights are the same size on both axes.
#
  if((reslist.has_key("nglScale") and reslist["nglScale"] > 0) or
     (not (reslist.has_key("nglScale")))):
    reslist1["tmEqualizeXYSizes"] = True

def set_contour_res(reslist,reslist1):
#
#  Set some contour resources of which we either don't like the NCL
#  defaults, or we want to set something on behalf of the user.
#
  if(reslist.has_key("cnFillOn") and reslist["cnFillOn"] > 0):
    if ( not (reslist.has_key("cnInfoLabelOn"))):
      reslist1["cnInfoLabelOn"] = False
    if ( not (reslist.has_key("pmLabelBarDisplayMode")) and 
         (not (reslist.has_key("lbLabelBarOn")) or 
               reslist.has_key("lbLabelbarOn") and 
                        reslist["lbLabelBarOn"] > 0)):
      reslist1["pmLabelBarDisplayMode"] = "Always"
#
# The ContourPlot object does not recognize the lbLabelBarOn resource
# so we have to remove it after we've used it.
#
  if (reslist1.has_key("lbLabelBarOn")):
    del reslist1["lbLabelBarOn"]
#
# If cnFillColors is set, don't set nglSpreadColors to True.
#
  if(reslist.has_key("cnFillColors")):
    if ( not (reslist.has_key("nglSpreadColors"))):
      lval = 0
      set_nglRes_i(23, lval) 
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
#
# Set some tickmark resources.
#
  set_tickmark_res(reslist,reslist1)

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
    if ( not (reslist.has_key("pmLabelBarDisplayMode")) and 
         (not (reslist.has_key("lbLabelBarOn")) or 
               reslist.has_key("lbLabelbarOn") and 
                        reslist["lbLabelBarOn"] > 0)):
      reslist1["pmLabelBarDisplayMode"] = "Always"
#
# The VectorPlot object does not recognize the lbLabelBarOn resource
# so we have to remove it after we've used it.
#
  if (reslist1.has_key("lbLabelBarOn")):
    del reslist1["lbLabelBarOn"]
#
# If vcLevelColors is set, don't set nglSpreadColors to True.
#
  if(reslist.has_key("vcLevelColors")):
    if ( not (reslist.has_key("nglSpreadColors"))):
      lval = 0
      set_nglRes_i(23, lval) 
#
# Set some tickmark resources.
#
  set_tickmark_res(reslist,reslist1)

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
#
# Set some tickmark resources.
#
  set_tickmark_res(reslist,reslist1)

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
  

def set_labelbar_res(reslist,reslist1,part_of_plot):
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
  if(part_of_plot):
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
  if ( not (reslist.has_key("lgPerimOn"))):
    reslist1["lgPerimOn"] = False
  if ( not (reslist.has_key("lgLabelAutoStride"))):
    reslist1["lgLabelAutoStride"] = True
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
    set_nglRes_i(0, 1)      # nglMaximize
    set_nglRes_i(1, 1)      # nglDraw
    set_nglRes_i(2, 1)      # nglFrame
  elif (type == 0):
    set_nglRes_i(0, 0)      # nglMaximize
    set_nglRes_i(1, 1)      # nglDraw
    set_nglRes_i(2, 0)      # nglFrame
#
  set_nglRes_i( 3, 1)       # nglScale
  set_nglRes_i( 4, 0)       # nglDebug
  set_nglRes_i( 5, -1)      # nglPaperOrientation
  set_nglRes_f( 6, 8.5)     # nglPaperWidth
  set_nglRes_f( 7, 11.)     # nglPaperHeight
  set_nglRes_f( 8, 0.5)     # nglPaperMargin
  set_nglRes_i( 9, 1)       # nglPanelCenter
  set_nglRes_i(10, 0)       # nglPanelRowSpec
  set_nglRes_f(11, 1.)      # nglPanelXWhiteSpacePercent
  set_nglRes_f(12, 1.)      # nglPanelYWhiteSpacePercent
  set_nglRes_i(13, 0)       # nglPanelBoxes
  set_nglRes_f(14, 0.)      # nglPanelLeft
  set_nglRes_f(15, 1.)      # nglPanelRight
  set_nglRes_f(16, 0.)      # nglPanelBottom
  set_nglRes_f(17, 1.)      # nglPanelTop
  set_nglRes_f(18, -999.)   # nglPanelInvsblTop
  set_nglRes_f(19, -999.)   # nglPanelInvsblLeft
  set_nglRes_f(20, -999.)   # nglPanelInvsblRight
  set_nglRes_f(21, -999.)   # nglPanelInvsblBottom
  set_nglRes_i(22, 0)       # nglPanelSave
  set_nglRes_i(23, 1)       # nglSpreadColors
  set_nglRes_i(24, 2)       # nglSpreadColorStart
  set_nglRes_i(25, -1)      # NglSpreadColorEnd
  set_nglRes_i(26, 0)       # nglPanelLabelBarOrientation
  set_nglRes_i(27, 0)       # nglPanelLabelBar
  set_nglRes_f(28, -999.)   # nglPanelLabelBarXF
  set_nglRes_f(29, -999.)   # nglPanelLabelBarYF
  set_nglRes_f(30, -999.)   # nglPanelLabelBarLabelFontHeightF
  set_nglRes_f(31, -999.)   # nglPanelLabelBarWidthF
  set_nglRes_f(32, -999.)   # nglPanelLabelBarHeightF
  set_nglRes_f(33, -999.)   # nglPanelLabelBarOrthogonalPosF
  set_nglRes_f(34, -999.)   # nglPanelLabelBarParallelPosF
  set_nglRes_i(35, 0)       # nglPanelLabelBarPerimOn
  set_nglRes_i(36, 1)       # nglPanelLabelBarAlignment
  set_nglRes_i(37, 1)       # nglPanelLabelBarLabelAutoStride
  set_nglRes_c(38, [])      # nglPanelFigureStrings
  set_nglRes_i(39, 0)       # nglPanelFigureStringsCount
  set_nglRes_i(40, 8)       # nglPanelFigureStringsJust
  set_nglRes_f(41, -999.)   # nglPanelFigureStringsOrthogonalPosF
  set_nglRes_f(42, -999.)   # nglPanelFigureStringsParallelPosF
  set_nglRes_i(43, 1)       # nglPanelFigureStringsPerimOn
  set_nglRes_i(44, 0)       # nglPanelFigureStringsBackgroundFillColor
  set_nglRes_f(45, -999.)   # nglPanelFigureStringsFontHeightF
  set_nglRes_s(46, "")      # nglAppResFileName
  set_nglRes_i(47, 0)       # nglXAxisType - default to irregular
  set_nglRes_i(48, 0)       # nglYAxisType - default to irregular


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

def get_workspace_id():
  return NhlGetWorkspaceObjectId()


def ck_type(fcn,arg,typ):
#
#  Check on the type of the variable "arg" that is a
#  variable in function "fcn" accroding to the flag "typ".
#  Returns 0 if OK, 1 otherwise.
#
  if (typ == 0):
#
#  arg should be a singly-dimensioned Numeric array with ints, longs,
#  or floats, or a scalar int, long, or float.
#
    if (type(arg) == type(Numeric.array([0],Numeric.Int))):
      if (len(arg) == 0):
        print "Warning: " + fcn + ": An empty array was encountered."
        return 0
      if (len(arg.shape) != 1):
        print fcn + ": Numeric array argument must be singly-dimensioned."
        return 1
      a0 = arg[0]
      if (type(a0)!=types.IntType and type(a0)!=types.FloatType and \
          type(a0)!=types.LongType):
        print fcn + \
          ": Numeric array argument must be integers, longs, or floats."
        return 1
    elif (type(arg)==types.IntType or type(arg)==types.LongType or \
          type(arg)==types.FloatType):
      return 0
    else:
      print fcn + ": argument must be a Numeric array or numeric scalar."
      return 1
  else:
    print "ck_type: invalid type flag"
    return 1
  return 0

def skewty(pres):    # y-coord given pressure (mb)
  if (ck_type("skewty",pres,0) != 0):
    return None
  try:
    return(132.182-44.061*Numeric.log10(pres))
  except:
    return None
# return (132.182-44.061*Numeric.log10(pres))
def skewtx(temp,y):  # x-coord given temperature (c)
  if (ck_type("skewtx",temp,0) != 0 or ck_type("skewtx",y,0) != 0):
    return None
  return (0.54*temp+0.90692*y)

#########################################################################
# 
#   Processing functions:
#     Ngl.add_cyclic
#     Ngl.fspan
#     Ngl.ftcurv
#     Ngl.ftcurvp
#     Ngl.ftcurvpi
#     Ngl.gaus
#     Ngl.gc_convert
#     Ngl.gc_dist
#     Ngl.gc_interp
#     Ngl.ind
#     Ngl.ismissing
#     Ngl.natgrid
#     Ngl.ncargpath
#     Ngl.normalize_angle
# 
#########################################################################


################################################################
#
#  Processing support functions.
#
################################################################

def dptlclskewt(p, tc, tdc):
  return c_dptlclskewt(p, tc, tdc)

def dtmrskewt(w, p):
  return c_dtmrskewt(w, p)

def dtdaskewt(o,p):
  return c_dtdaskewt(o, p)

def dsatlftskewt(thw,p):
  return c_dsatlftskewt(thw, p)

def dshowalskewt(p,t,td,nlvls):
  return c_dshowalskewt(p,t,td,nlvls)

def dpwskewt(td,p,n):
  return c_dpwskewt(td,p,n)

################################################################
#  
#  Public functions in alphabetical order.
#  
################################################################
def add_annotation(plot_id1,plot_id2):
  return NhlAddAnnotation(int_id(plot_id1),int_id(plot_id2))

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

def add_polygon(wks,plot,x,y,rlistc=None):
  return(add_poly(wks,plot,x,y,NhlPOLYGON,rlistc))

def add_polyline(wks,plot,x,y,rlistc=None):
  return(add_poly(wks,plot,x,y,NhlPOLYLINE,rlistc))

def add_polymarker(wks,plot,x,y,rlistc=None):
  return(add_poly(wks,plot,x,y,NhlPOLYMARKER,rlistc))

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

def asciiread(filename,dims,type="float"):
  file = open(filename)
#
#  If dims = -1, determine the number of valid tokens in
#  the input file, otherwise calculate the number from
#  the dims value.  If dims = -1 the return value will be
#  a Numeric array containing of all the legal values,
#  all other values are ignored.
#
  if (dims == -1):
    nnum = 0
    while (1):
      line = file.readline()[0:-1]
      if len(line) == 0:
        break
      toks = string.split(line)
      for str in toks:
        if (type == "integer"):
          try:
            string.atoi(str)
            nnum = nnum+1
          except:
            pass
        elif ((type == "float") or (type == "double")):
          try:
            string.atof(str)
            nnum = nnum+1
          except:
            pass
  else:
    nnum = 1
    if (not (isinstance(dims,types.ListType)) and \
        not (isinstance(dims,types.TupleType))):
      print 'asciiread: dims must be a list or a tuple'
      return None
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
  file.seek(0,0)
  while (1):
    line = file.readline()[0:-1]
    if len(line) == 0:
      break
    toks = string.split(line)
    for str in toks:
      if (type == "integer"):
        try:
          ar[count] = string.atoi(str)
          count = count+1
        except:
          pass
      elif ((type == "float") or (type == "double")):
        try:
          ar[count] = string.atof(str)
          count = count+1
        except:
          pass

  if (count < nnum and dims != -1):
    print "asciiread: Warning, fewer data items than specified array size."

  file.close()
  if (dims == -1):
    return ar
  else:
    return Numeric.reshape(ar,dims)

def change_workstation(obj,wks):
  return NhlChangeWorkstation(int_id(obj),wks)

def clear_workstation(obj):
  NhlClearWorkstation(int_id(obj))

def contour(wks,array,rlistc=None):

#
#  Make sure the array is 1D or 2D.
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
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "sf"):
      rlist1[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
#
# In addition, if this plot is potentially going to be overlaid
# on an Irregular Plot Class (in order to lineariize or logize it)
# then we need to keep track of all the tickmark resources, because
# we'll have to reapply them to the IrregularPlot class.
#
      if(key[0:2] == "vp" or key[0:2] == "tm" or key[0:6] == "pmTick"):
        rlist3[key] = rlist[key]

  set_contour_res(rlist,rlist2)       # Set some addtl contour resources
  set_labelbar_res(rlist,rlist2,True) # Set some addtl labelbar resources
  set_tickmark_res(rlist,rlist3)      # Set some addtl tickmark resources
#
#  Call the wrapped function and return.
#
  if (len(array.shape) == 2):
    icn = contour_wrap(wks,array,"double",array.shape[0],array.shape[1], \
                           0, pvoid(),"",0,pvoid(),"", 0, pvoid(), rlist1, \
                          rlist2,rlist3,pvoid())
  else:
    icn = contour_wrap(wks,array,"double",array.shape[0],-1, \
                           0, pvoid(),"",0,pvoid(),"", 0, pvoid(), rlist1, \
                          rlist2,rlist3,pvoid())

  del rlist
  del rlist1
  del rlist2
  del rlist3
  return(lst2pobj(icn))

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

  set_map_res(rlist,rlist2)           # Set some addtl map resources
  set_contour_res(rlist,rlist3)       # Set some addtl contour resources
  set_labelbar_res(rlist,rlist3,True) # Set some addtl labelbar resources

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

def destroy(obj):
  return(NhlDestroy(int_id(obj)))

def draw(obj):
  return(NhlDraw(int_id(obj)))

def draw_colormap(wks):
  draw_colormap_wrap(wks)

def end():
  NhlClose()

def frame(wks):
  return(NhlFrame(wks))

def fspan(min,max,num):
  delta = (float(max-min)/float(num-1))
  a = []
  for i in range(num-1):
    a.append(float(i)*delta)
  a.append(max)
  return Numeric.array(a,Numeric.Float0)

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

def gaus(n):
  return NglGaus_p(n,2*n,2)[1]

def gc_convert(angle,ctype):
#
#  Convert an angle in degrees along a great circle to
#  radians, meters, kilometers, or feet.
#
  d2r =  0.0174532952   # degrees to radians
  r2m = 6371220.        # radians to meters
  m2f = 3.2808          # meters to feet

  ck_type("gc_convert",angle,0)

  dtype = ctype
  if (ctype == 0):
    dtype = "ra"
  elif (ctype == 1):
    dtype = "me"
  elif (ctype == 2):
    dtype = "ki"
  elif (ctype == 3):
    dtype = "fe"
  elif (ctype == 4):
    dtype = "mi"

  if (dtype[0:2] == "ra"):
    return d2r*angle
  elif (dtype[0:2] == "me"):
    return d2r*angle*r2m
  elif (dtype[0:2] == "ki"):
    return d2r*angle*r2m/1000.
  elif (dtype[0:2] == "fe"):
    return d2r*angle*r2m*m2f
  elif (dtype[0:2] == "mi"):
    return d2r*angle*r2m*m2f/5280.
  else:
    print "gc_convert: unrecognized conversion type " + str(ctype)

def gc_dist(rlat1,rlon1,rlat2,rlon2):
  return c_dgcdist(rlat1,rlon1,rlat2,rlon2,2)

def gc_interp(rlat1,rlon1,rlat2,rlon2,numi):
  num = abs(numi)
  if (abs(num) < 2):
    print "gc_interp: the number of points must be at least two."
  elif (num == 2):
    lat = Numeric.array([rlat1,rlat2],Numeric.Float0)
    lon = Numeric.array([rlon1,rlon2],Numeric.Float0)
    return [lat,lon]
  else:
    lat_tmp = Numeric.zeros(num,Numeric.Float0) 
    lon_tmp = Numeric.zeros(num,Numeric.Float0) 
    lat,lon = mapgci(rlat1,rlon1,rlat2,rlon2,num-2)
    lon0_tmp = rlon1
    lon1_tmp = rlon2
#
#  Adjust points to be in the desired range.
#
    for i in range(0,num-2):
      if (numi > 0):
        lon[i] = normalize_angle(lon[i],0)
      else:
        lon[i] = normalize_angle(lon[i],1)
    if (numi > 0):
      lon0_tmp = normalize_angle(lon0_tmp,0) 
      lon1_tmp = normalize_angle(lon1_tmp,0)
    else:
      lon0_tmp = normalize_angle(lon0_tmp,1) 
      lon1_tmp = normalize_angle(lon1_tmp,1)

#
#  Set up return arrays.
#
    lat_tmp[1:num-1] = lat[0:num-2]
    lon_tmp[1:num-1] = lon[0:num-2]
    lat_tmp[0]     = rlat1
    lat_tmp[num-1] = rlat2
    lon_tmp[0]     = lon0_tmp
    lon_tmp[num-1] = lon1_tmp
    del lat,lon

    return lat_tmp,lon_tmp

def get_double(obj,name):
  return(NhlGetDouble(int_id(obj),name))

def get_double_array(obj,name):
  return(NhlGetDoubleArray(int_id(obj),name))

def get_float(obj,name):
  return(NhlGetFloat(int_id(obj),name))

def get_float_array(obj,name):
  return(NhlGetFloatArray(int_id(obj),name))

def get_integer(obj,name):
  return(NhlGetInteger(int_id(obj),name))

def get_integer_array(obj,name):
  return(NhlGetIntegerArray(int_id(obj),name))

def get_MDdouble_array(obj,name):
  return(NhlGetMDDoubleArray(int_id(obj),name))

def get_MDfloat_array(obj,name):
  rval = NhlGetMDFloatArray(int_id(obj),name)
  if (rval[0] != -1):
    print "get_MDfloat_array: error number %d" % (rval[0])
    return None
  return(rval[1])

def get_MDinteger_array(obj,name):
  return(NhlGetMDIntegerArray(int_id(obj),name))

#
#  Returns the color index whose associated color on the given
#  workstation is closest to the color name supplied.
#
def get_named_color_index(wkid,name):
  return(NhlGetNamedColorIndex(wkid,name))

def get_string(obj,name):
  return(NhlGetString(int_id(obj),name))

def get_string_array(obj,name):
  return(NhlGetStringArray(int_id(obj),name))

def hlsrgb(r,g,b):
  return(c_hlsrgb(r,g,b))

def hsvrgb(r,g,b):
  return(c_hsvrgb(r,g,b))

#
#  Get indices of a list where the list values are true.
#
def ind(seq):
  inds = []
  for i in xrange(len(seq)):
    if (seq[i] != 0):
      inds.append(i)
  return(inds)

def labelbar_ndc(wks,nbox,labels,x,y,rlistc=None):
  set_spc_defaults(0)
  rlist = crt_dict(rlistc)
  rlist1 = {}
  for key in rlist.keys():
    if(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]

  set_labelbar_res(rlist,rlist1,False) # Set some addtl labelbar resources

  ilb = labelbar_ndc_wrap(wks,nbox,labels,len(labels),x,y, 
                          "double","double",rlist1,pvoid())
  del rlist
  del rlist1
  return (lst2pobj(ilb))

def legend_ndc(wks,nitems,labels,x,y,rlistc=None):
  set_spc_defaults(0)
  rlist = crt_dict(rlistc)
  rlist1 = {}
  for key in rlist.keys():
    if(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]

  set_legend_res(rlist,rlist1)      # Set some addtl legend resources

  ilb = legend_ndc_wrap(wks,nitems,labels,len(labels),x,y, 
                        "double","double",rlist1,pvoid())
  del rlist
  del rlist1
  return (lst2pobj(ilb))

def map(wks,rlistc=None):
  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
  rlist1 = {}
  for key in rlist.keys():
    if (key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]

  set_map_res(rlist,rlist1)           # Set some addtl map resources

  imp = map_wrap(wks,rlist1,pvoid())

  del rlist
  del rlist1
  return(lst2pobj(imp))

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
  return pynglpath(type)

def new_color(wks_id,r,g,b):
  return NhlNewColor(int_id(wks_id),r,g,b)

def nngetp(pname):
  iparms = [                                                         \
            "adf", "asc", "dup", "ext", "igr", "non", "rad",         \
            "sdi", "upd", "mdm",                                     \
            "ADF", "ASC", "DUP", "EXT", "IGR", "NON", "RAD",         \
            "SDI", "UPD", "MDM"                                      \
           ]
  rparms = [                                                         \
            "bI", "bJ", "hor", "magx", "magy", "magz", "nul", "ver", \
            "Bi", "Bj", "HOR", "MAGX", "MAGY", "MAGZ", "NUL", "VER", \
            "bi", "bj", "BI", "BJ"                                   \
           ]
  cparms = [                                                         \
            "alg", "ALG", "erf", "ERF"                                \
           ]
  if (not isinstance(pname,types.StringType)):
    print "nngetp: Parameter '" + str(pname) + "' is not a string type." 
    return None
  if (iparms.count(pname) > 0):
    return c_nngeti(pname)
  elif (rparms.count(pname) > 0):
    return c_nngetrd(pname)
  elif (cparms.count(pname) > 0):
    return c_nngetcp(pname)
  else:
    print \
      "nngetp: specified value for " + pname + " is not of a recognized type." 
  return None

def nnsetp(pname,val):
  if (not isinstance(pname,types.StringType)):
    print "nnsetp: Parameter '" + str(pname) + "' is not a string type." 
    return None
  if (isinstance(val,types.IntType)):
    c_nnseti(pname,val)
  elif (isinstance(val,types.FloatType)): 
    c_nnsetrd(pname,val)
  elif (isinstance(val,types.StringType)):
    c_nnsetc(pname,val)
  else:
    print \
      "nnsetp: specified value for " + pname + " is not of a recognized type." 
  return None

def normalize_angle(ang,type):
#
#  This function normalizes the angle (assumed to be in degrees) to
#  an equivalent angle in the range [0.,360.) if type equals 0, or
#  to an equivalent angle in the range [-180.,180.) if type is not zero.
#
  bang = ang
  if (type == 0):
    while(bang < 0.):
      bang = bang + 360.
    while(bang >= 360.):
      bang = bang - 360.
  else:
    while(bang < -180.):
      bang = bang + 360.
    while(bang >= 180.):
      bang = bang - 360.
  return bang

def open_wks(wk_type,wk_name,wk_rlist=None):
  set_spc_defaults(1)
  global first_call_to_open_wks
  rlist = crt_dict(wk_rlist)

#
# Divide out "app" and other resources.
# 
  rlist1 = {}
  rlist2 = {}

  for key in rlist.keys():
    if (key[0:3] == "app"):
      rlist2[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist1[key] = rlist[key]

# 
#  Initialize the special resource values, and make sure 
#  NCARG_NCARG environment variable is set.
#
  if (first_call_to_open_wks == 0):

#
#  Set HLU environment variables.
#
    os.environ["NCARG_NCARG"] = pynglpath_ncarg()

    tmp_dir = os.environ.get("TMPDIR")
    if (tmp_dir != None and os.path.exists(tmp_dir)):
      os.environ["TMPDIR"] = tmp_dir
    else:
      os.environ["TMPDIR"] = "/tmp"

    color_dir_envn = os.environ.get("PYNGL_COLORMAPS")
    if (color_dir_envn != None and os.path.exists(color_dir_envn)):
      os.environ["NCARG_COLORMAPS"] = color_dir_envn

#
#  If "PYNGL_RANGS" is set, use it.  Otherwise set 
#  the default RANGS directory.
#
    rangs_dir_envn = os.environ.get("PYNGL_RANGS")
    if (rangs_dir_envn != None and os.path.exists(rangs_dir_envn)):
#
#  Check if the high-res file are there - if not issue a
#  an information message to that effect.
#
      ck_for_rangs(rangs_dir_envn)
      os.environ["NCARG_RANGS"] = rangs_dir_envn
    else:
      dflt_rangs_path = pynglpath_ncarg() + "/rangs"
      os.environ["NCARG_RANGS"] = pynglpath_ncarg() + "/rangs"

    ures_dir_envn = os.environ.get("PYNGL_USRRESFILE")
    if (ures_dir_envn != None and os.path.exists(ures_dir_envn)):
      os.environ["NCARG_USRRESFILE"] = ures_dir_envn
    else:
      os.environ["NCARG_USRRESFILE"] = "~/.hluresfile"

    sres_dir_envn = os.environ.get("PYNGL_SYSRESFILE")
    if (sres_dir_envn != None and os.path.exists(sres_dir_envn)):
      os.environ["NCARG_SYSRESFILE"] = sres_dir_envn

    ares_dir_envn = os.environ.get("PYNGL_SYSAPPRES")
    if (ares_dir_envn != None and os.path.exists(ares_dir_envn)):
      os.environ["NCARG_SYSAPPRES"] = ares_dir_envn
    else:
      os.environ["NCARG_SYSAPPRES"] = pynglpath_ncarg() + "/sysappres"

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
  iopn = open_wks_wrap(wk_type,wk_name,rlist1,rlist2,pvoid())
  del rlist
  del rlist1
  del rlist2
  return(iopn)

def overlay(plot_id1,plot_id2):
  NhlAddOverlay(int_id(plot_id1),int_id(plot_id2),-1)

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

def polygon(wks,plot,x,y,rlistc=None):
  return(poly(wks,plot,x,y,NhlPOLYGON,0,rlistc))

def polygon_ndc(wks,x,y,rlistc=None):
  return(poly(wks,0,x,y,NhlPOLYGON,1,rlistc))

def polyline(wks,plot,x,y,rlistc=None):
  return(poly(wks,plot,x,y,NhlPOLYLINE,0,rlistc))

def polyline_ndc(wks,x,y,rlistc=None):
  return(poly(wks,0,x,y,NhlPOLYLINE,1,rlistc))

def polymarker(wks,plot,x,y,rlistc=None):  # plot converted in poly
  return(poly(wks,plot,x,y,NhlPOLYMARKER,0,rlistc))

def polymarker_ndc(wks,x,y,rlistc=None):
  return(poly(wks,0,x,y,NhlPOLYMARKER,1,rlistc))

def pynglpath(name):
#
#  Return absolute pathnames for various directories.
#
  if (name == "tmp"):
    tmp_dir = os.environ.get("TMPDIR")
    if (tmp_dir != None and os.path.exists(tmp_dir)):
      return tmp_dir
    else:
      return "/tmp"
  elif (name == "examples"):
    examples_dir_envn = os.environ.get("PYNGL_EXAMPLES")
    examples_dir_dflt = pynglpath_ncarg() + "/pynglex"
    if (examples_dir_envn != None and os.path.exists(examples_dir_envn)):
      return examples_dir_envn
    elif (os.path.exists(examples_dir_dflt)):
      return examples_dir_dflt
    else:
      print "pynglpath: examples directory does not exist."
      return None
  elif (name == "data"):
    data_dir_envn = os.environ.get("PYNGL_DATA")
    data_dir_dflt = pynglpath_ncarg() + "/data"
    if (data_dir_envn != None and os.path.exists(data_dir_envn)):
      return data_dir_envn
    elif (os.path.exists(data_dir_dflt)):
      return data_dir_dflt
    else:
      print "pynglpath: data directory does not exist."
      return None
  elif (name == "colormaps"):
    color_dir_envn = os.environ.get("PYNGL_COLORMAPS")
    color_dir_dflt = pynglpath_ncarg() + "/colormaps"
    if (color_dir_envn != None and os.path.exists(color_dir_envn)):
      return color_dir_envn
    elif (os.path.exists(color_dir_dflt)):
      return color_dir_dflt
    else:
      print "pynglpath: colormaps directory does not exist."
      return None
  elif (name == "rangs"):
    rangs_dir_envn = os.environ.get("PYNGL_RANGS")
    rangs_dir_dflt = pynglpath_ncarg() + "/rangs"
    if (rangs_dir_envn != None and os.path.exists(rangs_dir_envn)):
      return rangs_dir_envn
    else: 
      return rangs_dir_dflt
  elif (name == "usrresfile"):
    ures_dir_envn = os.environ.get("PYNGL_USRRESFILE")
    ures_dir_dflt = commands.getoutput("ls ~/.hluresfile")
    if (ures_dir_envn != None and os.path.exists(ures_dir_envn)):
      return ures_dir_envn
    elif (os.path.exists(ures_dir_dflt)):
      return ures_dir_dflt
    else:
      print "pynglpath: useresfile directory does not exist."
      return None
  elif (name == "sysresfile"):
    sres_dir_envn = os.environ.get("PYNGL_SYSRESFILE")
    sres_dir_dflt = pynglpath_ncarg() + "/sysresfile"
    if (sres_dir_envn != None and os.path.exists(sres_dir_envn)):
      return sres_dir_envn
    elif (os.path.exists(sres_dir_dflt)):
      return sres_dir_dflt
    else:
      print "pynglpath: sysresfile directory does not exist."
      return None
  elif (name == "sysappres"):
    ares_dir_envn = os.environ.get("PYNGL_SYSAPPRES")
    ares_dir_dflt = pynglpath_ncarg() + "/sysappres"
    if (ares_dir_envn != None and os.path.exists(ares_dir_envn)):
      return ares_dir_envn
    elif (os.path.exists(ares_dir_dflt)):
      return ares_dir_dflt
    else:
      print "pynglpath: sysappres directory does not exist."
      return None
  else:
    print 'pynglpath: input name "%s" not recognized' % (name)

def remove_annotation(plot_id1,plot_id2):
  NhlRemoveAnnotation(int_id(plot_id1),int_id(plot_id2))

def remove_overlay(plot_id1,plot_id2,restore):
  NhlRemoveOverlay(int_id(plot_id1),int_id(plot_id2),restore)

def retrieve_colormap(wks):
  return get_MDfloat_array(wks,"wkColorMap")

def rgbhls(r,g,b):
  return(c_rgbhls(r,g,b))

def rgbhsv(r,g,b):
  return(c_rgbhsv(r,g,b))

def rgbyiq(r,g,b):
  return(c_rgbyiq(r,g,b))

def set_values(obj,rlistc=None):
  rlist = crt_dict(rlistc)
  values = NhlSetValues(int_id(obj),rlist)
  del rlist
  return values

def skewt_bkg(wks, Opts):
#
#  This program generates a skew-t, log p thermodynamic diagram.  
#  This program was derived to reproduce the USAF skew-t, log p diagram
#  (form dod-wpc 9-16-1  current as of march 1978).
#
#  wks is a workstation identifier and Opts is an NGL resource list.
#
#  Based on Fortran code supplied by Tom Schlatter and 
#  Joe Wakefield [NOAA/PROFS] supplied fortran codes.
#
#  history
#  -------
#  don baker      01 jul 85    original version.  [NOAA/PROFS]
#  don baker      01 dec 85    updated for product version.
#  dennis shea       oct 98    created the NCL version
#  dennis shea     9 feb 99    fix: MoistAdiabat labels at top of curves
#
  localOpts                       = Resources()
  localOpts.sktDrawIsotherm       = True
  localOpts.sktDrawIsobar         = True
  localOpts.sktDrawMixRatio       = True
  localOpts.sktDrawDryAdiabat     = True
  localOpts.sktDrawMoistAdiabat   = True  # aka: saturation or pseudo adibat
  localOpts.sktDrawWind           = True
  localOpts.sktDrawStandardAtm    = True
  localOpts.sktDrawColLine        = True
  localOpts.sktDrawColAreaFill    = False
  localOpts.sktDrawFahrenheit     = True  # Fahrenheit "x" axis
  localOpts.sktDrawHeightScale    = False
  localOpts.sktDrawHeightScaleFt  = True  # default is feet [otherwise km]
  localOpts.sktDrawStandardAtmThk = 2.0
  localOpts.Font                  = "helvetica"
  localOpts.tiMainString          = "   "
  localOpts.vpXF                  = 0.07
  localOpts.vpYF                  = 0.925
  localOpts.vpWidthF              = 0.85
  localOpts.vpHeightF             = 0.85


#
#  Override localOpts attributes with Opts attributes for
#  Opts attributes that overlap localOpts attributes; set
#  new localOpts attributes for Opts attributes that do not
#  overlap localOpts attributes.
#
  if (not isinstance(Opts,Resources)):
    print "skewt_bkg: argument 2 must be an Nlg Resources instance"
    return None
  OptsAtts = crt_dict(Opts)
  if (len(crt_dict(Opts)) != 0):
    for key in OptsAtts.keys():
      setattr(localOpts,key,OptsAtts[key])


#
#  Declare isotherm values (Celcius) and pressures (hPa) where 
#  isotherms intersect the edge of the skew-t diagram.
#
  temp = Numeric.array(                                         \
          [                                                     \
           -100.,-90.,-80.,-70.,-60.,-50.,-40.,-30.,            \
            -20.,-10.,  0., 10., 20., 30., 40., 50.             \
          ], Numeric.Float0)
  lendt = Numeric.array(                                        \
          [                                                     \
            132., 181., 247., 337., 459., 625., 855.,1050.,     \
           1050.,1050.,1050.,1050.,1050.,1050.,1050.,1050.      \
          ], Numeric.Float0)
  rendt = Numeric.array(                                        \
          [                                                     \
            100., 100., 100., 100., 100., 100., 100., 135.,     \
            185., 251., 342., 430., 500., 580., 730., 993.      \
          ], Numeric.Float0)
          
  ntemp = len(temp)
  if (len(temp) != len(lendt) or len(lendt) != len(rendt)):
    print "skewt_bkg: lengths of temp, lendt, rendt do not match"

#
#  Declare pressure values [hPa] and x coordinates of the endpoints 
#  of each isobar.  These x,y values are computed from the equations 
#  in the transform functions listed at the beginning of this program.
#  Refer to a skew-t diagram for reference if necessary.
#
  pres = Numeric.array(                              \
         [                                           \
          1050., 1000.,  850.,  700.,  500.,  400.,  \
           300.,  250.,  200.,  150.,  100.          \
         ], Numeric.Float0)
  xpl  = Numeric.array(                              \
         [                                           \
          -19.0, -19.0, -19.0, -19.0, -19.0, -19.0,  \
          -19.0, -19.0, -19.0, -19.0, -19.0          \
         ], Numeric.Float0)
  xpr  = Numeric.array(                              \
         [                                           \
           27.10, 27.10, 27.10, 27.10, 22.83, 18.60, \
           18.60, 18.60, 18.60, 18.60, 18.60         \
         ], Numeric.Float0)
  npres = len(pres)
  if (len(pres) != len(xpl) or len(xpl) != len(xpr)):
    print "skewt_bkg: lengths of pres, xpl, xpr do not match"

#
#  Declare adiabat values [C] and pressures where adiabats 
#  intersect the edge of the skew-t diagram.  Refer to a 
#  skew-t diagram if necessary.
#
  theta  = Numeric.array(                              \
           [                                           \
            -30., -20., -10.,   0.,  10.,  20.,  30.,  \
             40.,  50.,  60.,  70.,  80.,  90., 100.,  \
            110., 120., 130., 140., 150., 160., 170.   \
           ], Numeric.Float0)
  lendth = Numeric.array(                              \
           [                                           \
            880., 670., 512., 388., 292., 220., 163.,  \
            119., 100., 100., 100., 100., 100., 100.,  \
            100., 100., 100., 100., 100., 100., 100.   \
           ], Numeric.Float0)
  rendth = Numeric.array(                                    \
           [                                                 \
            1050., 1050., 1050., 1050., 1050., 1050., 1050., \
            1050., 1003.,  852.,  728.,  618.,  395.,  334., \
             286.,  245.,  210.,  180.,  155.,  133.,  115.  \
           ], Numeric.Float0)
  ntheta = len(theta)
  if (len(theta) != len(lendth) or len(lendth) != len(rendth)):
    print "skewt_bkg: lengths of pres, xpl, xpr do not match"

#
#  Declare moist adiabat values and pressures of the tops of the
#  moist adiabats.  All moist adiabats to be plotted begin at 1050 mb.
#
  pseudo = Numeric.array(                               \
           [                                            \
              32., 28., 24., 20., 16., 12.,  8.         \
           ], Numeric.Float0)
  lendps = Numeric.array(                               \
           [                                            \
              250., 250., 250., 250., 250., 250., 250.  \
           ], Numeric.Float0)
  npseudo= len(pseudo)          # moist adiabats

#
#  Declare mixing ratio lines.  All mixing ratio lines will begin
#  at 1050 mb and end at 400 mb.
#
  mixrat = Numeric.array(                      \
           [                                   \
              20., 12., 8., 5., 3., 2., 1.     \
           ], Numeric.Float0)
  nmix  = len(mixrat)           # mixing ratios

#
#  Declare local stuff: arrays/variables for storing x,y positions
#  during iterations to draw curved line, etc.
#
  sx    = Numeric.zeros(200, Numeric.Float0)
  sy    = Numeric.zeros(200, Numeric.Float0)
  xx    = Numeric.zeros(  2, Numeric.Float0)
  yy    = Numeric.zeros(  2, Numeric.Float0)
  m2f   =    3.2808            # meter-to-feet
  f2m   = 1./3.2808            # feet-to-meter

#
#  Define absolute x,y max/min bounds corresponding to the outer
#  edges of the diagram.  These are computed by inserting the appropriate
#  pressures and temperatures at the corners of the diagram.
#
                    # xmin = skewtx ( -33.60,skewty(1050.)) [t=deg C]
  xmin = -18.9551048  # xmin = skewtx (-109.10,skewty( 100.)) [t=deg C]
  xmax =  27.0973729  # xmax = skewtx (  51.75,skewty(1050.)) [t=deg C]
  ymax =  -0.9346217  # ymax = skewty (1050.)
  ymin =  44.0600000  # ymin = skewty ( 100.)

#
#  Specify arrays to hold corners of the diagram in x,y space.
#
  xc = Numeric.array(                               \
       [                                            \
          xmin, xmin, xmax, xmax, 18.60, 18.6, xmin \
       ], Numeric.Float0)
  yc = Numeric.array(                               \
       [                                            \
          ymin, ymax, ymax,  9.0, 17.53, ymin, ymin \
       ], Numeric.Float0)

#
#  Depending on how options are set, create Standard Atm Info.

  if (localOpts.sktDrawStandardAtm or  \
      localOpts.sktDrawHeightScale or  \
      localOpts.sktDrawWind):

#
#  U.S. Standard ATmosphere (km), source: Hess/Riegel.
#
    zsa = Numeric.array(range(0,17)).astype(Numeric.Float0)
    psa = Numeric.array(                                     \
          [                                                  \
           1013.25, 898.71, 794.90, 700.99, 616.29, 540.07,  \
            471.65, 410.46, 355.82, 307.24, 264.19, 226.31,  \
            193.93, 165.33, 141.35, 120.86, 103.30
          ], Numeric.Float0)
    tsa = Numeric.array(                                     \
          [                                                  \
              15.0,   8.5,    2.0,   -4.5,  -11.0,  -17.5,   \
             -24.0, -30.5,  -37.0,  -43.5,  -50.0,  -56.5,   \
             -56.5, -56.5,  -56.5,  -56.5,  -56.5            \
          ], Numeric.Float0)
    nlvl = len(psa)

#
#  Plot.
#
  if (localOpts.sktDrawColLine):
    colGreen  = "Green"
    colBrown  = "Brown"
    colTan    = "Tan"
  else:
    colGreen  = "Foreground"
    colBrown  = "Foreground"
    colTan    = "Foreground"

#
#  Draw outline of the skew-t, log p diagram.  Proceed in the upper left
#  corner of the diagram and draw counter-clockwise.  The array locations
#  below that are hardcoded refer to points on the background where the
#  skew-t diagram deviates from a rectangle, along the right edge.  Remember,
#  the set call defines a rectangle, so as long as the boundary is along
#  the edge of the set call, the points plotted are combinations of the min
#  and max x,y values in the set call.
#

  if (localOpts.sktDrawFahrenheit):
    tf = Numeric.array(range(-20,110,20),Numeric.Int)  # deg F
    tc = 0.55555 * (tf - 32.)                           # deg C
  else:
    tc = Numeric.array(range(-30,50,10)).astype(Numeric.Float0)

#
#  Don't draw the plot or advance the frame in the call to xy.
#  Specify location/size of the skewT diagram.
#
  xyOpts             = Resources()
  xyOpts.nglDraw     = False
  xyOpts.nglFrame    = False
  xyOpts.vpXF        = localOpts.vpXF
  xyOpts.vpYF        = localOpts.vpYF
  xyOpts.vpWidthF    = localOpts.vpWidthF
  xyOpts.vpHeightF   = localOpts.vpHeightF
  if (hasattr(localOpts,"nglMaximize")):
    xyOpts.nglMaximize = localOpts.nglMaximize

#
#  Define y tick mark labels.
#
  xyOpts.tmYLMode   = "Explicit"
  xyOpts.tmYLValues = skewty(pres[1:]) # skip 1050
  xyOpts.tmYLLabels =                            \
          [                                      \
           "1000", "850", "700", "500", "400",   \
            "300", "250", "200", "150", "100"    \
          ]

#
#  Define x tick mark labels.
#
  xyOpts.tmXBMode   = "Explicit"
  xyOpts.tmXBValues = skewtx (tc, skewty(1050.)) # transformed vals
  if (localOpts.sktDrawFahrenheit):
      xyOpts.tmXBLabels = tf                     # plot the nice deg F
  else:
      xyOpts.tmXBLabels = tc                     # plot the nice deg C

  xyOpts.trXMinF       = xmin
  xyOpts.trXMaxF       = xmax
  xyOpts.trYMinF       = ymax       # Note: ymin > ymax
  xyOpts.trYMaxF       = ymin
  xyOpts.xyComputeXMin = False
  xyOpts.xyComputeXMax = False
  xyOpts.xyComputeYMin = False
  xyOpts.xyComputeYMax = False
  xyOpts.tmXTOn        = False
  xyOpts.tmYROn        = False
  xyOpts.tmXTBorderOn  = False
  xyOpts.tmXBBorderOn  = False
  xyOpts.tmYRBorderOn  = False
  xyOpts.tmYLBorderOn  = False

#
#  Tune the plot.
#
  xyOpts.tmXBMajorLengthF        = 0.01
  xyOpts.tmXBMajorThicknessF     = 1.0         # default is 2.0
  xyOpts.tmXBMajorOutwardLengthF = 0.01
  xyOpts.tmXTMajorLengthF        = 0.
#
#  Suppress the tickmarks on the Y axis - need to turn
#  nglScale off, since it will override the tickmark settings
#  for the Y axis by making them the same as the X axis.
#
  xyOpts.tmYLMajorLengthF        = 0.
  xyOpts.tmYLMinorLengthF        = 0.
  xyOpts.nglScale                = False       

  xyOpts.tmXBLabelFontHeightF = 0.014
  xyOpts.tmYLLabelFontHeightF = xyOpts.tmXBLabelFontHeightF
  if (localOpts.sktDrawFahrenheit):
      xyOpts.tiXAxisString        = "Temperature (F)"
  else:
      xyOpts.tiXAxisString        = "Temperature (C)"
  xyOpts.tiXAxisFont          = localOpts.Font
  xyOpts.tiXAxisFontColor     = "Foreground"   # colTan
  xyOpts.tiXAxisFontHeightF   = 0.0125
  xyOpts.tiYAxisFont          = localOpts.Font
  xyOpts.tiYAxisString        = "P (hPa)"
  xyOpts.tiYAxisOffsetXF      = 0.0200
  xyOpts.tiYAxisOffsetYF      = -0.0200
  xyOpts.tiYAxisFontColor     = "Foreground"   # colTan
  xyOpts.tiYAxisFontHeightF   = xyOpts.tiXAxisFontHeightF
  xyOpts.tiMainString         = localOpts.tiMainString
  if (hasattr(localOpts,"tiMainFuncCode")):
    xyOpts.tiMainFuncCode     = localOpts.tiMainFuncCode
#
#  Uncomment if X or Y axis labels are allowed to be set by the user.
#
# if (hasattr(localOpts,"tiXAxisFuncCode")):
#   xyOpts.tiXAxisFuncCode    = localOpts.tiXAxisFuncCode
# if (hasattr(localOpts,"tiYAxisFuncCode")):
#   xyOpts.tiYAxisFuncCode    = localOpts.tiYAxisFuncCode
 
  xyOpts.tiMainFont           = localOpts.Font
  xyOpts.tiMainFontColor      = "Foreground"
  xyOpts.tiMainFontHeightF    = 0.025
  xyOpts.tiMainOffsetXF       = -0.1

#
#  Special for the right hand side.
#
  if (localOpts.sktDrawHeightScale):  
    xyOpts.trXMaxF      = skewtx (55. , skewty(1013.)) # extra wide
    xyOpts.tmYUseLeft   = False # Keep right axis independent of left.
    xyOpts.tmYRBorderOn = True
    xyOpts.tmYROn       = True  # Turn on right axis tick marks.
    xyOpts.tmYRLabelsOn = True  # Turn on right axis labels.
    xyOpts.tmYRLabelFontHeightF    = xyOpts.tmYLLabelFontHeightF
    xyOpts.tmYRMajorThicknessF     = xyOpts.tmXBMajorThicknessF
    xyOpts.tmYRMajorLengthF        = xyOpts.tmXBMajorLengthF
    xyOpts.tmYRMajorOutwardLengthF = xyOpts.tmXBMajorOutwardLengthF
    xyOpts.tmYRMinorOn             = False       # No minor tick marks.
    xyOpts.tmYRMode                = "Explicit"  # Define tick mark labels.

    zkm = Numeric.array(range(0,17)).astype(Numeric.Float0)
    pkm = ftcurv(zsa, psa, zkm)
    zft = Numeric.array(                                  \
          [                                               \
             0.,  2.,  4.,  6.,  8., 10., 12., 14., 16.,  \
            18., 20., 25., 30., 35., 40., 45., 50.        \
          ], Numeric.Float0)
    pft = ftcurv(zsa, psa, f2m * zft)  # p corresponding to zkm

    if (localOpts.sktDrawHeightScaleFt):
      znice =  zft
      pnice = skewty(pft)
      zLabel= "Height (1000 Feet)"
    else:
      znice = zkm
      pnice = skewty(pkm)
      zLabel= "Height (Km)"
#
#  At each "nice" pressure value, put a "height" value label.
#
    xyOpts.tmYRValues   = pnice
    xyOpts.tmYRLabels   = znice

#
#  Draw outline and the x and y axes.
#
  xyplot = xy (wks,xc,yc,xyOpts)

#
#  right *label* MUST be added AFTER xy created.
#
  if (localOpts.sktDrawHeightScale):
    txOpts                   = Resources()
    txOpts.txAngleF          = 270.
    txOpts.txFontColor       = "Foreground"   # colTan
    txOpts.txFontHeightF     = xyOpts.tmYLLabelFontHeightF
    xlab                     = skewtx (53., skewty(1013.))
    ylab                     = skewty (350.)
    text (wks,xyplot,zLabel,xlab,ylab,txOpts) 
    del txOpts

  if (localOpts.sktDrawColAreaFill):
    color1 = "PaleGreen1"           # "LightGreen"
    color2 = "MintCream"            # "Azure"
    gsOpts = Resources()
    for i in xrange(0,ntemp-1):
      if (i%2 == 0):                 # alternate colors
        gsOpts.gsFillColor = color1
      else:
        gsOpts.gsFillColor = color2

      nx = 3
      sy[0] = skewty(lendt[i  ])
      sx[0] = skewtx( temp[i  ], sy[0])
      sy[1] = skewty(lendt[i+1] )
      sx[1] = skewtx( temp[i+1], sy[1])
      sy[2] = skewty(rendt[i+1] )
      sx[2] = skewtx( temp[i+1], sy[2])
      sy[3] = skewty(rendt[i  ] )   
      sx[3] = skewtx( temp[i  ], sy[3])


#
#  Make sure the right sides line up with vertical line x=18.6 
#  and the left sides line up with vertical line x=xmin.
#
      if (i > 6 and i < 10):
        sx[2] = sx[3] = 18.6
      if (i < 6):
        sx[0] = sx[1] = -18.955

#
#  Special cases.
#
      if (temp[i] == -40.):
        nx = 5
        sy[0:nx+1] = Numeric.array(                               \
                     [                                            \
                       2.9966, ymax, ymax, 38.317, ymin, ymin     \
                     ], Numeric.Float0)                         
        sx[0:nx+1] = Numeric.array(                               \
                     [                                            \
                       xmin, xmin, -17.0476, 18.60, 18.60, 18.359 \
                     ], Numeric.Float0)                         
      if (temp[i] ==   0.):
        nx = 4
        sy[0:nx+1] = Numeric.array(                               \
                     [                                            \
                        ymax, ymax, 16.148, 17.53, 20.53          \
                     ], Numeric.Float0)                         
        sx[0:nx+1] = Numeric.array(                               \
                     [                                            \
                       -0.8476, 4.5523, 20.045, 18.60, 18.60      \
                     ], Numeric.Float0)                         
      if (temp[i] == 30.):
        nx = 4
        sy[0:nx+1] = Numeric.array(                               \
                     [                                            \
                       ymax, ymax, 6.021, 9.0, 10.422             \
                     ], Numeric.Float0)                         
        sx[0:nx+1] = Numeric.array(                               \
                     [                                            \
                       15.3523 , 20.7523 , 27.0974, 27.0974, 25.6525  \
                     ], Numeric.Float0)                         
      polygon(wks, xyplot, sx[0:nx+1], sy[0:nx+1], gsOpts)
#
#  Upper left triangle.
#
    gsOpts.gsFillColor = color2
    sy[0:3] = Numeric.array(              \
                 [                        \
                   ymin, ymin, 38.747     \
                 ], Numeric.Float0)
    sx[0:3] = Numeric.array(              \
                 [                        \
                   -14.04, -18.955, -18.955 \
                 ], Numeric.Float0)                         
    polygon(wks, xyplot, sx[0:3], sy[0:3], gsOpts)
#
#  Lower right triangle.
#
    gsOpts.gsFillColor = color2
    sy[0:3] = Numeric.array(              \
                 [                        \
                   ymax, 0.1334, ymax     \
                 ], Numeric.Float0)
    sx[0:3] = Numeric.array(              \
                 [                        \
                   xmax, xmax, 26.1523    \
                 ], Numeric.Float0)                         
    polygon(wks, xyplot, sx[0:3],sy[0:3],gsOpts)
    del gsOpts

#
#  Draw diagonal isotherms.
#  [brown with labels interspersed at 45 degree angle]
#  http://ngwww.ucar.edu/ngdoc/ng/ref/hlu/obj/GraphicStyle.obj.html
#
  if (localOpts.sktDrawIsotherm):
    gsOpts                   = Resources()
    gsOpts.gsLineDashPattern = 0            # solid
    gsOpts.gsLineColor       = colTan
    gsOpts.gsLineThicknessF  = 1.0
    #gsOpts.gsLineLabelFontColor   = colTan
    #gsOpts.gsLineLabelFontHeightF = 0.0125

    txOpts                   = Resources()
    txOpts.txAngleF          = 45.
    txOpts.txFontColor       = gsOpts.gsLineColor
    txOpts.txFontHeightF     = 0.0140
    txOpts.txFontThicknessF  = 1.0

    for i in range(0,len(temp)-2):
      yy[1] = skewty(rendt[i])
      xx[1] = skewtx( temp[i], yy[1])
      yy[0] = skewty(lendt[i])
      xx[0] = skewtx( temp[i], yy[0])
      #gsOpts.gsLineLabelString  = int(temp[i])
      polyline(wks, xyplot, xx, yy, gsOpts)

      xlab  = xx[1] + 0.625
      ylab  = yy[1] + 0.55
      label = int(temp[i])
      text(wks, xyplot, str(label), xlab, ylab, txOpts)
    del gsOpts
    del txOpts

#
#  Draw horizontal isobars.
#
  if (localOpts.sktDrawIsobar):
      gsOpts                   = Resources()
      gsOpts.gsLineDashPattern = 0            # solid
      gsOpts.gsLineColor       = colTan
      gsOpts.gsLineThicknessF  = 1.0
      #gsOpts.gsLineLabelFontColor    = colTan
      #gsOpts.gsLineLabelFontHeightF  = 0.0125

      for i in range(0,npres):
        xx[0] = xpl[i]
        xx[1] = xpr[i]
        ypl   = skewty(pres[i])
        yy[0] = ypl
        yy[1] = ypl
        polyline(wks, xyplot, xx, yy, gsOpts)
      del gsOpts

#
#  Draw saturation mixing ratio lines.  These lines run 
#  between 1050 and 400 mb.  The 20 line intersects the 
#  sounding below 400 mb, thus a special case is made for 
#  it.  The lines are dashed green.  The temperature where 
#  each line crosses 400 mb is computed in order to get x,y 
#  locations of the top of the lines.
#
  if (localOpts.sktDrawMixRatio):
    gsOpts                   = Resources()   # polyline graphic style opts
    gsOpts.gsLineThicknessF  = 1.0
#
#  saturation  mix ratio only.
#
    gsOpts.gsLineDashPattern = 2        
    gsOpts.gsLineColor       = colGreen 

    txOpts                   = Resources()
    txOpts.txAngleF          = 65.     
    txOpts.txFontColor       = colGreen
    txOpts.txFontHeightF     = 0.0100  

    yy[1] = skewty( 400.)    # y at top [right end of slanted line]
    yy[0] = skewty(1000.)    # y at bottom of line [was 1050.]

    for i in range(0,nmix):
      if (mixrat[i] == 20.):
        yy[1] = skewty(440.)
        tmix  = dtmrskewt(mixrat[i],440.)
      else:
        yy[1] = skewty(400.)
        tmix  = dtmrskewt(mixrat[i],400.)
      xx[1] = skewtx(tmix,yy[1])
      tmix  = dtmrskewt(mixrat[i],1000.)   # was 1050
      xx[0] = skewtx(tmix,yy[0])
      polyline (wks,xyplot,xx,yy,gsOpts)   # dashed green

      xlab  = xx[0]-0.25
      ylab  = yy[0]-0.45
      label = int(mixrat[i])
      text(wks, xyplot, str(label), xlab, ylab, txOpts)
    del gsOpts
    del txOpts

#
#  Draw dry adiabats.  Iterate in 10 mb increments to compute the x,y
#  points on the curve.
#
  if (localOpts.sktDrawDryAdiabat):
    gsOpts                   = Resources()
    gsOpts.gsLineDashPattern = 0            
    gsOpts.gsLineColor       = colTan     
    gsOpts.gsLineThicknessF  = 1.0

    txOpts                   = Resources()
    txOpts.txAngleF          = 300.
    txOpts.txFontColor       = colTan
    txOpts.txFontHeightF     = 0.01
    txOpts.txFontThicknessF  = 1.0

    pinc = 10.
    for i in range(0,ntheta):
      p = lendth[i]-pinc
      for j in range(0,len(sy)):
        p = p+pinc
        if (p > rendth[i]):
          sy[j] = skewty(rendth[i])
          t     = dtdaskewt(theta[i],p)   # get temp on dry adiabat at p
          sx[j] = skewtx(t,sy[j])
          break 
        sy[j] = skewty(p)
        t     = dtdaskewt(theta[i],p)
        sx[j] = skewtx(t,sy[j])
      #polyline (wks,xyplot,sx[:j-1],sy[:j-1],gsOpts)     # whole line  

      if (theta[i] < 170.):
        polyline (wks,xyplot,sx[1:j],sy[1:j],gsOpts)  # label room  
        ylab  = skewty(lendth[i]+5.)
        t     = dtdaskewt(theta[i],lendth[i]+5.)
        xlab  = skewtx(t,ylab)
        label = int(theta[i])
        text(wks,xyplot,str(label),xlab,ylab,txOpts)
      else:                                            # no label
        polyline (wks,xyplot,sx[:j],sy[:j],gsOpts)     # whole line
    del gsOpts
    del txOpts

#
#  Draw moist adiabats up to 230 [was 250] mb.
#  Draw the lines.  Dterate in 10 mb increments from 1060 mb.
#
  if (localOpts.sktDrawMoistAdiabat):
    gsOpts                   = Resources()
    gsOpts.gsLineColor       = colGreen
    gsOpts.gsLineThicknessF  = 0.5
    gsOpts.gsLineDashPattern = 0

    txOpts                   = Resources()
    txOpts.txAngleF          = 0.
    txOpts.txFontColor       = colGreen
    txOpts.txFontHeightF     = 0.0125
    txOpts.txFontThicknessF  = 1.0

    pinc = 10.

    for i in range(0,npseudo):
      p = 1060.
      for j in range (0, len (sy)):
        p = p - pinc
        if (p < 230.):     # was "250"
          break
        sy[j] = skewty(p)
        t     = dsatlftskewt(pseudo[i],p)    # temp on moist adiabat at p.
        sx[j] = skewtx(t,sy[j])

      polyline (wks, xyplot, sx[:j-1], sy[:j-1], gsOpts)

      ylab  = skewty(p + 0.5*pinc)
      t     = dsatlftskewt(pseudo[i], p + 0.75*pinc)
      xlab  = skewtx(t,ylab)
      label = int(pseudo[i])     # 9 Feb 99 fix
      text(wks, xyplot, str(label), xlab, ylab, txOpts)

    del gsOpts
    del txOpts

  if (localOpts.sktDrawStandardAtm):
      gsOpts                   = Resources()
      gsOpts.gsLineColor       = colTan   
      gsOpts.gsLineThicknessF  = localOpts.sktDrawStandardAtmThk
      gsOpts.gsLineDashPattern = 0             

      for i in range(0,nlvl):
         sy[i] = skewty(psa[i])
         sx[i] = skewtx(tsa[i], sy[i])

      polyline (wks, xyplot, sx[0:nlvl], sy[0:nlvl], gsOpts)
      del gsOpts

#
#  Draw vertical line upon which to plot wind barbs.
#
  if (localOpts.sktDrawWind):
      gsOpts = Resources()
      gsOpts.gsLineColor       = "Foreground"
      gsOpts.gsLineThicknessF  = 0.5
      gsOpts.gsLineDashPattern = 0
      gsOpts.gsMarkerIndex     = 4   # "hollow_circle"=> std pres
      gsOpts.gsMarkerColor     = "Foreground"

      presWind      = pres
      presWind[0]   = 1013.          # override 1050
      xWind         = skewtx (45. , skewty(presWind[0])) 
      sx[0:npres] = xWind            # "x" location of wind plot
      sy[0:npres] = skewty(presWind).astype(Numeric.Float0)
      polyline   (wks, xyplot, sx[0:npres], sy[0:npres], gsOpts)
      polymarker (wks, xyplot, sx[1:npres], sy[1:npres], gsOpts)
                                     # zwind => Pibal reports
      zftWind = Numeric.array(                                    \
                              [0.,  1.,  2.,  3.,  4.,  5.,  6.,  \
                               7.,  8.,  9., 10., 12., 14., 16.,  \
                              18., 20., 25., 30., 35., 40., 45.,  \
                              50.], Numeric.Float0)
      zkmWind = zftWind*f2m
      pkmWind = ftcurv(zsa, psa, zkmWind)
      nzkmW   = len(zkmWind)

      sx[0:nzkmW]  = xWind              # "x" location of wind plot
      sy[0:nzkmW]  = skewty(pkmWind).astype(Numeric.Float0)

      gsOpts.gsMarkerIndex      = 16     # "circle_filled" -> Pibal
      gsOpts.gsMarkerSizeF      = 0.0035 # 0.007 is default
      gsOpts.gsMarkerThicknessF = 0.5    # 1.0 is default
      polymarker (wks, xyplot, sx[0:nzkmW], sy[0:nzkmW], gsOpts)
      del gsOpts

  return xyplot

def skewt_plt(wks, skewt_bkgd, P, TC, TDC, Z, WSPD, WDIR, 
             dataOpts=None):
#
#  p    =  pressure     [mb / hPa]
#  tc   = temperature   [C]
#  tdc  = dew pt temp   [C]
#  z    = geopotential  [gpm]
#  wspd = wind speed    [knots or m/s]
#  wdir = meteorological wind direction
#
#
#  Set missing values for variables
#  used in plotting the sounding and 
#  in calculating thermodynamic quantities.
#
  if (hasattr(dataOpts,"sktPmissingV")):
    Pmissing = dataOpts.sktPmissingV
  else:
    Pmissing = -999.
#
  if (hasattr(dataOpts,"sktTCmissingV")):
    TCmissing = dataOpts.sktTCmissingV
  else:
    TCmissing = -999.
#
  if (hasattr(dataOpts,"sktTDCmissingV")):
    TDCmissing = dataOpts.sktTDCmissingV
  else:
    TDCmissing = -999.
#
  if (hasattr(dataOpts,"sktZmissingV")):
    Zmissing = dataOpts.sktZmissingV
  else:
    Zmissing = -999.
#
  if (hasattr(dataOpts,"sktWSPDmissingV")):
    WSPDmissing = dataOpts.sktWSPDmissingV
  else:
    WSPDmissing = -999.
#
  if (hasattr(dataOpts,"sktWDIRmissingV")):
    WDIRmissing = dataOpts.sktWDIRmissingV
  else:
    WDIRmissing = -999.
#
  if (hasattr(dataOpts,"sktHmissingV")):
    Hmissing = dataOpts.sktHmissingV
  else:
    Hmissing = -999.

  mv0 = Numeric.logical_and(Numeric.logical_not(ismissing( P,Pmissing)),   \
                            Numeric.logical_not(ismissing(TC,TCmissing)))
  mv1 = Numeric.logical_and(mv0,Numeric.logical_not(ismissing(TDC,TDCmissing)))
  mv2 = Numeric.logical_and(mv1,Numeric.greater_equal(P,100.))
  idx = ind(mv2)
  del mv0,mv1,mv2
  p   = Numeric.take(  P,idx)
  tc  = Numeric.take( TC,idx)
  tdc = Numeric.take(TDC,idx)

#
#  Local options describing data and ploting.
#
  localOpts           = Resources()
  localOpts.sktPrintZ    = True     # print geopotential (Z) on skewT diagram
  localOpts.sktPlotWindP = True     # plot wind barbs at p lvls
  localOpts.sktWspdWdir  = True     # wind speed and dir [else: u,v]
  localOpts.sktPlotWindH = False    # plot wind barbs at h lvls [pibal; special]
  localOpts.sktHspdHdir  = True     # wind speed and dir [else: u,v]
  localOpts.sktThermoInfo= True     # print thermodynamic info
  localOpts.sktCape      = True     # plot CAPE parcel profile if cape > 0
  localOpts.sktParcel    = 0        # subscript corresponding to initial parcel

#
#  Override localOpts attributes with dataOpts attributes for
#  dataOpts attributes that overlap localOpts attributes; set
#  new localOpts attributes for dataOpts attributes that do not
#  overlap localOpts attributes.
#
  if (dataOpts != None):
    if (not isinstance(dataOpts,Resources)):
      print "skewt_plt: last argument must be an Nlg Resources instance."
      return None
    OptsAtts = crt_dict(dataOpts)
    if (len(OptsAtts) != 0):
      for key in OptsAtts.keys():
        setattr(localOpts,key,OptsAtts[key])

  vpXF                 = get_float(skewt_bkgd,"vpXF")
  vpYF                 = get_float(skewt_bkgd,"vpYF")
  vpWidthF             = get_float(skewt_bkgd,"vpWidthF")
  vpHeightF            = get_float(skewt_bkgd,"vpHeightF")
  tiMainFont           = get_string(skewt_bkgd,"tiMainFont")
  tiMainFontHeightF    = get_float(skewt_bkgd,"tiMainFontHeightF")
  tiMainOffsetXF       = get_float(skewt_bkgd,"tiMainOffsetXF")
  tmYLLabelFontHeightF = get_float(skewt_bkgd,"tmYLLabelFontHeightF")

#
#  Specify various colors.
#
  colForeGround     = "Foreground"
  sktcolTemperature = "Foreground"
  sktcolDewPt       = "RoyalBlue"
  sktcolPpath       = "Red"
  sktcolZLabel      = "Foreground"
  sktcolWindP       = "Black"   
  sktcolWindZ       = "Blue"   
  sktcolWindH       = "Magenta"    
  sktcolThermoInfo  = "Sienna"

#
#  Change defaults.
#
  if (hasattr(localOpts,"sktcolTemperature")):
    sktcolTemperature  = localOpts.sktcolTemperature
  if (hasattr(localOpts,"sktcolDewPt")):
    sktcolDewPt  = localOpts.sktcolDewPt
  if (hasattr(localOpts,"sktcolPpath")):
    sktcolPpath  = localOpts.sktcolPpath 
  if (hasattr(localOpts,"sktcolZLabel")):
    sktcolZLabel  = localOpts.sktcolZLabel 
  if (hasattr(localOpts,"sktcolWindP")):
    sktcolWindP  = localOpts.sktcolWindP 
  if (hasattr(localOpts,"sktcolWindZ")):
    sktcolWindZ  = localOpts.sktcolWindZ 
  if (hasattr(localOpts,"sktcolWindH")):
    sktcolWindH  = localOpts.sktcolWindH 
  if (hasattr(localOpts,"sktcolThermoInfo")):
    sktcolThermoInfo  = localOpts.sktcolThermoInfo 


#
#  Graphics style settings for the polyline draw.
#
  gsOpts                   = Resources()
  gsOpts.gsLineDashPattern = 0      # solid (default)
  gsOpts.gsLineThicknessF  = 3.0    # make thicker

  if (hasattr(localOpts,"gsLineDashPattern")):
    gsOpts.gsLineDashPattern = localOpts.gsLineDashPattern
  if (hasattr(localOpts,"gsLineThicknessF")):
    gsOpts.gsLineThicknessF = localOpts.gsLineThicknessF

  yp   = skewty(p)
  xtc  = skewtx(tc, yp)
  gsOpts.gsLineColor  = sktcolTemperature
  polyline(wks, skewt_bkgd, xtc, yp, gsOpts)

  xtdc = skewtx(tdc, yp)
  gsOpts.gsLineColor  = sktcolDewPt
  polyline(wks, skewt_bkgd, xtdc, yp, gsOpts)

  del gsOpts

  if (localOpts.sktThermoInfo):
    nP   = localOpts.sktParcel  # default is the lowest level [0]
    nlvls= len(p)
    plcl = -999.             # p (hPa) Lifting Condensation Lvl (lcl)
    tlcl = -999.             # temperature (C) of lcl
    plcl, tlcl = dptlclskewt(p[nP],tc[nP],tdc[nP])
    shox = dshowalskewt(p,tc,tdc,nlvls)     # Showwalter Index

    pwat = dpwskewt(tdc,p,nlvls)            # precipitable water (cm)

    iprnt= 0                                # debug only (>0)
    nlLcl= 0                              
    nlLfc= 0
    nlCross= 0

    if (hasattr(localOpts,"sktTCmissingV")):
      TCmissing = localOpts.sktTCmissingV
      if (ismissing(tc,TCmissing)):
        print "skewt_plt: tc (temperature) cannot have missing values if sktThermoInfo is True."
        return None
    TCmissing = -999.
    cape,tpar,nlLcl,nlLfc,nlCross  =  \
         dcapethermo(p,tc,len(p),plcl,iprnt,TCmissing)
#        nglf.dcapethermo(p,tc,len(p),plcl,iprnt,TCmissing)

                                            # 0.5 is for rounding
    info = " Plcl="     + str(int(plcl+0.5)) \
         + " Tlcl[C]="  + str(int(tlcl+0.5)) \
         + " Shox="     + str(int(shox+0.5)) \
         + " Pwat[cm]=" + str(int(pwat+0.5)) \
         + " Cape[J]= " + str(int(cape))

    txOpts                   = Resources()
    txOpts.txAngleF          = 0.
    txOpts.txFont            = tiMainFont
    txOpts.txFontColor       = sktcolThermoInfo
    txOpts.txFontHeightF     = 0.5*tiMainFontHeightF
    xinfo                    = vpXF  + 0.5*vpWidthF + tiMainOffsetXF
    yinfo                    = vpYF  + 0.5*tiMainFontHeightF
    text_ndc (wks,info,xinfo,yinfo,txOpts)
    del txOpts

    if (localOpts.sktCape and cape > 0.):
      gsOpts                   = Resources()
      gsOpts.gsLineColor       = sktcolPpath
      gsOpts.gsLineDashPattern = 1         # 14
      gsOpts.gsLineThicknessF  = 2.0

      yp   = skewty(p)
      xtp  = skewtx(tpar, yp)
      polyline(wks, skewt_bkgd, xtp[nlLfc:nlCross+1], \
                   yp[nlLfc:nlCross+1], gsOpts)
      del gsOpts

#
#  Print geopotential if requested.
#
  if (localOpts.sktPrintZ):
    txOpts               = Resources()
    txOpts.txAngleF      = 0.
    txOpts.txFontColor   = sktcolZLabel
    txOpts.txFontHeightF = 0.9*tmYLLabelFontHeightF
#
#  Levels at which Z is printed.
#
    Pprint = Numeric.array(                                 \
                           [1000., 850., 700., 500., 400.,  \
                             300., 250., 200., 150., 100.   \
                           ], Numeric.Float0)

    yz = skewty(1000.)
    xz = skewtx(-30., yz)        # constant "x"
    for nl in range(len(P)):

     if ( Numeric.logical_not(ismissing(P[nl],Pmissing)) and   \
          Numeric.logical_not(ismissing(Z[nl],Zmissing)) and   \
          Numeric.sometrue(Numeric.equal(Pprint,P[nl])) ):
       yz  = skewty(P[nl])
       text(wks, skewt_bkgd, str(int(Z[nl])), xz, yz, txOpts)
    del txOpts

  if (localOpts.sktPlotWindP):
    gsOpts                   = Resources()
    gsOpts.gsLineThicknessF  = 1.0

#
#  Check if WSPD has a missing value attribute specified and
#  that not all WSPD values are missing values.
#
    if (Numeric.logical_not(Numeric.alltrue(ismissing(WSPD,WSPDmissing)))):
#
#  IDW - indices where P/WSPD/WDIR are all not missing.
#
      mv0 = Numeric.logical_and(Numeric.logical_not(ismissing(P,Pmissing)), \
                     Numeric.logical_not(ismissing(WSPD,WSPDmissing)))
      mv1 = Numeric.logical_and(mv0,  \
                     Numeric.logical_not(ismissing(WDIR,WDIRmissing)))
      mv2 = Numeric.logical_and(mv1,Numeric.greater_equal(P,100.))
      IDW = ind(mv2)
      if (hasattr(localOpts,"sktWthin") and localOpts.sktWthin > 1):
        nThin = localOpts.sktWthin
        idw   = IDW[::nThin]
      else:
        idw   = IDW

      pw  = Numeric.take(P,idw)

      wmsetp("wdf", 1)         # meteorological dir (Sep 2001)

#
#  Wind speed and direction.
#
      if (localOpts.sktWspdWdir):
        dirw = 0.017453 * Numeric.take(WDIR,idw)

        up   = -Numeric.take(WSPD,idw) * Numeric.sin(dirw)
        vp   = -Numeric.take(WSPD,idw) * Numeric.cos(dirw)
      else:
        up   = Numeric.take(WSPD,idw)      # must be u,v components
        vp   = Numeric.take(WDIR,idw)

      wbcol = wmgetp("col")                # get current wbarb color
      wmsetp("col",get_named_color_index(wks,sktcolWindP)) # set new color
      ypWind = skewty(pw)
      xpWind = Numeric.ones(len(pw),Numeric.Float0)
#
#  Location of wind barb.
#
      xpWind = skewtx(45., skewty(1013.)) * xpWind
      wmbarb(wks, xpWind, ypWind, up, vp)
      wmsetp("col",wbcol)               # restore initial color.

      mv0 = Numeric.logical_and(Numeric.logical_not(ismissing( Z,Zmissing)), \
                      Numeric.logical_not(ismissing(WSPD,WSPDmissing)))
      mv1 = Numeric.logical_and(mv0, \
                      Numeric.logical_not(ismissing(WDIR,WDIRmissing)))
      mv2 = Numeric.logical_and(mv1,ismissing(P,Pmissing))
      idz = ind(mv2)

      if (len(idz) > 0):
        zw  = Numeric.take(Z,idz)
        if (localOpts.sktWspdWdir):          # wind spd,dir (?)
          dirz = 0.017453 * Numeric.take(WDIR,idz)
          uz   = -Numeric.take(WSPD,idz) * Numeric.sin(dirz)
          vz   = -Numeric.take(WSPD,idz) * Numeric.cos(dirz)
        else:
          uz   = WSPD(idz)              # must be u,v components
          vz   = WDIR(idz)

#
#  idzp flags where Z and P have non-missing values.
#
        mv0  = Numeric.logical_not(ismissing(P,Pmissing))
        mv1  = Numeric.logical_not(ismissing(Z,Zmissing))
        mv2  = Numeric.logical_and(mv0,mv1)
        idzp = ind(mv2)
        Zv   = Numeric.take(Z,idzp)
        Pv   = Numeric.take(P,idzp)
        pz   = ftcurv(Zv,Pv,zw)               # map zw to p levels.

        wbcol = wmgetp("col")
        wmsetp("col",get_named_color_index(wks,sktcolWindZ)) 
        yzWind = skewty(pz)
        xzWind = Numeric.ones(len(pz),Numeric.Float0)
        xzWind = skewtx(45., skewty(1013.)) * xzWind
 
        wmbarb(wks, xzWind, yzWind, uz, vz )
        wmsetp("col",wbcol)

#
#  Allows other winds to be input as attributes of sounding.
#
  if (localOpts.sktPlotWindH):
    if (hasattr(dataOpts,"sktHeight") and hasattr(dataOpts,"sktHspd") and  \
        hasattr(dataOpts,"sktHdir")):
      dimHeight = len(dataOpts.sktHeight)
      dimHspd   = len(dataOpts.sktHspd  )
      dimHdir   = len(dataOpts.sktHdir  )
      if (dimHeight == dimHspd and dimHeight == dimHdir and \
          Numeric.logical_not(Numeric.alltrue(ismissing(dataOpts.sktHeight,Hmissing)))):
        if (localOpts.sktHspdHdir):
          dirh = 0.017453 * dataOpts.sktHdir
          uh   = -dataOpts.sktHspd * Numeric.sin(dirh)
          vh   = -dataOpts.sktHspd * Numeric.cos(dirh)
        else:
          uh   = dataOpts.sktHspd
          vh   = dataOpts.sktHdir

        mv0  = Numeric.logical_not(ismissing(P,Pmissing))
        mv1  = Numeric.logical_not(ismissing(Z,Zmissing))
        mv2  = Numeric.logical_and(mv0,mv1)
        idzp = ind(mv2)
        Zv   = Numeric.take(Z,idzp)
        if (len(Zv) == 0):
          print "Warning - skewt_plt: attempt to plot wind barbs at specified heights when there are no coordinates where pressure and geopotential are both defined."
        else:
          Pv   = Numeric.take(P,idzp)
          ph   = ftcurv(Zv,Pv,dataOpts.sktHeight)

          wbcol = wmgetp("col")             # get current color index
          wmsetp("col",get_named_color_index(wks,sktcolWindH)) # set new color
  
          yhWind = skewty(ph)
          xhWind = Numeric.ones(len(ph), Numeric.Float0)
          xhWind = skewtx(45., skewty(1013.)) * xhWind
          if (yhWind != None and xhWind != None):
            wmbarb(wks, xhWind, yhWind, uh, vh )
          wmsetp("col",wbcol)              # reset to initial color value
    else:
      print ("skewt_plt: Opts.sktPlotWindH = True but dataOpts.sktHeight/Hspd/Hdir are missing")
  
  return skewt_bkgd

def streamline(wks,uarray,varray,rlistc=None):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to VectorField and those that apply to
#  StreamlinePlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
#
# In addition, if this plot is potentially going to be overlaid
# on an Irregular Plot Class (in order to lineariize or logize it)
# then we need to keep track of all the tickmark resources, because
# we'll have to reapply them to the IrregularPlot class.
#
      if(key[0:2] == "vp" or key[0:2] == "tm" or key[0:6] == "pmTick"):
        rlist3[key] = rlist[key]
    
  set_streamline_res(rlist,rlist2)    # Set some addtl streamline resources
  set_tickmark_res(rlist,rlist3)      # Set some addtl tickmark resources

#
#  Call the wrapped function and return.
#
  strm = streamline_wrap(wks,uarray,varray,"double","double",            \
                         uarray.shape[0],uarray.shape[1],0,              \
                         pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(),\
                         rlist1,rlist2,rlist3,pvoid())
  del rlist
  del rlist1
  del rlist2
  del rlist3
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

  set_map_res(rlist,rlist3)           # Set some addtl map resources
  set_streamline_res(rlist,rlist2)    # Set some addtl streamline resources
    
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

def update_workstation(obj):
  NhlUpdateWorkstation(int_id(obj))

def vector(wks,uarray,varray,rlistc=None):

  set_spc_defaults(1)
  rlist = crt_dict(rlistc)  
 
#  Separate the resource dictionary into those resources
#  that apply to VectorField and those that apply to
#  VectorPlot.
#
  rlist1 = {}
  rlist2 = {}
  rlist3 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist2[key] = rlist[key]
#
# In addition, if this plot is potentially going to be overlaid
# on an Irregular Plot Class (in order to lineariize or logize it)
# then we need to keep track of all the tickmark resources, because
# we'll have to reapply them to the IrregularPlot class.
#
      if(key[0:2] == "vp" or key[0:2] == "tm" or key[0:6] == "pmTick"):
        rlist3[key] = rlist[key]
    
  set_vector_res(rlist,rlist2)        # Set some addtl vector resources
  set_labelbar_res(rlist,rlist2,True) # Set some addtl labelbar resources
  set_tickmark_res(rlist,rlist3)      # Set some addtl tickmark resources

#
#  Call the wrapped function and return.
#
  ivct = vector_wrap(wks,uarray,varray,"double","double",             \
                     uarray.shape[0],uarray.shape[1],0,               \
                     pvoid(),"",0,pvoid(),"", 0, 0, pvoid(), pvoid(), \
                     rlist1,rlist2,rlist3,pvoid())
  del rlist
  del rlist1
  del rlist2
  del rlist3
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

  set_map_res(rlist,rlist3)           # Set some addtl map resources
  set_vector_res(rlist,rlist2)        # Set some addtl vector resources
  set_labelbar_res(rlist,rlist2,True) # Set some addtl labelbar resources
    
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
  rlist4 = {}
  for key in rlist.keys():
    if (key[0:2] == "vf"):
      rlist1[key] = rlist[key]
    elif(key[0:2] == "sf"):
      rlist2[key] = rlist[key]
    elif(key[0:3] == "ngl"):
      set_spc_res(key[3:],rlist[key])      
    else:
      rlist3[key] = rlist[key]
#
# In addition, if this plot is potentially going to be overlaid
# on an Irregular Plot Class (in order to lineariize or logize it)
# then we need to keep track of all the tickmark resources, because
# we'll have to reapply them to the IrregularPlot class.
#
      if(key[0:2] == "vp" or key[0:2] == "tm" or key[0:6] == "pmTick"):
        rlist4[key] = rlist[key]
    
  set_vector_res(rlist,rlist3)        # Set some addtl vector resources
  set_labelbar_res(rlist,rlist3,True) # Set some addtl labelbar resources
  set_tickmark_res(rlist,rlist4)      # Set some addtl tickmark resources

#
#  Call the wrapped function and return.
#
  ivct = vector_scalar_wrap(wks,uarray,varray,tarray,  \
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
    
  set_map_res(rlist,rlist4)           # Set some addtl map resources
  set_vector_res(rlist,rlist3)        # Set some addtl vector resources
  set_labelbar_res(rlist,rlist3,True) # Set some addtl labelbar resources

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

def vinth2p (dati, hbcofa, hbcofb, plevo, psfc, intyp, p0, ii, kxtrp):     
#
#  Argument plevi is calculated in the Fortran code, just zero it out below.
#
#  Argument ii is not used at this time - it is set to 1 in the
#    call to the Fortran routine.
#
  if (len(dati.shape) > 4):
    print "\n vinth2p: requires a minimum of 3 dimensions [lev]x[lat]x[lon] \n"\
          "          and a maximum of 4 dimensions [time]x[lev]x[lat]x[lon] - \n"\
          "          an array with " + str(len(dati.shape)) + " dimensions was entered.\n"
    return None
  if (len(dati.shape) == 3):
    plevi = Numeric.zeros(dati.shape[0]+1,Numeric.Float)
    return NglVinth2p (dati, len(plevo), dati.shape[1], dati.shape[2],    \
                       hbcofa, hbcofb, p0, plevi, plevo, intyp,           \
                       1, psfc, 1.e30, kxtrp, dati.shape[0]+1, dati.shape[0])
#
#  The case with an input array having four dimensions is resolved
#  by calling the 3D case over the time variavle.
#
  elif (len(dati.shape) == 4):
    if (type(dati[0,0,0,0]) == type(Numeric.array([0.],Numeric.Float))):
#
#  Delete ar_out if it exists, and define it to be the correct
#  shape to hold the output.
#
      try:
        del ar_out
      except:
        pass
      ar_out = Numeric.zeros([dati.shape[0],len(plevo),dati.shape[2],  \
                              dati.shape[3]],Numeric.Float)
      plevi  = Numeric.zeros(dati.shape[1]+1,Numeric.Float)
    else:
      print "vinth2p: input data must be a Numeric array"
      return None
    for i in xrange(dati.shape[0]):
      ar_out[i,:,:,:] = NglVinth2p (dati[i,:,:,:], len(plevo),              \
                        dati.shape[2], dati.shape[3], hbcofa, hbcofb,       \
                        p0, plevi, plevo, intyp, 1, psfc[i,:,:], 1.e30,     \
                        kxtrp, dati.shape[1]+1, dati.shape[1])
    return ar_out
  else:
    print "vinth2p - invalid input data array."
    return None

def wmbarb(wks,x,y,u,v):
#
#  Get the GKS workstaton ID.
#
  gksid = get_integer(wks,"wkGksWorkId")

#
#  Process depending on whether we have scalar coordinates,
#  Numeric arrays, or Python lists or tuples.
#
  t = type(Numeric.array([0],Numeric.Int0))   #  Type for Numeric arrays.
  if (type(x) == t):
    if ( (type(y) != t) or (type(u) != t) or (type(v) != t)):
      print "wmbarb: If any argument is a Numeric array, they must all be."
      return 1
    rx = Numeric.ravel(x)
    ry = Numeric.ravel(y)
    ru = Numeric.ravel(u)
    rv = Numeric.ravel(v)
    for i in range(len(rx)):
      c_wmbarbp(gksid,rx[i],ry[i],ru[i],rv[i])
  elif(type(x) == types.ListType):
    l = types.ListType
    if ( (type(y) != l) or (type(u) != l) or (type(v) != l)):
      print "wmbarb: If any argument is a Python list, they must all be."
      return 1
    for i in range(len(x)):
      c_wmbarbp(gksid,x[i],y[i],u[i],v[i])
  elif(type(x) == types.TupleType):
    l = types.TupleType
    if ( (type(y) != l) or (type(u) != l) or (type(v) != l)):
      print "wmbarb: If any argument is a Python tuple, they must all be."
      return 1
    for i in range(len(x)):
      c_wmbarbp(gksid,x[i],y[i],u[i],v[i])
  elif (type(x)==types.IntType or type(x)==types.LongType or \
        type(x)==types.FloatType):
    c_wmbarbp(gksid,x,y,u,v)
  return 0

def wmbarbmap(wks,x,y,u,v):
  ezf = wmgetp("ezf")
  wdf = wmgetp("wdf")
  wmsetp("ezf",1)
  wmsetp("wdf",1)
  wmbarb(wks,x,y,u,v)
  wmsetp("ezf",ezf)
  wmsetp("wdf",wdf)

def wmgetp(pname):
  iparms = [                                                                \
             "alo", "aoc", "asc", "awc", "cbc", "cc1", "cc2", "cc3", "cfc", \
             "col", "dbc", "dtc", "hib", "hic", "hif", "his", "lc1", "lc2", \
             "lc3", "lob", "lof", "los", "mxs", "nbz", "nms", "pai", "rbs", \
             "rc1", "rc2", "rc3", "rc4", "rc5", "rev", "rfc", "rls", "ros", \
             "sc1", "sc2", "sc3", "sc4", "slf", "sty", "t1c", "t2c", "wbf", \
             "wfc", "wty", "ezf", "smf", "loc", "wdf", "unt",               \
             "ALO", "AOC", "ASC", "AWC", "CBC", "CC1", "CC2", "CC3", "CFC", \
             "COL", "DBC", "DTC", "HIB", "HIC", "HIF", "HIS", "LC1", "LC2", \
             "LC3", "LOB", "LOF", "LOS", "MXS", "NBZ", "NMS", "PAI", "RBS", \
             "RC1", "RC2", "RC3", "RC4", "RC5", "REV", "RFC", "RLS", "ROS", \
             "SC1", "SC2", "SC3", "SC4", "SLF", "STY", "T1C", "T2C", "WBF", \
             "WFC", "WTY", "EZF", "SMF", "LOC", "WDF", "UNT"                \
           ] 

  rparms = [                                                                \
             "arc", "ard", "arl", "ars", "beg", "bet", "cht", "cmg", "cs1", \
             "cs2", "dts", "dwd", "end", "lin", "lwd", "oer", "rht", "rmg", \
             "sht", "sig", "sl1", "sl2", "smt", "swi", "tht", "wba", "wbc", \
             "wbd", "wbl", "wbr", "wbs", "wbt", "wht", "blw",               \
             "ARC", "ARD", "ARL", "ARS", "BEG", "BET", "CHT", "CMG", "CS1", \
             "CS2", "DTS", "DWD", "END", "LIN", "LWD", "OER", "RHT", "RMG", \
             "SHT", "SIG", "SL1", "SL2", "SMT", "SWI", "THT", "WBA", "WBC", \
             "WBD", "WBL", "WBR", "WBS", "WBT", "WHT", "BLW"                \
           ]

  cparms = [ "erf", "fro", "ERF", "FRO" ]

  if (not isinstance(pname,types.StringType)):
    print "wmgetp: Parameter '" + str(pname) + "' is not a string type." 
    return None
  if (iparms.count(pname) > 0):
    return c_wmgetip(pname)
  elif (rparms.count(pname) > 0):
    return c_wmgetrp(pname)
  elif (cparms.count(pname) > 0):
    return c_wmgetcp(pname)
  else:
    print \
      "wmgetp: specified value for " + pname + " is not of a recognized type." 
  return None

def wmsetp(pname,val):
  if (not isinstance(pname,types.StringType)):
    print "wmsetp: Parameter '" + str(pname) + "' is not a string type." 
    return None
  if (isinstance(val,types.FloatType)):
    c_wmsetrp(pname,val)
  elif (isinstance(val,types.IntType)): 
    c_wmsetip(pname,val)
  elif (isinstance(val,types.StringType)):
    c_wmsetcp(pname,val)
  else:
    print \
      "wmsetp: specified value for " + pname + " is not of a recognized type." 
  return None

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

def yiqrgb(r,g,b):
  return(c_yiqrgb(r,g,b))
