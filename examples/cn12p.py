#
#   File:
#     cn12p.py
#
#   Synopsis:
#     Draws a color filled contour map over Africa.
#
#   Category:
#     Contours over maps
#
#   Author:
#     Fred Clare (based on examples of Mary Haley and Dave Kennison).
#
#   Date of initial publication:    
#     September, 2004
#  
#   Description:
#     This example draws a map with filled contours appearing
#     only over Africa.  In order to mask Africa from the map fill,
#     the mpMaskAreaSpecifiers resource is used to mask all of
#     the countries in Africa.
#
#  Effects illustrated
#     o  Reading data from an ASCII file.
#     o  Drawing color filled contours over specified geographic areas.
#     o  Using a ".res" file to set many resources, for example to:
#          + set fill colors
#          + turn off high labels, info label, line labels, low labels.
#          + turn on area masking.
#          + set the map projection to "orthographic".
#          + set several label bar resources.
#
#  Output:
#     A single visualization is produced.
#
#  Notes: 
#     This example requires the resource file cn12p.res.

#
#  Import NumPy.
#
import Numeric,sys

#
#  Import Ngl support functions.
#
import Ngl

N = 40
len_dims = [N,N]
#
#  These are the geographical areas we want to fill.
#
fill_specs = ["water","land"]
#
#  These are the geographical areas we want to mask.
#
mask_specs =                                                                 \
 ["algeria","angola","angola-exclave-called-cabinda","benin","botswana",     \
  "burundi","cameroon","central-african-republic","chad","congo","djibouti", \
  "egypt","equatorial-guinea","ethiopia","gabon","gambia","ghana","guinea",  \
  "guinea-bissau","ivory-coast","kenya","lesotho","liberia","libya",         \
  "madagascar","malawi","mali","mauritania","mauritius","morocco",           \
  "mozambique","namibia","niger","nigeria","rwanda","senegal","sierra-leone",\
  "somalia","south-africa","sudan","swaziland","tanzania","togo","tunisia",  \
  "uganda","upper-volta","western-sahara","zaire","zambia","zimbabwe"]

#
#  Open a workstation.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"cn12p")

dirc = Ngl.pynglpath("data")
z    = Ngl.asciiread(dirc+"/asc/cn12n.asc",len_dims,"float")
 
resources = Ngl.Resources()
resources.sfXCStartV  = -18.0
resources.sfXCEndV    =  52.0
resources.sfYCStartV  = -35.0
resources.sfYCEndV    =  38.0
resources.vpXF        =   0.1
resources.mpMaskAreaSpecifiers  =  mask_specs
resources.mpFillAreaSpecifiers  =  fill_specs
resources.pmLabelBarDisplayMode =  "always"
Ngl.contour_map(wks,z[:,:],resources)

del resources
Ngl.end()
