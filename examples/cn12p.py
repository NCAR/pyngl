#
#  Python version of the NCL example cn12n.
#
#  Import NumPy.
#
import Numeric,sys

#
#  Import Ngl support functions.
#
from Ngl import *

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
wks = ngl_open_wks(wks_type,"cn12p")

dirc = ncargpath("data")
z    = ngl_asciiread(dirc+"/asc/cn12n.asc",len_dims,"float")
 
resources = Resources()
resources.sfXCStartV  = -18.0
resources.sfXCEndV    =  52.0
resources.sfYCStartV  = -35.0
resources.sfYCEndV    =  38.0
resources.vpXF        =   0.1
resources.mpMaskAreaSpecifiers  =  mask_specs
resources.mpFillAreaSpecifiers  =  fill_specs
resources.pmLabelBarDisplayMode =  "always"
ngl_contour_map(wks,z[:,:],resources)

del resources
ngl_end()
