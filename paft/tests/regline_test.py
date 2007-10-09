import Ngl,types
default_type = "numpy"

#
# Leave these both alone, regardless of what module you are testing.
#
import numpy
from utils import *

#
# Begin regline tests.
#

x = numpy.array([1190.,1455.,1550.,1730.,1745.,1770., \
                 1900.,1920.,1960.,2295.,2335.,2490., \
                 2720.,2710.,2530.,2900.,2760.,3010.])

y = numpy.array([1115.,1425.,1515.,1795.,1715.,1710., \
                 1830.,1920.,1970.,2300.,2280.,2520., \
                 2630.,2740.,2390.,2800.,2630.,2970.])


rc = Ngl.regline(x,y)
check_type(rc,types.ListType)
rcl,attrs = rc    # Separate into value and dictionary

check_type(rcl,numpy.core.ma.MaskedArray)
check_type(attrs,types.DictType)

#
# Correct values for first set of tests.
#
rcl_value  = 0.9745614528656006
xave_value = 2165
yave_value = 2125.27777778
tval_value = 38.7428595367
rstd_value = 0.0251546076193
nptxy_value = 18
yint_value = 15.352282489361187

test_value("regline",rcl,rcl_value,delta=1e-7)
test_value("regline (fill_value)",rcl.fill_value(),1e20)
test_value("regline tval",attrs["tval"],tval_value)
test_value("regline xave",attrs["xave"],xave_value)
test_value("regline yave",attrs["yave"],yave_value)
test_value("regline rstd",attrs["rstd"],rstd_value)
test_value("regline yintercept",attrs["yintercept"],yint_value)
test_value("regline nptxy",attrs["nptxy"],nptxy_value)
del rcl

rcl = Ngl.regline(x,y,return_info=False)
check_type(rcl,numpy.core.ma.MaskedArray)
test_value("regline (ma)",rcl,rcl_value,delta=1e-7)
test_value("regline (fill_value)",rcl.fill_value(),1e20)

