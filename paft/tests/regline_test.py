# To test with the Numeric module, be sure to uncomment these three
# lines, and comment the NumPy lines after this.
#
# import PAF_numeric.NglA as NglA
# import MA
# default_type = "numeric"

# To test with the NumPy module, be sure to uncomment these two lines,
# and comment the three Numeric lines above.
import NglA
default_type = "numpy"

#
# Leave these both alone, regardless of what module you are testing.
#
import numpy
import Numeric
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


tval  = 0.
nptxy = 0
rcl,fv,attrs = NglA.regline(x,y)

#
# Correct values for first set of tests.
#
rcl_value  = 0.9745614528656006
tval_value = 38.7428595367

test_value("regline",rcl,rcl_value,delta=1e-7)
test_value("regline",attrs["tval"],tval_value)

