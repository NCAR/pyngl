import sys

def test_modules(title):
  module_list = sys.modules.keys()

  imported_Numeric  = "Numeric" in module_list
  imported_numpy    = "numpy" in module_list
  imported_numpy_ma = "numpy.core.ma" in module_list
  imported_MA       = "MA" in module_list

  print "\n" + title
  print "imported_Numeric is '" + str(imported_Numeric) + "'"
  print "imported_numpy is '" + str(imported_numpy) + "'"
  print "imported_MA is '" + str(imported_MA) + "'"
  print "imported_numpy_ma is '" + str(imported_numpy_ma) + "'"


test_modules("imported nothing yet")

import Numeric
test_modules("import Numeric only")

sys.modules.pop("Numeric")
import numpy
test_modules("import numpy only")

import MA
test_modules("import numpy,import MA")

sys.modules.pop("numpy")
test_modules("import MA only")
