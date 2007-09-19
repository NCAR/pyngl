import os,re
import fileinput
import tempfile

#
# Modify the example sources appropriately if Numeric support is
# requested.  For all examples except for a few such as "meteogram.py,"
# "scatter1.py," and "ngl09p.py" this is just a matter of replacing
# "import numpy" with "import Numeric as numpy".  The cases of
# "meteogram.py" and "ngl09p.py" are handled as special cases in the if
# block below; "scatter1.py" is then handled separately.
#
def modify_pynglex_files(files_to_modify):

  print '====> Modifying some of the pynglex examples to work with NumPy.'

  for line in fileinput.input(files_to_modify,inplace=1):
    if (re.search("^import numpy.core.ma as ma",line) != None):
      print "import MA as ma"
    elif (re.search("^import numpy",line) != None):
      print "import Numeric as numpy"
    elif(re.search("^import Ngl",line) != None):
      print "import PyNGL_numeric.Ngl as Ngl"
    elif(re.search("^import Nio",line) != None):
      print "import PyNGL_numeric.Nio as Nio"
    elif(re.search("from numpy import ma as MA",line) != None):
      print "import MA"
    elif (os.path.basename(fileinput.filename()) == "meteogram.py" and  \
      re.search("dtype.char",line) != None):
      print line.replace("dtype.char","typecode()"),
    elif (os.path.basename(fileinput.filename()) == "meteogram.py" and  \
      re.search("zero\[0\]",line) != None):
      print line.replace("zero[0]","zero"),
    elif (os.path.basename(fileinput.filename()) == "ngl09p.py" and     \
      re.search("dtype=float",line) != None):
      print line.replace("dtype=float","MA.Float0"),
    elif (os.path.basename(fileinput.filename()) == "panel2.py" and  \
      re.search("min\(",line) != None):
      print line.replace(")",")[0]"),
    elif (os.path.basename(fileinput.filename()) == "panel2.py" and  \
      re.search("max\(",line) != None):
      print line.replace(")",")[0]"),
    else:
      print line,
  for file in files_to_modify:
    if (os.path.basename(file) == "scatter1.py"):
      scatter_src = open(file,"r")
      scatter_new = tempfile.TemporaryFile()

      while(1):
        line = scatter_src.readline()
        if (line == ""):
          break
        elif (re.search("From Scientific import",line) != None):
          while (re.search("^from",line) == None):
            line = scatter_src.readline()
          line = scatter_src.readline()
        elif (re.search("Do a quadratic",line) != None):
          while (re.search("^plot =",line) == None):
            line = scatter_src.readline()
          line = scatter_src.readline()
          scatter_new.write("""#
#  From Scientific import the the polynomial least squares function.
#
# You can download ScientificPython from:
#
#  http://starship.python.net/~hinsen/ScientificPython/
#
from Scientific.Functions.LeastSquares import polynomialLeastSquaresFit

#
#  Put the data in the correct format for the least squares 
#  function and do the fit.
#
data = []
for j in xrange(len(x)):
  data.append((x[j],y[j]))
params = [0.,0.,1.e-7]
a = polynomialLeastSquaresFit(params, data)

#
#  Draw the least squares quadratic curve.
#
num  = 301
delx = 1000./num
u    = numpy.zeros(num,'f')
v    = numpy.zeros(num,'f')
for i in xrange(num):
  u[i] = float(i)*delx
  v[i] = a[0][0]+a[0][1]*u[i]+a[0][2]*u[i]*u[i]
plot = Ngl.xy(wks,u,v,resources) # Draw least squares quadratic.
""")
        scatter_new.write(line)

#
#  Write the new Numeric source back over the NumPy source.
#
      scatter_src.close()
      scatter_src = open(file,"w+")
      scatter_new.seek(0)
      for line in scatter_new.readlines():
        scatter_src.write(line)
      scatter_src.close()
      scatter_new.close()


