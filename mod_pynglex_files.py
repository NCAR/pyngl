import os,re
import fileinput
import tempfile

#
# Modify the example sources appropriately if numpy support is
# requested.  For all examples except "metrogram.py," "scatter1.py," 
# and "ngl09p.py" this is just a matter of replacing "import Numeric"
# with "import numpy as Numeric".  The cases of "meteogram.py"
# and "ngl09p.py" are handled as special cases in the if block below; 
# "scatter1.py" is then handled separately.
#
def modify_pynglex_files(files_to_modify):

  print '====> Modifying some of the pynglex examples to work with numpy.'

  for line in fileinput.input(files_to_modify,inplace=1):
    if (re.search("import Numeric",line) != None):
      print "import numpy as Numeric"
    elif(re.search("^import Ngl",line) != None):
      print "import PyNGL_numpy.Ngl as Ngl"
    elif(re.search("^import Nio",line) != None):
      print "import PyNGL_numpy.Nio as Nio"
    elif(re.search("Numeric.Float0",line) != None):
      print line.replace("Numeric.Float0","Numeric.float"),
    elif(re.search("Numeric.Int0",line) != None):
      print line.replace("Numeric.Int0","Numeric.int"),
    elif(re.search("Numeric.Float",line) != None):
      print line.replace("Numeric.Float","Numeric.float"),
    elif(re.search("Numeric.Int",line) != None):
      print line.replace("Numeric.Int","Numeric.int"),
    elif (os.path.basename(fileinput.filename()) == "meteogram.py" and  \
      re.search("typecode()",line) != None):
      print line.replace("typecode()","dtype.char"),
    elif (os.path.basename(fileinput.filename()) == "meteogram.py" and  \
      re.search("(ind_above_zero)",line) != None):
      print line.replace("(ind_above_zero)","(ind_above_zero[0])"),
    elif (os.path.basename(fileinput.filename()) == "ngl09p.py" and     \
        re.search("import MA",line) != None):
      print line.replace("import MA","import numpy.core.ma as MA"),
    elif (os.path.basename(fileinput.filename()) == "ngl09p.py" and     \
        re.search("MA.Float0",line) != None):
      print line.replace("MA.Float0","dtype=float"),
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
        elif (re.search("Put the data",line) != None):
          while (re.search("^plot =",line) == None):
            line = scatter_src.readline()
          line = scatter_src.readline()
          scatter_new.write("""#
#  Do a quadratic least squares fit.
#
npoints = len(x)
a = Numeric.zeros([npoints,3],Numeric.float)
for m in xrange(npoints):
  a[m,0] = 1.
  for j in xrange(1,3):
    a[m,j] = x[m]*a[m,j-1]
c = (Numeric.linalg.lstsq(a,y,rcond=1.e-15))[0]

#
#  Draw the least squares quadratic curve.
#
num  = 301
delx = 1000./num
xp    = Numeric.zeros(num,Numeric.float)
yp    = Numeric.zeros(num,Numeric.float)
for i in xrange(num):
  xp[i] = float(i)*delx
  yp[i] = c[0]+c[1]*xp[i]+c[2]*xp[i]*xp[i]
plot = Ngl.xy(wks,xp,yp,resources) # Draw least squares quadratic.

""")
        scatter_new.write(line)

#
#  Write the new NumPy source back over the Numeric source.
#
      scatter_src.close()
      scatter_src = open(file,"w+")
      scatter_new.seek(0)
      for line in scatter_new.readlines():
        scatter_src.write(line)
      scatter_src.close()
      scatter_new.close()


