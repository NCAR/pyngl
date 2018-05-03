import numpy,Ngl,sys

print "Testing Ngl.nice_cntr_levels code....."
print "--------------------------------------------------"
print "Testing CINT = 0.0, outside = False / returnLevels = False"
print "Ngl.nice_cntr_levels(-23.1, 70.7, outside=False)"
answer    = (-20.0, 70.0, 9.0)
answer_is = Ngl.nice_cntr_levels(-23.1, 70.7, outside=False)
if (answer_is == answer):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is
  print "answer",answer

print "--------------------------------------------------"
print "Testing CINT = 0.0, outside = True / returnLevels = False"
print "Ngl.nice_cntr_levels(-23.1, 70.7, outside=True)"
answer    = (-30.0, 80.0, 11.0)
answer_is = Ngl.nice_cntr_levels(-23.1, 70.7, outside=True)
if (answer_is == answer):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is
  print "answer",answer

print "--------------------------------------------------"
print "Testing cint = 0.0, outside = False / returnLevels = True"
print "Ngl.nice_cntr_levels(-23.1, 70.7, outside=False, returnLevels=True)"
answer1    = (-20.0, 70.0, 9.0)
answer2    = numpy.array([-20., -11.,  -2.,   7.,  16.,  25.,  34.,  43.,  52.,  61.,  70.])
answer_is = Ngl.nice_cntr_levels(-23.1, 70.7, outside=False, returnLevels=True)
if (answer_is[0:3] == answer1):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[0:3]
  print "answer",answer1

if (numpy.all([answer_is[3],answer2])):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[3]
  print "answer",answer2

print "--------------------------------------------------"
print "Testing CINT = 0.0, outside = True / returnLevels = True"
print "Ngl.nice_cntr_levels(-23.1, 70.7, outside=True, returnLevels=True)"
answer1    = (-30.0, 80.0, 11.0)
answer2    = numpy.array([-30., -19.,  -8.,   3.,  14.,  25.,  36.,  47.,  58.,  69.,  80.])
answer_is = Ngl.nice_cntr_levels(-23.1, 70.7, outside=True, returnLevels=True)
if (answer_is[0:3] == answer1):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[0:3]
  print "answer",answer1

if (numpy.all([answer_is[3],answer2])):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[3]
  print "answer",answer2

print "--------------------------------------------------"
print "Testing CINT = 5.0, outside = True / returnLevels = True"
print "Ngl.nice_cntr_levels(-11.1, 26.7, outside=True, cint = 5.0)"
answer1    = (-15.0, 30.0, 5.0)
answer2    = numpy.array([-15., -10., -5., 0., 5.,10., 15., 20., 25., 30.])
answer_is = Ngl.nice_cntr_levels(-11.1, 26.7, outside=True, cint = 5.0, returnLevels=True)

if (answer_is[0:3] == answer1):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[0:3]
  print "answer",answer1

# I don't know why this next test doesn't work. Instead, I have
# to subtract the two arrays and compare to 0.0. Makes no sense.
#if (numpy.all([answer_is[3],answer2])):
if (numpy.all((answer_is[3] - answer2) == 0)):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[3]
  print "answer",answer2

print "--------------------------------------------------"
print "Testing CINT = 3.0, outside = True / returnLevels = True"
print "Ngl.nice_cntr_levels(-2.1, 7.1, outside=True, cint = 3.0, returnLevels=True)"
answer1    = (-3.0, 9.0, 3.0)
answer2    = numpy.array([-3.,  0.,  3.,  6.,  9.])
answer_is = Ngl.nice_cntr_levels(-2.1, 7.1, outside=True, cint = 3.0, returnLevels=True)

if (answer_is[0:3] == answer1):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[0:3]
  print "answer",answer1

#if (numpy.all([answer_is[3],answer2])):
if (numpy.all((answer_is[3] - answer2) == 0)):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[3]
  print "answer",answer2


print "--------------------------------------------------"
print "Testing CINT = 3.0, outside = False / returnLevels = True"
print "Ngl.nice_cntr_levels(-2.1, 7.1, outside=False, cint = 3.0, returnLevels=True)"
answer1    = (0.0, 6.0, 3.0)
answer2    = numpy.array([0.,  3.,  6.])
answer_is = Ngl.nice_cntr_levels(-2.1, 7.1, outside=False, cint = 3.0, returnLevels=True)

if (answer_is[0:3] == answer1):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[0:3]
  print "answer",answer1

#if (numpy.all([answer_is[3],answer2])):
if (numpy.all((answer_is[3] - answer2) == 0)):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[3]
  print "answer",answer2


print "--------------------------------------------------"
print "Testing CINT = 3.0, outside = True / returnLevels = True, aboutZero = True"
print "Gl.nice_cntr_levels(-2.1,7.1,outside=True,cint=3.0,returnLevels=True,aboutZero=True)"
answer1    = (-9.0, 9.0, 3.0)
answer2    = numpy.array([-9., -6., -3.,  0.,  3.,  6.,  9.])
answer_is = Ngl.nice_cntr_levels(-2.1, 7.1, outside=True, cint = 3.0, returnLevels=True, aboutZero=True)
if (answer_is[0:3] == answer1):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[0:3]
  print "answer",answer1

#if (numpy.all([answer_is[3],answer2])):
if (numpy.all((answer_is[3] - answer2) == 0)):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[3]
  print "answer",answer2

print "--------------------------------------------------"
print "Testing CINT = 3.0, outside = False / returnLevels = True, aboutZero = True"
print "Ngl.nice_cntr_levels(-2.1,7.1,outside=False,cint=3.0,returnLevels=True,aboutZero=True)"
answer1    = (-6.0, 6.0, 3.0)
answer2    = numpy.array([-6., -3.,  0.,  3.,  6.])
answer_is = Ngl.nice_cntr_levels(-2.1, 7.1, outside=False, cint = 3.0, returnLevels=True, aboutZero=True)
if (answer_is[0:3] == answer1):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[0:3]
  print "answer",answer1

#if (numpy.all([answer_is[3],answer2])):
if (numpy.all((answer_is[3] - answer2) == 0)):
  print "...SUCCESSFUL..."
else:
  print "...UNSUCCESSFUL..."
  print "answer_is",answer_is[3]
  print "answer",answer2

