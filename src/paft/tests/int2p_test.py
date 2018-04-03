import Ngl,types
from utils import *

# Test with 1D pi, xi, po lists
pi = [1000.,925.,850.,700.,600.,500.,400.,300.,250.,\
      200.,150.,100.,70.,50.,30.,20.,10.]
po = [1000.,950.,900.,850.,800.,750.,700.,600.,500., \
      425.,400.,300.,250.,200.,100.,85.,70.,50.,40.,\
      30.,25.,20.,15.,10. ]
xi    = pi

linlog = 1
xo = Ngl.int2p(pi,xi,po,linlog)

xo_value =  [1000.,950.,900.,850.,800.,750.,700.,600.,\
             500.,425.,400.,300.,250.,200.,100.,85.,70.,\
             50.,40.,30.,25.,20.,15.,10.]
test_values("int2p",xo,xo_value)

# Test with numpy arrays
pi = numpy.array([1000.,925.,850.,700.,600.,500.,400.,300.,250.,\
      200.,150.,100.,70.,50.,30.,20.,10.])
po = numpy.array([1000.,950.,900.,850.,800.,750.,700.,600.,500., \
      425.,400.,300.,250.,200.,100.,85.,70.,50.,40.,\
      30.,25.,20.,15.,10. ])
xi    = pi

xo = Ngl.int2p(pi,xi,po,linlog)
test_values("int2p",xo,xo_value)

# Test with multi-d xim array
xim = multid(xi,[3,2])
xom_value = multid(xo,[3,2])

# Test with different value for linlog
linlog = 2
xo = Ngl.int2p(pi,xi,po,linlog)

xo_value = [1000.,950.65519477,900.69788859,850.,\
            803.16295867,753.30213825,700.,600.,500.,\
            427.16843998,400.,300.,250.,200.,100.,\
            86.33050073,70.,50.,41.26341589,30.,\
            25.50339713,20.,15.84962501,10.]
xom_value = multid(xo,[3,2])

xo  = Ngl.int2p(pi,xi,po,linlog)
xom = Ngl.int2p(pi,xim,po,linlog)
test_values("int2p",xo,xo_value)
test_values("int2p",xom,xom_value)

# Test with multi-d xim and pim array
pim = multid(pi,[3,2])
xom = Ngl.int2p (pim,xim,po,linlog)
test_values("int2p",xom,xom_value)

# Test with missing values
xmsg   = -999
xi[1]  = xmsg
xi[2]  = xmsg
xi[3]  = xmsg
xi[4]  = xmsg
xi[16] = xmsg

linlog = 1
xi_msg = numpy.ma.masked_values(xi,xmsg)  # convert to masked array
xo = Ngl.int2p (pi,xi_msg,po,linlog)

xo_values = numpy.ma.masked_values([1000.,950.,900.,850.,800.,750.,700.,600.,500.,425.,400.,300.,250.,200.,100.,85.,70.,50.,40.,30.,25.,20.,-999.,-999.],xmsg)

test_values("int2p",xom,xom_value)

