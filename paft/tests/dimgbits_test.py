import numpy,Ngl,sys
from utils import *

#
# 1440052 = "0101011111100100110100"
#
#  skip  keep skip keep skip keep skip keep
#  01010  11   111  10   010  01  101   00
#         3         2         1         0
#
b     = 1440052
ibit  = 5
nbits = 2
nskip = 3
iter  = 4 
jout  = numpy.array([0,1,3,2],'i')             # Correct values
j     = Ngl.dim_gbits(b,ibit,nbits,nskip,iter)

test_values("dim_gbits (int input)",j,jout)

# 
# 1466224640 = "01010111011001001101000000000000"
#
#  skip  keep skip keep skip keep skip keep
#  01010  11   101  10   010  01  101   00
#         3         2         1         0
#
b    = 1466224640
jout = numpy.array([3,2,1,0],'i')   # Correct values
j    = Ngl.dim_gbits(b,ibit,nbits,nskip,iter)

test_values("dim_gbits (int input)",j,jout)

#
# Test for ushort
#
# 45673 = "1011001001101001"
#
#  skip  keep skip keep skip keep
#  10     11   001  00   110  10
#
b     = numpy.array(45673,'H')
ibit  = 2
nbits = 2
nskip = 3
iter  = 3
jout  = [3,0,2]                                 # Correct values
j     = Ngl.dim_gbits(b,ibit,nbits,nskip,iter)
test_values("dim_gbits (ushort input)",j,jout)

#
# Similar test as above, but with three of the same values input
#
iter  = 6
b     = numpy.array([45673,45673,45673],'H')
jout  = [3,0,2,1,2,1]                                 # Correct values
j     = Ngl.dim_gbits(b,ibit,nbits,nskip,iter)
test_values("dim_gbits (ushort multid input)",j,jout)

#
#  "01010111011001001101000000000000"
#
#  skip  keep skip keep skip keep skip keep
#  01010  11   101  10   010  01  101   00
#
b     = numpy.array(12905,'H')
ibit  = 2
nbits = 2
nskip = 3
iter  = 3
jout  = [3,0,2]                                 # Correct values
j     = Ngl.dim_gbits(b,ibit,nbits,nskip,iter)
test_values("dim_gbits (ushort input)",j,jout)

#
#  Test for ubyte
#
#  "11110100"
#
#  skip  keep skip keep skip keep
#   1     11   1    01   0    0(0)
#
b     = numpy.array(244,'B')
ibit  = 1
nbits = 2
nskip = 1
iter  = 3
jout  = [3,1,0]                                 # Correct values
j     = Ngl.dim_gbits(b,ibit,nbits,nskip,iter)
test_values("dim_gbits (ubyte input)",j,jout)

#
#  Test for byte
#
#  "01110100"
#
#  skip  keep skip keep skip keep
#   0     11   1    01   0    0(0)
#
b     = numpy.array(116,'b')
ibit  = 1
nbits = 2
nskip = 1
iter  = 3
jout  = [3,1,0]                                 # Correct values
j     = Ngl.dim_gbits(b,ibit,nbits,nskip,iter)
test_values("dim_gbits (byte input)",j,jout)
