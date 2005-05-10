import Ngl
#
#  Generate gaussian latitudes and weights for a specified 
#  number of latitudes:
#

#
#  Specify the number of latitudes for the entire globe.
#
nlat = 64

#
#  Divide the number of latitudes for the globe by two to get the
#  number of latitudes per hemisphere.
#
gau_info = Ngl.gaus(nlat/2)

#
#  Extract the latitudes and weights from the returned array.
#
glat     = gau_info[:,0]
gwgt     = gau_info[:,1]

Ngl.end()
