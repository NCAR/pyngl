#
#  File:
#    gaus.py
#
#  Synopsis:
#    Illustrates usage of the gaus function.
#
#  Category:
#    Processing
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    June, 2005
#
#  Description:
#    Calls the gaus function to generate Gaussian
#    latitutes and weights for a specified number
#    of latitudes.
#
#  Effects illustrated:
#    Calling the gaus function.
# 
#  Output:
#    The calculated latitudes and weights are printed to
#    standard output.  No visualization is produced.
#     

import Ngl

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

print glat,gwgt

Ngl.end()
