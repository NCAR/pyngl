#
#   File:    
#     cn01p.py
#
#   Synopsis:
#     Draws a simple contour using all defaults.
#
#   Category:
#     Contouring
#
#   Author:  
#     Fred Clare (based on an example of Dave Brown)
#
#   Date of initial publication:    
#     November, 2004
#
#   Description:     
#     Given a simple mathematically generated data set,
#     demonstrates Ngl.contour with all resources set
#     to their defaults.
#
#  Effects illustrated:
#     Drawing a contour visualization using all defaults.
# 
#  Output:
#     A single visualization is produced.
#
#
import Ngl
import numpy

M=29
N=25
T = numpy.zeros([N,M])
 
#
# create a mound as a test data set
#
jspn = numpy.power(xrange(-M/2+5,M/2+5),2)
ispn = numpy.power(xrange(-N/2-3,N/2-3),2)
for i in xrange(len(ispn)):
  T[i,:] = ispn[i] + jspn
T = 100. - 8.*numpy.sqrt(T)

#
#  Open a workstation and draw the contour plot.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"cn01p")
Ngl.contour(wks,T)

Ngl.end()
