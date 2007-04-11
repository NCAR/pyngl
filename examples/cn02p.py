#
#   File:    
#     cn02p.py
#
#   Synopsis:
#     Draws a simple color filled contour.
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
#     demonstrates Ngl.contour with color filled contour
#     levels.
#
#  Effects illustrated:
#     o  Drawing a contour visualization with color filled
#        contour levels.
#
#  Output
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
#  Open a workstation and draw the contour plot with color fill.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"cn02p")

res = Ngl.Resources()
res.cnFillOn = True
Ngl.contour(wks,T,res)

Ngl.end()
