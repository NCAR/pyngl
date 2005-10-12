#
#   File:    
#     cns01p.py
#
#   Synopsis:
#     Draws contours using dashed lines for contour levels.
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
#     demonstrates Ngl.contour with dashed contour-line levels.
#
#  Effects illustrated:
#     o  Drawing a contour visualization with dashed contour lines
#        produced by setting the resourcer "cnMonoLineDashPattern" 
#        to False.
#
#  Output
#     A single visualization is produced.
# 
#
import Ngl
import Numeric

M=29
N=25
T = Numeric.zeros([N,M])
 
#
# create a mound as a test data set
#
jspn = Numeric.power(xrange(-M/2+5,M/2+5),2)
ispn = Numeric.power(xrange(-N/2-3,N/2-3),2)
for i in xrange(len(ispn)):
  T[i,:] = ispn[i] + jspn
T = 100. - 8.*Numeric.sqrt(T)

#
#  Open a workstation and draw the contour plot.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"cns01p")

res = Ngl.Resources()
res.cnMonoLineDashPattern = False

Ngl.contour(wks,T,res)

Ngl.end()
