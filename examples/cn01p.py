#
#   File:    cex01.py
#
#   Author:  Fred Clare (based on an example of Dave Brown)
#            National Center for Atmospheric Research
#            PO 3000, Boulder, Colorado
#
#   Date:    Fri Nov 19 13:00:45 MST 2004
#
#   Description:     
#            Given a simple mathematically generated data set,
#            demonstrates Ngl.contour with all resources set
#            to their defaults.
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
wks = Ngl.open_wks(wks_type,"cn01p")
Ngl.contour(wks,T)

Ngl.end()
