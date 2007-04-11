#
#   File:    
#     cns01p.py
#
#   Synopsis:
#     Draws contours using dashed lines and user-specified labels.
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
#     o  Drawing a contour visualization with user-specified labels
#     o  Retrieving string arrays.
#
#  Output
#     Two visualizations are produced:
#       1.)  A simple contour plot with dashed contour lines.
#       2.)  Same as 1.), but with user-specified line labels.
#
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
#  Open a workstation and draw a contour plot.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"cns01p")

res = Ngl.Resources()
res.cnMonoLineDashPattern = False

plot1 = Ngl.contour(wks,T,res)

# 
#  Retrieve the automatically set line labels.  
#  These will be: ['-80', '-60', '-40', '-20', '0', '20', '40', '60', '80']
#
line_labels1 = Ngl.get_string_array(plot1,"cnLineLabelStrings")

#
#  Set explicit line labels.  Notice that the dash line
#  setting carries over from the first plot.
#
res.cnExplicitLineLabelsOn = True
res.cnLineLabelStrings = ["Lab1",  "Lab2", "Lab3", "Lab4", "Lab5",
                          "Lab6",  "Lab7", "Lab8", "Lab9"]
plot2 = Ngl.contour(wks,T,res)

#
#  Retrieve the explicitly set line labels.
#  These will be: ['Lab1', 'Lab2', 'Lab3', 'Lab4', 'Lab5', 
#                  'Lab6', 'Lab7', 'Lab8', 'Lab9']
#
line_labels2 = Ngl.get_string_array(plot2,"cnLineLabelStrings")

Ngl.end()
