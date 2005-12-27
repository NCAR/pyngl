#
#  File:
#    legend.py
#
#  Synopsis:
#    Demonstrates legends and legend resource settings.
#
#  Category:
#    Legend
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    March, 2005
#
#  Description:
#    This example illustrates the effects of setting values
#    for various legend resources.
#
#  Effects illustrated:
#    Legend fonts, sizes, and orientation.
# 
#  Output:
#    Four labelbar visualizations are produced showing:
#      1.) default settings
#      2.) changing the label font
#      3.) changing the size and orientation
#      4.) using lots of user-specified labels
#
#  Notes:
      
#
#  Import Ngl support functions.
#
import Ngl

wkres            = Ngl.Resources()
wkres.wkColorMap = "default"
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"legend",wkres)

labels = ["One","Two","Three","Four","Five","Six"]

#
# Generate a legend with the default settings.
#
lg = Ngl.legend_ndc(wks,5,labels,0.3,0.9)
Ngl.frame(wks)

#
# Change the font and line thickness of the labels.
#
rlist                  = Ngl.Resources()
rlist.lgLineThicknessF = 4.0
rlist.lgLabelFont      = "Times-bold"

lg = Ngl.legend_ndc(wks,5,labels,0.3,0.9,rlist)
Ngl.frame(wks)

#
# Change the orientation and size.
#
del rlist.lgLabelFont
rlist.vpWidthF          = 0.85
rlist.vpHeightF         = 0.20
rlist.lgOrientation     = "Horizontal"
lg = Ngl.legend_ndc(wks,6,labels,0.1,0.2,rlist)
Ngl.frame(wks)

#
# Generate a lot of labels. Notice how the legend labels are automatically
# adjusted and not every one is shown. To turn this off, set
# lgAutoStride to False, but then your labels will run into each other.
# You could set lgLabelAngleF to -45 or 90 to get slanted labels.
#

lotta_labels = ["AL","AR","AZ","CA","CO","CT","DE","FL","GA","IA","ID","IL",\
                "IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT",\
                "NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA",\
                "RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"]

rlist.lgLabelAlignment   = "AboveItems"
rlist.lgLabelFontHeightF = 0.014
rlist.lgMonoDashIndex    = True
lg = Ngl.legend_ndc(wks,len(lotta_labels),lotta_labels,0.1,0.5,rlist)
Ngl.frame(wks)


Ngl.end()
