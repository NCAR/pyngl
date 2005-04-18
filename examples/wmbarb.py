import Ngl

#
#  Draw four wind barbs of the same magnitude, but at different 
#  locations and in different directions.

#
#  Open a workstation.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type, "wmbarb")

#
#  Draw wind barbs.
#
x = [0.25, 0.75, 0.75, 0.25]  # x,y,u,v can also be Numeric arrays.
y = [0.25, 0.25, 0.75, 0.75]
u = [50., -50., -50.,  50.0]
v = [50.,  50., -50., -50.0]
Ngl.wmsetp("wbs", 0.2)        # Scale the size.
Ngl.wmbarb(wks, x, y, u, v)   # Draw barbs.
Ngl.frame(wks)                # Draw plot.

#
#  Retrieve the value of the wbs parameter.
#
size = Ngl.wmgetp("wbs")
print "Current scale factor for wind barb size = %10.7f" % (size)

Ngl.end()
