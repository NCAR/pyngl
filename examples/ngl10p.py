#
#  Import NumPy.
#
import Numeric

#
#  Import PyNGL support functions.
#
import Ngl

#
#  The function "wigley" performs computations on the input array
#  based on where the values are less than 1953, are between
#  1953 and 1973 inclusive, or greater than 1973.
#  
def wigley(time):
  y = Numeric.zeros(time.shape,Numeric.Float0)
  Numeric.putmask(y,Numeric.less(time,1953.),         \
                ((time-1860.)/(1953.-1860.)) * 35.0)
  Numeric.putmask(y,Numeric.logical_and(Numeric.greater_equal(time,1953.),  \
                                        Numeric.less_equal(time,1973.)),    \
                ((time-1953.)/(1973.-1953.)) * (68. - 35.) + 35.)
  Numeric.putmask(y,Numeric.logical_and(Numeric.greater(time,1973.),        \
                                        Numeric.less_equal(time,1990.)),    \
                ((time-1973.)/(1990.-1973.)) * (75. - 68.) + 68.)
  return y

#
#  Main program.
#
time1 = Numeric.array([1990,  1985,  1980,  1970,  1960,  1950,  1940,  1930, \
                       1920,  1910,  1900,  1890,  1880,  1870,  1860],
                       Numeric.Int)
y1    = Numeric.array([68.065, 65.00, 70.67, 63.06, 43.42, 28.28, 23.00,  \
                       20.250, 17.77, 15.36, 10.01,  6.40,  3.98,  2.18,  \
                        1.540], Numeric.Float0)

time2 = Numeric.arange(min(time1),max(time1)+1,1)

y2 = wigley(time2)      # Calculate proposed values as a function of time.

maxdim = max(y1.shape[0],y2.shape[0])
y    = -999.*Numeric.ones((2,maxdim),Numeric.Float0)  # Create 2D arrays to
time = -999.*Numeric.ones((2,maxdim),Numeric.Float0)  # hold 1D arrays above.

y[0,0:y1.shape[0]] = y1
y[1,0:y2.shape[0]] = y2

time[0,0:time1.shape[0]] = time1.astype(Numeric.Float0)
time[1,0:time2.shape[0]] = time2.astype(Numeric.Float0)

#
#  Define a color map and open a workstation.
#
cmap = Numeric.zeros((2,3),Numeric.Float0)
cmap[0] = [1.,1.,1.]
cmap[1] = [0.,0.,0.]
rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl10p",rlist)

resources = Ngl.Resources()

resources.caXMissingV = -999.
resources.caYMissingV = -999.

resources.vpWidthF     = 0.8
resources.vpXF         = 0.13

resources.tiMainString  = ":F22:Sulfur Emissions" # ":F22:" changes
resources.tiXAxisString = ":F22:Year"             # the font to "22"
resources.tiYAxisString = ":F22:Tg s/yr"          # which is helvetica
                                                    # bold.
resources.tmXBLabelFont = 21
resources.tmYLLabelFont = 21

resources.trXMinF              = 1855 # Set minimum X axes value.

resources.xyDashPatterns       = [16,0]   # ( dash, solid )
resources.xyMarkLineModes      = ["MarkLines","Lines"]
resources.xyMarker             = 1
resources.xyMarkerSizeF        = 0.05 # Default is 0.01

resources.nglFrame             = False # Don't advance the frame.

xy = Ngl.xy(wks,time,y,resources)  # Create and draw XY plot.

txresources               = Ngl.Resources()
txresources.txFontHeightF = 0.015
txresources.txJust        = "CenterLeft" # Default is "CenterCenter".
txresources.txFuncCode    = "~"          # Default is ":"

strings = ["Wigley (Moller/IPCC)",\
       "~F22~CSM-proposed:~F~~C~(Orn et.al./GEIA + Smith)",\
       "~F22~CSM SO~B~4~N~ Scaling Factor: ~V1Q~~F22~S~B~emis~N~ (yr)~H-7Q~~V-1Q~---------------~H-9Q~~V-1Q~S~B~emis~N~ (1985)"]

xpos = [1885.,1940.,1860.]   # Define X/Y locations for text.
ypos = [30.,18.,70.]

#
#  Loop through text strings and plot them.
#
for i in xrange(0,len(strings)):
  Ngl.text(wks,xy,strings[i],xpos[i],ypos[i],txresources)

Ngl.frame(wks)

Ngl.end()
