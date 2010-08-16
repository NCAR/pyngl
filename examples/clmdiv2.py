#
#  File:
#    clmdiv2.py
#
#  Synopsis:
#    Draws U.S. geographic climate divisions colored by third field.
#
#  Categories:
#    maps only
#    labelbars
#
#  Author:
#    Mary Haley (based on an NCL script of Dave Brown)
#  
#  Date of initial publication:
#    September, 2007
#
#  Description:
#    This example reads an ASCII file containing state and climate 
#    division numbers along with an associated correlation value.
#    These values are used to group the data into 20 bins and color
#    fill them.
#
#  Effects illustrated:
#      o  Defining a color map.
#      o  Using a Cylindrical Equidistant map projection.
#      o  How to select a map database resolution.
#      o  How to color only specified map areas.
#      o  How to retrieve group information to color divisions as you want.
# 
#  Output:
#     A single visualization is produced showing the
#     U.S. climate divisions.
#

#
#  Import numpy.
#
import numpy, os

#
#  Import Ngl support functions.
#
import Ngl

#
#  Open the ASCII file containing the climate division numbers
#  and additional data.
#
dirc     = Ngl.pynglpath("data")
filename = os.path.join(dirc,"asc","climdivcorr.txt")
cldata = Ngl.asciiread(filename,[345,3],"float")
clmin = min(cldata[:,2])
clmax = max(cldata[:,2])


# Group each datum into 1 of 20 equally-spaced bins
bins    = Ngl.fspan(clmin,clmax,20)
lencl   = len(cldata[:,0])
databin = numpy.zeros([lencl],'i')
for i in range(lencl):
  ii = numpy.greater_equal(bins,cldata[i,2])
  for j in range(len(ii)):
    if ii[j]:
      databin[i] = j + 1
      break
#
# Start the graphics.
#
rlist = Ngl.Resources()
rlist.wkColorMap = "gui_default"
wks_type = "ps"
wks = Ngl.open_wks (wks_type,"clmdiv2",rlist)

#
#  Map resources.
#
res = Ngl.Resources()

#
# Set the name of the map database that contains the county outlines.
#
res.mpDataSetName     = "Earth..3"
res.mpDataBaseVersion = "MediumRes"   # select the medium resolution database

#
# Note that for Earth..3, the climate divisions are the 
# equivalent of "counties".
#
res.mpOutlineSpecifiers   = "conterminous us : counties"
res.mpFillAreaSpecifiers  = "conterminous us : counties"
res.mpOutlineBoundarySets = "usstates"      # "geophysical"
res.mpFillBoundarySets    = "geophysical"

res.mpLimitMode    = "LatLon"            # Mode for zooming in on map
res.mpMinLatF      = 20.                 # select region to be plotted
res.mpMaxLatF      = 50.
res.mpMinLonF      = -130.
res.mpMaxLonF      = -65.
res.mpFillColors   = range(23)
res.mpFillOn       = True

res.mpOceanFillColor       = "transparent"
res.mpLandFillColor        = "transparent"
res.mpInlandWaterFillColor = "transparent"

res.mpOutlineOn     = True
res.mpGridAndLimbOn = False
res.pmTickMarkDisplayMode = "Never"

res.tiMainFontHeightF = 0.012
res.tiMainString      = "  ~F22~Climate divisions colored by 3rd field~C~~F21~climdivcorr.24.8.154.75.267.22.53.59.data"

res.nglDraw  = False
res.nglFrame = False

map = Ngl.map(wks,res)

#
# Get all of the area names associated with Earth..3 and the
# corresponding group numbers. The area names will be of the
# format "Arkansas : 01", where "01" is the climate division number.
#
# The group numbers are essentially color index values that were
# chosen such that adjacent areas will not have the same color
# (if you choose to color each one randomly).
#
# For example, Arkansas has 9 climate divisions, with the following
# groups:
#       Arkansas : 01, group 7
#       Arkansas : 02, group 8
#       Arkansas : 03, group 7
#       Arkansas : 04, group 6
#       Arkansas : 05, group 5
#       Arkansas : 06, group 6
#       Arkansas : 07, group 4
#       Arkansas : 08, group 7
#       Arkansas : 09, group 4
#
# From this, if you chose to use the default group numbers to color each 
# climate area, then climate areas (01,03,08) would be the same color, 
# (07,09) would be the same, and (04,06) would be the same, and 
# 02 and 05 would each be their own color.
#
anames = Ngl.get_string_array(map,"mpAreaNames")
groups = Ngl.get_integer_array(map,"mpDynamicAreaGroups")

# Areas 659 through 1116 are the states with their climates.
#  print(anames[659:1116] + ", group " + groups[659:1116])

states = ["Alabama", "Arizona", "Arkansas", "California", "Colorado",\
          "Connecticut", "Delaware", "Florida", "Georgia", "Idaho",\
          "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",\
          "Louisiana", "Maine", "Maryland", "Massasachusetts", "Michigan",\
          "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", \
          "Nevada", "New Hampshire", "New Jersey", "New Mexico", \
          "New York", "North Carolina", "North Dakota", "Ohio", \
          "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", \
          "South Carolina", "South Dakota", "Tennessee", "Texas", \
          "Utah", "Vermont", "Virginia", "Washington", "West Virginia",\
          "Wisconsin", "Wyoming"]
#
# For each state and climate division id in the data (first two
# columns), get the state name and build the area name. These are all in
# the form "State : 01". Use this name to then retrieve the index
# value into the anames array, and then use this same index to reset
# the group id to the index calculated above in the databin array.
# This will give us our new set of colors for the climate areas.
#
# Note the first four indexes (0-3) are reserved (undefined, land,
# ocean, inland water), so add 3 to start at index 4.
#
# A prerequisite was to increase the number of area groups to the number of
# bins (20) + the four predefined indexes. The result is that the areas
# corresponding to the group ids that are set will be colored based on fill
# color indexes 4-23.
#
ming = max(databin) + 5      # Use to keep track of actual min/max of
maxg = -1                    # group ids used.

for i in range(len(databin)):
  state_ix = int(cldata[i,0]) - 1
  clim_ix  = int(cldata[i,1])
  state    = states[state_ix]
  areaname = state + " : %02i" % clim_ix

  for j in range(659,1118):
    if areaname == anames[j]:         # Find the index into the anames array.
      groups[j] = databin[i] + 3     # Reset the group id
      ming = min(ming,groups[j])
      maxg = max(maxg,groups[j])

#
# Increase the area group count (which lets you have more
# different colors) and set the new group ids.
#
sres = Ngl.Resources()
sres.mpAreaGroupCount = 24         # Default is 10
sres.mpDynamicAreaGroups = groups
Ngl.set_values(map,sres)

#
# Create a labelbar using the min/max group ids that are actually
# in use.
#
colors = range(ming,maxg+1)
nboxes = len(colors)

# Get an array of strings for our labelbar boxes.
labels = str(bins).split()

# Format the end labels
labels[0]     = "%.2f" % min(bins)
labels.append("%.2f" % max(bins))

# Blank out rest of labels
for j in range(1,nboxes):
  labels[j] = " "

lbres                    = Ngl.Resources()
lbres.vpWidthF           = 0.70
lbres.vpHeightF          = 0.10
lbres.lbPerimOn          = False            # Turn off perimeter.
lbres.lbOrientation      = "Horizontal"     # Default is vertical.
lbres.lbLabelAlignment   = "ExternalEdges"  
lbres.lbFillColors       = colors
lbres.lbMonoFillPattern  = True             # Fill them all solid.
lbres.lbLabelFontHeightF = 0.013            # label font height
lbres.nglDraw            = False

# The labelbar will be created, but not drawn at this point.
lbid = Ngl.labelbar_ndc(wks,nboxes,labels,0.5,0.2,lbres)

#
# Now create some annotation resources indicating how we want to
# attach the labelbar to the plot. Here, we are using the top center
# of the labelbar as the point which we are going to position
# it, and then we use amOrthogonalPosF to move it down.
#
# amParallelPosF/amOrthogonalPosF
#    0.0/ 0.0  -  annotation in dead center of plot
#    0.5/ 0.5  -  annotation at bottom right of plot
#    0.5/-0.5  -  annotation at top right of plot
#   -0.5/-0.5  -  annotation at top left of plot
#   -0.5/ 0.5  -  annotation at bottom left of plot
#                                                                 
amres = Ngl.Resources()
amres.amJust           = "TopCenter"
amres.amOrthogonalPosF = 0.4
annoid = Ngl.add_annotation(map,lbid,amres)

Ngl.draw(map)
Ngl.frame(wks)

Ngl.end ()
