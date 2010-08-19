#
#  File:
#    coast1.py
#
#  Synopsis:
#    Draws the same map three times, showing the different resolutions
#    of coastlines that you can create.
#
#  Categories:
#    maps only
#    high-res map coastlines
#
#  Author:
#    Mary Haley (based on an NCL script of Sylvia Murphy, CGD, NCAR)
#  
#  Date of initial publication:
#    December, 2005
#
#  Description:
#    This example generates three different versions of the same map
#    area, showing the variations of coastline resolutions.
#
#  Effects illustrated:
#      o  Zooming in on a map
#      o  How to select a map database resolution.
# 
#  Output:
#     A two frame (or three frame if you have the RANGS database
#     installed) visualization is created.
#
#  Notes:
#     You must have the RANGS/GSHHS database installed in order to
#     generate the third frame of this example.
#
#     For information on getting the RANGS/GSHHS database, see:
#
#        http://www.pyngl.ucar.edu/Graphics/rangs.shtml
#

#
#  Import packages needed.
#
import Ngl
import os

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"coast1")

#
# Set some map resources.
#
mpres                             = Ngl.Resources()
mpres.mpLimitMode                 = "LatLon"
mpres.mpMinLonF                   = -15.
mpres.mpMaxLonF                   =  15
mpres.mpMinLatF                   =  40.
mpres.mpMaxLatF                   =  70.
mpres.mpGridAndLimbOn             = False
mpres.mpGeophysicalLineThicknessF = 2.
mpres.pmTitleDisplayMode          = "Always"

resolutions = ["LowRes","MediumRes","HighRes"]

#
# Loop through the map resolutions. The HighRes map is only generated
# if PyNGL can find the RANGS database.  If the PYNGL_RANGS environment
# variable is set, then PyNGL will look in this directory. Otherwise,
# it will look in the default PyNGL directory. (The Ngl.pynglpath function
# is used to determine the directory.)
#

rangs_dir = Ngl.pynglpath("rangs")     # Location of RANGS dir.
for i in range(len(resolutions)):
  mpres.tiMainString      = "mpDataBaseVersion = " + resolutions[i]
  mpres.mpDataBaseVersion = resolutions[i]

  if (resolutions[i] != "HighRes"):
    map = Ngl.map(wks,mpres)
  else:
#
# Make sure the rangs database exists before we try to generate this frame.
#
    if(os.path.exists(rangs_dir)):
      map = Ngl.map(wks,mpres)
    else:
      print "Sorry, you do not have the RANGS database installed."
      print "Not generating the third frame of this example."

Ngl.end()
