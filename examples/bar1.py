#
#  File:
#    bar1.py
#
#  Synopsis:
#    Illustrates how to draw bar charts.
#
#  Categories:
#    xy plots
#    bar charts
#    polygons
#    polylines
#    text
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    March 2008
#
#  Description:
#    This example shows how to generate bar charts, using polylines and
#    polygons.
#
#  Effects illustrated:
#    o  Drawing primitives on a plot.
#    o  Drawing text on a plot.
# 
#  Output:
#     This example produces two frames.
#
#  Notes:
#     
import numpy
import Ngl

#
# This script plots, as a series of bars, the number of times US
# baseball teams have won or lost the World Series.
#

# 
# Function that returns coordinates of a bar, given the x,y values,
# the dx (between bars), the width of the bar as a percentage of the
# distance between x values (bar_width_perc), and the minimum y to
# start the bar from.
#
def get_bar(x,y,dx,ymin,bar_width_perc=0.6):
  dxp = (dx * bar_width_perc)/2.
  xbar = numpy.array([x-dxp,x+dxp,x+dxp,x-dxp,x-dxp])
  ybar = numpy.array([ ymin, ymin,    y,    y, ymin])
  return xbar,ybar

# 
# Main program
#

#
# Define long and short name of baseball teams, along with their 
# colors in RGB percentages. Note that some of these teams became
# other teams, but the original and the new names are both included.
#
teams = {
     "Anaheim Angels"       : {"abbrev" : "AA",  "colors" : [  73,  0, 13]}, \
     "Arizona Diamondbacks" : {"abbrev" : "AD",  "colors" : [  39, 21, 63]}, \
     "Atlanta Braves"       : {"abbrev" : "AB",  "colors" : [  69,  1, 22]}, \
     "Baltimore Orioles"    : {"abbrev" : "BO",  "colors" : [  82, 35,  4]}, \
     "Boston Braves"        : {"abbrev" : "BB",  "colors" : [  66, 13, 24]}, \
     "Boston Red Sox"       : {"abbrev" : "BRS", "colors" : [ 73, 19, 24]}, \
     "Brooklyn Dodgers"     : {"abbrev" : "BD",  "colors" : [   5, 18, 52]}, \
     "Brooklyn Robins"      : {"abbrev" : "BR",  "colors" : [   3, 16, 52]}, \
     "Chicago Cubs"         : {"abbrev" : "CC",  "colors" : [   6, 20, 53]}, \
     "Chicago White Sox"    : {"abbrev" : "CWS", "colors" : [  0,  0,  0]}, \
     "Cincinnati Reds"      : {"abbrev" : "CRe", "colors" : [ 78,  0, 12]}, \
     "Cleveland Indians"    : {"abbrev" : "CI",  "colors" : [   1, 20, 40]}, \
     "Colorado Rockies"     : {"abbrev" : "CRo", "colors" : [ 20, 20, 40]}, \
     "Detroit Tigers"       : {"abbrev" : "DT",  "colors" : [  92, 49, 22]}, \
     "Florida Marlins"      : {"abbrev" : "FM",  "colors" : [  14, 62, 64]}, \
     "Houston Astros"       : {"abbrev" : "HA",  "colors" : [  58, 20, 17]}, \
     "Kansas City Royals"   : {"abbrev" : "KCR", "colors" : [  0,  2, 45]}, \
     "Los Angeles Dodgers"  : {"abbrev" : "LAD", "colors" : [  3, 24, 42]}, \
     "Milwaukee Braves"     : {"abbrev" : "MBa", "colors" : [ 93,  9, 12]}, \
     "Milwaukee Brewers"    : {"abbrev" : "MBe", "colors" : [  4, 13, 32]}, \
     "Minnesota Twins"      : {"abbrev" : "MT",  "colors" : [  74,  0, 20]}, \
     "New York Giants"      : {"abbrev" : "NYG", "colors" : [  2,  6, 26]}, \
     "New York Mets"        : {"abbrev" : "NYM", "colors" : [  1, 17, 40]}, \
     "New York Yankees"     : {"abbrev" : "NYY", "colors" : [ 11, 16, 26]}, \
     "Oakland Athletics"    : {"abbrev" : "OA",  "colors" : [   0, 22, 19]}, \
     "Philadelphia Athletics": {"abbrev" : "PA", "colors" : [   7,  0, 55]}, \
     "Philadelphia Phillies": {"abbrev" : "PPh", "colors" : [ 91,  9, 16]}, \
     "Pittsburgh Pirates"   : {"abbrev" : "PPi", "colors" : [ 46, 38, 13]}, \
     "San Diego Padres"     : {"abbrev" : "SDP", "colors" : [  2,  8, 25]}, \
     "San Francisco Giants" : {"abbrev" : "SFG", "colors" : [100, 35, 12]}, \
     "Seattle Mariners"     : {"abbrev" : "SM",  "colors" : [   5, 17, 34]}, \
     "St. Louis Browns"     : {"abbrev" : "SLB", "colors" : [ 81, 32, 14]}, \
     "St. Louis Cardinals"  : {"abbrev" : "SLC", "colors" : [ 77, 12, 23]}, \
     "Tampa Bay Devil Rays" : {"abbrev" : "TBDR","colors" : [ 0, 43, 24]}, \
     "Texas Rangers"        : {"abbrev" : "TR",  "colors" : [   0, 20, 48]}, \
     "Toronto Blue Jays"    : {"abbrev" : "TBJ", "colors" : [  0, 20, 60]}, \
     "Washington Senators"  : {"abbrev" : "WS",  "colors" : [   0,  4, 25]}, \
     "Washington Nationals" : {"abbrev" : "WN",  "colors" : [   7, 13, 36]}  \
        }
#
# Special notes: 
#
#  - No world series in 1904 (boycotted by New York Giants) 
#    or 1994 (strike).
#
#  - Boston Red Sox first won their first WS as the Boston Americans.
#    I didn't include a separate entry for the Americans because the
#    bars were getting too thin.
#
#  - Brooklyn Robins became the Brooklyn Dodgers became the LA Dodgers.
#
#  - Boston Braves became the Milwaukee Braves became the Atlanta Braves.
#
#  - St. Louis Browns became the Baltimore Orioles.
#
#  - Washington Senators became the Minnesota Twins.
#
#  - New York Giants became the San Francisco Giants.
#
#  - Philadelphia Athletics became the Oakland Athletics.

#
# List each world series results as {year, [winning team, losing team]}.
#
world_series = {
                1903 : ["BRS", "PPi"] ,\
                1904 : ["",    ""], \
                1905 : ["NYG", "PA"], \
                1906 : ["CWS", "CC"], \
                1907 : ["CC",  "DT"], \
                1908 : ["CC",  "DT"], \
                1909 : ["PPi", "DT"], \
                1910 : ["PA",  "CC"], \
                1911 : ["PA",  "NYG"], \
                1912 : ["BRS", "NYG"], \
                1913 : ["PA",  "NYG"], \
                1914 : ["BB",  "PA"], \
                1915 : ["BRS", "PPh"], \
                1916 : ["BRS", "BR"], \
                1917 : ["CWS", "NYG"], \
                1918 : ["BRS", "CC"], \
                1919 : ["CRe", "CWS"], \
                1920 : ["CI",  "BR"], \
                1921 : ["NYG", "NYY"], \
                1922 : ["NYG", "NYY"], \
                1923 : ["NYY", "NYG"], \
                1924 : ["WS",  "NYG"], \
                1925 : ["PPi", "WS"], \
                1926 : ["SLC", "NYY"], \
                1927 : ["NYY", "PPi"], \
                1928 : ["NYY", "SLC"], \
                1929 : ["PA",  "CC"], \
                1930 : ["PA",  "SLC"], \
                1931 : ["SLC", "PA"], \
                1932 : ["NYY", "CC"], \
                1933 : ["NYG", "WS"], \
                1934 : ["SLC", "DT"], \
                1935 : ["DT",  "CC"], \
                1936 : ["NYY", "NYG"], \
                1937 : ["NYY", "NYG"], \
                1938 : ["NYY", "CC"], \
                1939 : ["NYY", "CRe"], \
                1940 : ["CRe", "DT"], \
                1941 : ["NYY", "BD"], \
                1942 : ["SLC", "NYY"], \
                1943 : ["NYY", "SLC"], \
                1944 : ["SLC", "SLB"], \
                1945 : ["DT",  "CC"], \
                1946 : ["SLC", "BRS"], \
                1947 : ["NYY", "BD"], \
                1948 : ["CI",  "BB"], \
                1949 : ["NYY", "BD"], \
                1950 : ["NYY", "PPh"], \
                1951 : ["NYY", "NYG"], \
                1952 : ["NYY", "BD"], \
                1953 : ["NYY", "BD"], \
                1954 : ["NYG", "CI"], \
                1955 : ["BD",  "NYY"], \
                1956 : ["NYY", "BD"], \
                1957 : ["MBa", "NYY"], \
                1958 : ["NYY", "MBa"], \
                1959 : ["LAD", "CWS"], \
                1960 : ["PPi", "NYY"], \
                1961 : ["NYY", "CRe"], \
                1962 : ["NYY", "SFG"], \
                1963 : ["LAD", "NYY"], \
                1964 : ["SLC", "NYY"], \
                1965 : ["LAD", "MT"], \
                1966 : ["BO",  "LAD"], \
                1967 : ["SLC", "BRS"], \
                1968 : ["DT",  "SLC"], \
                1969 : ["NYM", "BO"], \
                1970 : ["BO",  "CRe"], \
                1971 : ["PPi", "BO"], \
                1972 : ["OA",  "CRe"], \
                1973 : ["OA",  "NYM"], \
                1974 : ["OA",  "LAD"], \
                1975 : ["CRe", "BRS"], \
                1976 : ["CRe", "NYY"], \
                1977 : ["NYY", "LAD"], \
                1978 : ["NYY", "LAD"], \
                1979 : ["PPi", "BO"], \
                1980 : ["PPh", "KCR"], \
                1981 : ["LAD", "NYY"], \
                1982 : ["SLC", "MBe"], \
                1983 : ["BO",  "PPh"], \
                1984 : ["DT",  "SDP"], \
                1985 : ["KCR", "SLC"], \
                1986 : ["NYM", "BRS"], \
                1987 : ["MT",  "SLC"], \
                1988 : ["LAD", "OA"], \
                1989 : ["OA",  "SFG"], \
                1990 : ["CRe", "OA"], \
                1991 : ["MT",  "AB"], \
                1992 : ["TBJ", "AB"], \
                1993 : ["TBJ", "PPh"], \
                1994 : ["",    ""], \
                1995 : ["AB",  "CI"], \
                1996 : ["NYY", "AB"], \
                1997 : ["FM",  "CI"], \
                1998 : ["NYY", "SDP"], \
                1999 : ["NYY", "AB"], \
                2000 : ["NYY", "NYM"], \
                2001 : ["AD",  "NYY"], \
                2002 : ["AA",  "SFG"], \
                2003 : ["FM",  "NYY"], \
                2004 : ["BRS", "SLC"], \
                2005 : ["CWS", "HA"], \
                2006 : ["SLC", "DT"], \
                2007 : ["BRS", "CRo"], \
                2008 : ["PPh", "TBR"], \
                2009 : ["NYY", "Pph"], \
                2010 : ["SFG", "TR"] \
           }

#
# Count the number of times each team has won or lost and store
# in an array.
#
nteams = len(teams)               # Number of teams
max_year = max(world_series.keys())
values = numpy.array(world_series.values())
won    = values[:,0].tolist()     # Index 0 = winning teams
lost   = values[:,1].tolist()     # Index 1 = losing teams
ngames = len(won)                 # Number of games

#
# Set up arrays to Define colors for each winning/losing team.
#
winning_colors      = numpy.zeros((ngames+1,3),'f')
losing_colors       = numpy.zeros((ngames+1,3),'f')
winning_colors[0,:] = [1.,1.,1.]
winning_colors[1,:] = [0.,0.,0.]
winning_colors[ngames,:] = [0.9,0.9,0.9]    # Gray
losing_colors[0,:]  = [1.,1.,1.]
losing_colors[1,:]  = [0.,0.,0.]
losing_colors[ngames,:] = [0.9,0.9,0.9]    # Gray

#
# Loop through each team and count the number of teams it won and/or
# lost the world series. If this number is > 0, then store in an
# array.
#
winning_teams_ct = []
winning_teams_nm = []
losing_teams_ct  = []
losing_teams_nm  = []
nw = 2                     # color index counter for winning team
nl = 2                     # color index counter for losing team

sorted_teams = teams.keys()
sorted_teams.sort()           # Sort the team names
for team in sorted_teams:
  steam = teams[team]["abbrev"]
  if won.count(steam) > 0:
    winning_teams_ct.append(won.count(steam))   # Here's the count
    winning_teams_nm.append(team)               # Name of winning team
    winning_colors[nw,:] = numpy.array(teams[team]["colors"])/100.
    nw += 1

  if lost.count(steam) > 0:
    losing_teams_ct.append(lost.count(steam))   # Here's the count
    losing_teams_nm.append(team)                # Name of losing team
    losing_colors[nl,:] = numpy.array(teams[team]["colors"])/100.
    nl += 1
  
# Store the winning and losing counts in numpy arrays.
y_win  = numpy.array(winning_teams_ct)
x_win  = numpy.array(range(1,y_win.shape[0]+1))
y_lose = numpy.array(losing_teams_ct)
x_lose = numpy.array(range(1,y_lose.shape[0]+1))

#
# Start the graphics portion of the script.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"bar1")   # Open an X11 window
Ngl.define_colormap(wks,winning_colors)

res = Ngl.Resources()

res.nglMaximize           = False       # Need to set to False if using
                                        # vp resources.

res.vpXF                  = 0.12        # Move plot to left a little.
res.vpYF                  = 0.98        # Move plot up a little.
res.vpHeightF             = 0.90        # Make plot higher than
res.vpWidthF              = 0.80        # it is wide.

res.tmXBOn                = False       # Turn off bottom tickmarks & labes
res.tmXTOn                = False       # Turn off top tickmarks & labes
res.tmYROn                = False       # Turn off right tickmarks & labes
res.tmYLMinorOn           = False       # Turn off left minor tickmarks
res.tmEqualizeXYSizes     = False       # Don't try to equalize the lengths
                                        # of the tickmarks.

res.tmYLMajorLengthF            = 0.01       # Total length
res.tmYLMajorOutwardLengthF     = 0.01       # Outward length

res.trYMinF               = 0              # Minimum value on Y axis
res.trXMinF               = 0              # Minimum value on X axis.
res.trYMaxF               = max(y_win)+1   # Maximum value on Y axis.
res.trXMaxF               = max(x_win)+1   # Maximum value on X axis.
  
res.tiXAxisString          = "# of World Series Wins through " + str(max_year)
res.tiXAxisFontHeightF     = 0.03

res.nglFrame              = False          # Don't advance frame.

ymin           = 0.                                # For bar plot.
dx             = min(x_win[1:-1]-x_win[0:-2])      # Distance between X values.
bar_width_perc = 0.6                  
bar_width      = bar_width_perc * dx               # Bar width.

txres               = Ngl.Resources()    # Resource list for text strings.
txres.txFontHeightF = 0.015

gsres = Ngl.Resources()                  # Resource list for bars.
#
# Plot results for winning teams.
#
# Loop through each value, and create and draw a bar for it.
#
imax = numpy.where(y_win == max(y_win))[0]
for i in xrange(len(y_win)):
  xbar,ybar = get_bar(x_win[i],y_win[i],dx,ymin)
  plot = Ngl.xy(wks,xbar,ybar,res)

  gsres.gsFillColor = [i+2]               # Set color for bar.
  Ngl.polygon(wks,plot,xbar,ybar,gsres)   # Fill the bar.
  Ngl.polyline(wks,plot,xbar,ybar)        # Outline the bar.
  xbar,ybar = get_bar(x_win[i],y_win[i],dx,ymin)
#
# Put names of teams vertically above the bar. Have to treat the team
# with the most wins (NY Yankees as of 2008) special because
# the text runs off the screen otherwise.
#
  if i == imax:
    txres.txJust   = "BottomCenter"
    txres.txAngleF = 0. 
    Ngl.text(wks,plot,winning_teams_nm[i],x_win[i],y_win[i],txres)
  else:
    txres.txJust   = "CenterLeft"
    txres.txAngleF = 90. 
    Ngl.text(wks,plot," " + winning_teams_nm[i],x_win[i],y_win[i],txres)

Ngl.frame(wks)

#
# Now plot losing team results.
#
Ngl.define_colormap(wks,losing_colors)

dx        = min(x_lose[1:-1]-x_lose[0:-2])    # Distance between X values.
bar_width = bar_width_perc * dx               # Bar width.

res.trYMaxF               = max(y_lose)+1   # Maximum value on Y axis.
res.trXMaxF               = max(x_lose)+1   # Maximum value on X axis.
res.tiXAxisString         = "# of World Series Losses through " + str(max_year)

#
# Loop through each value, and create and draw a bar for it.
#
imax = numpy.where(y_lose == max(y_lose))[0]

for i in xrange(len(y_lose)):
  xbar,ybar = get_bar(x_lose[i],y_lose[i],dx,ymin)
  plot = Ngl.xy(wks,xbar,ybar,res)

  gsres.gsFillColor = [i+2]               # Set color for bar.
  Ngl.polygon(wks,plot,xbar,ybar,gsres)   # Fill the bar.
  Ngl.polyline(wks,plot,xbar,ybar)        # Outline the bar.
#
# Put names of teams vertically above the bar. Have to treat the team
# with the most losses (NY Yankees as of 2008) special because
# the text runs off the screen otherwise.
#
  if i == imax:
    txres.txJust   = "BottomCenter"
    txres.txAngleF = 0. 
    Ngl.text(wks,plot,losing_teams_nm[i],x_lose[i],y_lose[i],txres)
  else:
    txres.txJust   = "CenterLeft"
    txres.txAngleF = 90. 
    Ngl.text(wks,plot," " + losing_teams_nm[i],x_lose[i],y_lose[i],txres)

Ngl.frame(wks)

Ngl.end()
