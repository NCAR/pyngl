#
#  File:
#    taylor_diagram1.py
#
#  Synopsis:
#    Illustrates how to create a taylor diagram.
#
#  Categories:
#    taylor diagrams
#
#  Author:
#    Fred Castruccio
#  
#  Date of initial publication:
#    March 2015
#
#  Description:
#    This example shows how to use the taylor_diagram function added
#    in PyNGL 1.5.0 to create a taylor diagram plot.
#
#  Effects illustrated:
# 
#  Output:
#     A single visualization is produced.
#
#  Notes:
#     

from __future__ import print_function
import numpy as np
import Ngl

# Cases [Model]
case      = [ "Case A", "Case B" ]
nCase     = np.size( case )                 # # of Cases [Cases]

# variables compared
var       = [ "SLP", "Tsfc", "Prc", "Prc 30S-30N", "LW", "SW", "U300", "Guess" ]
nVar      = np.size(var)                    # # of Variables

# "Case A"                        
CA_ratio   = np.array([1.230, 0.988, 1.092, 1.172, 1.064, 0.966, 1.079, 0.781])
CA_cc      = np.array([0.958, 0.973, 0.740, 0.743, 0.922, 0.982, 0.952, 0.433])

# "Case B" 
CB_ratio   = np.array([1.129, 0.996, 1.016, 1.134, 1.023, 0.962, 1.048, 0.852])
CB_cc      = np.array([0.963, 0.975, 0.801, 0.814, 0.946, 0.984, 0.968, 0.647])

# arrays to be passed to taylor plot 
ratio      = np.zeros((nCase, nVar))  
cc         = np.zeros((nCase, nVar)) 

ratio[0,:] = CA_ratio 
ratio[1,:] = CB_ratio

cc[0,:]    = CA_cc 
cc[1,:]    = CB_cc



#**********************************
# create plot
#**********************************

res = Ngl.Resources()                   # default taylor diagram
        
res.Markers      = [16, 16]             # make all solid fill
res.Colors       = ["red", "blue" ]
res.varLabels    = var
res.caseLabels   = case

res.caseLabelsXloc = 1.2                # Move location of variable labels [default 0.45]
res.caseLabelsYloc = 1.7                # Move location of variable labels [default 0.45]
res.caseLabelsFontHeightF = 0.025       # make slight larger   [default=0.12 ]
res.varLabelsYloc = 1.5                 # Move location of variable labels [default 0.45]
res.varLabelsFontHeightF  = 0.02        # make slight smaller  [default=0.013]

res.stnRad        = [ 0.5, 4 ]          # additional standard radii
res.ccRays        = [ 0.6, 0.9 ]        # correllation rays
res.centerDiffRMS = True                # RMS 'circles'

wks_type = "png"
wks = Ngl.open_wks(wks_type,"taylor_diagram1")
plot  = Ngl.taylor_diagram(wks,ratio,cc,res)

del res

