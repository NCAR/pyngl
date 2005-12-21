# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.
import _hlu
def _swig_setattr(self,class_type,name,value):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    self.__dict__[name] = value

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


NhlBACKGROUND = _hlu.NhlBACKGROUND
NhlTOPLEFT = _hlu.NhlTOPLEFT
NhlCENTERLEFT = _hlu.NhlCENTERLEFT
NhlBOTTOMLEFT = _hlu.NhlBOTTOMLEFT
NhlTOPCENTER = _hlu.NhlTOPCENTER
NhlCENTERCENTER = _hlu.NhlCENTERCENTER
NhlBOTTOMCENTER = _hlu.NhlBOTTOMCENTER
NhlTOPRIGHT = _hlu.NhlTOPRIGHT
NhlCENTERRIGHT = _hlu.NhlCENTERRIGHT
NhlBOTTOMRIGHT = _hlu.NhlBOTTOMRIGHT
NhlNwkOrientation = _hlu.NhlNwkOrientation
NhlNamDataXF = _hlu.NhlNamDataXF
NhlNamDataYF = _hlu.NhlNamDataYF
NhlNamJust = _hlu.NhlNamJust
NhlNamOn = _hlu.NhlNamOn
NhlNamOrthogonalPosF = _hlu.NhlNamOrthogonalPosF
NhlNamParallelPosF = _hlu.NhlNamParallelPosF
NhlNamResizeNotify = _hlu.NhlNamResizeNotify
NhlNamSide = _hlu.NhlNamSide
NhlNamTrackData = _hlu.NhlNamTrackData
NhlNamViewId = _hlu.NhlNamViewId
NhlNamZone = _hlu.NhlNamZone
NhlNappDefaultParent = _hlu.NhlNappDefaultParent
NhlNappFileSuffix = _hlu.NhlNappFileSuffix
NhlNappResources = _hlu.NhlNappResources
NhlNappSysDir = _hlu.NhlNappSysDir
NhlNappUsrDir = _hlu.NhlNappUsrDir
NhlNcaCopyArrays = _hlu.NhlNcaCopyArrays
NhlNcaXArray = _hlu.NhlNcaXArray
NhlNcaXCast = _hlu.NhlNcaXCast
NhlNcaXMaxV = _hlu.NhlNcaXMaxV
NhlNcaXMinV = _hlu.NhlNcaXMinV
NhlNcaXMissingV = _hlu.NhlNcaXMissingV
NhlNcaYArray = _hlu.NhlNcaYArray
NhlNcaYCast = _hlu.NhlNcaYCast
NhlNcaYMaxV = _hlu.NhlNcaYMaxV
NhlNcaYMinV = _hlu.NhlNcaYMinV
NhlNcaYMissingV = _hlu.NhlNcaYMissingV
NhlNcnConpackParams = _hlu.NhlNcnConpackParams
NhlNcnConstFLabelAngleF = _hlu.NhlNcnConstFLabelAngleF
NhlNcnConstFLabelBackgroundColor = _hlu.NhlNcnConstFLabelBackgroundColor
NhlNcnConstFLabelConstantSpacingF = _hlu.NhlNcnConstFLabelConstantSpacingF
NhlNcnConstFLabelFont = _hlu.NhlNcnConstFLabelFont
NhlNcnConstFLabelFontAspectF = _hlu.NhlNcnConstFLabelFontAspectF
NhlNcnConstFLabelFontColor = _hlu.NhlNcnConstFLabelFontColor
NhlNcnConstFLabelFontHeightF = _hlu.NhlNcnConstFLabelFontHeightF
NhlNcnConstFLabelFontQuality = _hlu.NhlNcnConstFLabelFontQuality
NhlNcnConstFLabelFontThicknessF = _hlu.NhlNcnConstFLabelFontThicknessF
NhlNcnConstFLabelFormat = _hlu.NhlNcnConstFLabelFormat
NhlNcnConstFLabelFuncCode = _hlu.NhlNcnConstFLabelFuncCode
NhlNcnConstFLabelJust = _hlu.NhlNcnConstFLabelJust
NhlNcnConstFLabelOn = _hlu.NhlNcnConstFLabelOn
NhlNcnConstFLabelOrthogonalPosF = _hlu.NhlNcnConstFLabelOrthogonalPosF
NhlNcnConstFLabelParallelPosF = _hlu.NhlNcnConstFLabelParallelPosF
NhlNcnConstFLabelPerimColor = _hlu.NhlNcnConstFLabelPerimColor
NhlNcnConstFLabelPerimOn = _hlu.NhlNcnConstFLabelPerimOn
NhlNcnConstFLabelPerimSpaceF = _hlu.NhlNcnConstFLabelPerimSpaceF
NhlNcnConstFLabelPerimThicknessF = _hlu.NhlNcnConstFLabelPerimThicknessF
NhlNcnConstFLabelSide = _hlu.NhlNcnConstFLabelSide
NhlNcnConstFLabelString = _hlu.NhlNcnConstFLabelString
NhlNcnConstFLabelTextDirection = _hlu.NhlNcnConstFLabelTextDirection
NhlNcnConstFLabelZone = _hlu.NhlNcnConstFLabelZone
NhlNcnConstFUseInfoLabelRes = _hlu.NhlNcnConstFUseInfoLabelRes
NhlNcnExplicitLabelBarLabelsOn = _hlu.NhlNcnExplicitLabelBarLabelsOn
NhlNcnExplicitLegendLabelsOn = _hlu.NhlNcnExplicitLegendLabelsOn
NhlNcnExplicitLineLabelsOn = _hlu.NhlNcnExplicitLineLabelsOn
NhlNcnFillBackgroundColor = _hlu.NhlNcnFillBackgroundColor
NhlNcnFillColor = _hlu.NhlNcnFillColor
NhlNcnFillColors = _hlu.NhlNcnFillColors
NhlNcnFillDrawOrder = _hlu.NhlNcnFillDrawOrder
NhlNcnFillOn = _hlu.NhlNcnFillOn
NhlNcnFillPattern = _hlu.NhlNcnFillPattern
NhlNcnFillPatterns = _hlu.NhlNcnFillPatterns
NhlNcnFillScaleF = _hlu.NhlNcnFillScaleF
NhlNcnFillScales = _hlu.NhlNcnFillScales
NhlNcnFixFillBleed = _hlu.NhlNcnFixFillBleed
NhlNcnGridBoundPerimColor = _hlu.NhlNcnGridBoundPerimColor
NhlNcnGridBoundPerimDashPattern = _hlu.NhlNcnGridBoundPerimDashPattern
NhlNcnGridBoundPerimOn = _hlu.NhlNcnGridBoundPerimOn
NhlNcnGridBoundPerimThicknessF = _hlu.NhlNcnGridBoundPerimThicknessF
NhlNcnHighLabelAngleF = _hlu.NhlNcnHighLabelAngleF
NhlNcnHighLabelBackgroundColor = _hlu.NhlNcnHighLabelBackgroundColor
NhlNcnHighLabelConstantSpacingF = _hlu.NhlNcnHighLabelConstantSpacingF
NhlNcnHighLabelFont = _hlu.NhlNcnHighLabelFont
NhlNcnHighLabelFontAspectF = _hlu.NhlNcnHighLabelFontAspectF
NhlNcnHighLabelFontColor = _hlu.NhlNcnHighLabelFontColor
NhlNcnHighLabelFontHeightF = _hlu.NhlNcnHighLabelFontHeightF
NhlNcnHighLabelFontQuality = _hlu.NhlNcnHighLabelFontQuality
NhlNcnHighLabelFontThicknessF = _hlu.NhlNcnHighLabelFontThicknessF
NhlNcnHighLabelFormat = _hlu.NhlNcnHighLabelFormat
NhlNcnHighLabelFuncCode = _hlu.NhlNcnHighLabelFuncCode
NhlNcnHighLabelPerimColor = _hlu.NhlNcnHighLabelPerimColor
NhlNcnHighLabelPerimOn = _hlu.NhlNcnHighLabelPerimOn
NhlNcnHighLabelPerimSpaceF = _hlu.NhlNcnHighLabelPerimSpaceF
NhlNcnHighLabelPerimThicknessF = _hlu.NhlNcnHighLabelPerimThicknessF
NhlNcnHighLabelString = _hlu.NhlNcnHighLabelString
NhlNcnHighLabelsOn = _hlu.NhlNcnHighLabelsOn
NhlNcnHighLowLabelOverlapMode = _hlu.NhlNcnHighLowLabelOverlapMode
NhlNcnHighUseLineLabelRes = _hlu.NhlNcnHighUseLineLabelRes
NhlNcnInfoLabelAngleF = _hlu.NhlNcnInfoLabelAngleF
NhlNcnInfoLabelBackgroundColor = _hlu.NhlNcnInfoLabelBackgroundColor
NhlNcnInfoLabelConstantSpacingF = _hlu.NhlNcnInfoLabelConstantSpacingF
NhlNcnInfoLabelFont = _hlu.NhlNcnInfoLabelFont
NhlNcnInfoLabelFontAspectF = _hlu.NhlNcnInfoLabelFontAspectF
NhlNcnInfoLabelFontColor = _hlu.NhlNcnInfoLabelFontColor
NhlNcnInfoLabelFontHeightF = _hlu.NhlNcnInfoLabelFontHeightF
NhlNcnInfoLabelFontQuality = _hlu.NhlNcnInfoLabelFontQuality
NhlNcnInfoLabelFontThicknessF = _hlu.NhlNcnInfoLabelFontThicknessF
NhlNcnInfoLabelFormat = _hlu.NhlNcnInfoLabelFormat
NhlNcnInfoLabelFuncCode = _hlu.NhlNcnInfoLabelFuncCode
NhlNcnInfoLabelJust = _hlu.NhlNcnInfoLabelJust
NhlNcnInfoLabelOn = _hlu.NhlNcnInfoLabelOn
NhlNcnInfoLabelOrthogonalPosF = _hlu.NhlNcnInfoLabelOrthogonalPosF
NhlNcnInfoLabelParallelPosF = _hlu.NhlNcnInfoLabelParallelPosF
NhlNcnInfoLabelPerimColor = _hlu.NhlNcnInfoLabelPerimColor
NhlNcnInfoLabelPerimOn = _hlu.NhlNcnInfoLabelPerimOn
NhlNcnInfoLabelPerimSpaceF = _hlu.NhlNcnInfoLabelPerimSpaceF
NhlNcnInfoLabelPerimThicknessF = _hlu.NhlNcnInfoLabelPerimThicknessF
NhlNcnInfoLabelSide = _hlu.NhlNcnInfoLabelSide
NhlNcnInfoLabelString = _hlu.NhlNcnInfoLabelString
NhlNcnInfoLabelTextDirection = _hlu.NhlNcnInfoLabelTextDirection
NhlNcnInfoLabelZone = _hlu.NhlNcnInfoLabelZone
NhlNcnLabelBarEndLabelsOn = _hlu.NhlNcnLabelBarEndLabelsOn
NhlNcnLabelDrawOrder = _hlu.NhlNcnLabelDrawOrder
NhlNcnLabelMasking = _hlu.NhlNcnLabelMasking
NhlNcnLabelScaleFactorF = _hlu.NhlNcnLabelScaleFactorF
NhlNcnLabelScaleValueF = _hlu.NhlNcnLabelScaleValueF
NhlNcnLabelScalingMode = _hlu.NhlNcnLabelScalingMode
NhlNcnLegendLevelFlags = _hlu.NhlNcnLegendLevelFlags
NhlNcnLevelCount = _hlu.NhlNcnLevelCount
NhlNcnLevelFlag = _hlu.NhlNcnLevelFlag
NhlNcnLevelFlags = _hlu.NhlNcnLevelFlags
NhlNcnLevelSelectionMode = _hlu.NhlNcnLevelSelectionMode
NhlNcnLevelSpacingF = _hlu.NhlNcnLevelSpacingF
NhlNcnLevels = _hlu.NhlNcnLevels
NhlNcnLineColor = _hlu.NhlNcnLineColor
NhlNcnLineColors = _hlu.NhlNcnLineColors
NhlNcnLineDashPattern = _hlu.NhlNcnLineDashPattern
NhlNcnLineDashPatterns = _hlu.NhlNcnLineDashPatterns
NhlNcnLineDashSegLenF = _hlu.NhlNcnLineDashSegLenF
NhlNcnLineDrawOrder = _hlu.NhlNcnLineDrawOrder
NhlNcnLineLabelAngleF = _hlu.NhlNcnLineLabelAngleF
NhlNcnLineLabelBackgroundColor = _hlu.NhlNcnLineLabelBackgroundColor
NhlNcnLineLabelConstantSpacingF = _hlu.NhlNcnLineLabelConstantSpacingF
NhlNcnLineLabelFont = _hlu.NhlNcnLineLabelFont
NhlNcnLineLabelFontAspectF = _hlu.NhlNcnLineLabelFontAspectF
NhlNcnLineLabelFontColor = _hlu.NhlNcnLineLabelFontColor
NhlNcnLineLabelFontColors = _hlu.NhlNcnLineLabelFontColors
NhlNcnLineLabelFontHeightF = _hlu.NhlNcnLineLabelFontHeightF
NhlNcnLineLabelFontQuality = _hlu.NhlNcnLineLabelFontQuality
NhlNcnLineLabelFontThicknessF = _hlu.NhlNcnLineLabelFontThicknessF
NhlNcnLineLabelFormat = _hlu.NhlNcnLineLabelFormat
NhlNcnLineLabelFuncCode = _hlu.NhlNcnLineLabelFuncCode
NhlNcnLineLabelInterval = _hlu.NhlNcnLineLabelInterval
NhlNcnLineLabelPerimColor = _hlu.NhlNcnLineLabelPerimColor
NhlNcnLineLabelPerimOn = _hlu.NhlNcnLineLabelPerimOn
NhlNcnLineLabelPerimSpaceF = _hlu.NhlNcnLineLabelPerimSpaceF
NhlNcnLineLabelPerimThicknessF = _hlu.NhlNcnLineLabelPerimThicknessF
NhlNcnLineLabelPlacementMode = _hlu.NhlNcnLineLabelPlacementMode
NhlNcnLineLabelStrings = _hlu.NhlNcnLineLabelStrings
NhlNcnLineLabelsOn = _hlu.NhlNcnLineLabelsOn
NhlNcnLineThicknessF = _hlu.NhlNcnLineThicknessF
NhlNcnLineThicknesses = _hlu.NhlNcnLineThicknesses
NhlNcnLinesOn = _hlu.NhlNcnLinesOn
NhlNcnLowLabelAngleF = _hlu.NhlNcnLowLabelAngleF
NhlNcnLowLabelBackgroundColor = _hlu.NhlNcnLowLabelBackgroundColor
NhlNcnLowLabelConstantSpacingF = _hlu.NhlNcnLowLabelConstantSpacingF
NhlNcnLowLabelFont = _hlu.NhlNcnLowLabelFont
NhlNcnLowLabelFontAspectF = _hlu.NhlNcnLowLabelFontAspectF
NhlNcnLowLabelFontColor = _hlu.NhlNcnLowLabelFontColor
NhlNcnLowLabelFontHeightF = _hlu.NhlNcnLowLabelFontHeightF
NhlNcnLowLabelFontQuality = _hlu.NhlNcnLowLabelFontQuality
NhlNcnLowLabelFontThicknessF = _hlu.NhlNcnLowLabelFontThicknessF
NhlNcnLowLabelFormat = _hlu.NhlNcnLowLabelFormat
NhlNcnLowLabelFuncCode = _hlu.NhlNcnLowLabelFuncCode
NhlNcnLowLabelPerimColor = _hlu.NhlNcnLowLabelPerimColor
NhlNcnLowLabelPerimOn = _hlu.NhlNcnLowLabelPerimOn
NhlNcnLowLabelPerimSpaceF = _hlu.NhlNcnLowLabelPerimSpaceF
NhlNcnLowLabelPerimThicknessF = _hlu.NhlNcnLowLabelPerimThicknessF
NhlNcnLowLabelString = _hlu.NhlNcnLowLabelString
NhlNcnLowLabelsOn = _hlu.NhlNcnLowLabelsOn
NhlNcnLowUseHighLabelRes = _hlu.NhlNcnLowUseHighLabelRes
NhlNcnMaxDataValueFormat = _hlu.NhlNcnMaxDataValueFormat
NhlNcnMaxLevelCount = _hlu.NhlNcnMaxLevelCount
NhlNcnMaxLevelValF = _hlu.NhlNcnMaxLevelValF
NhlNcnMaxPointDistanceF = _hlu.NhlNcnMaxPointDistanceF
NhlNcnMinLevelValF = _hlu.NhlNcnMinLevelValF
NhlNcnMissingValFillColor = _hlu.NhlNcnMissingValFillColor
NhlNcnMissingValFillPattern = _hlu.NhlNcnMissingValFillPattern
NhlNcnMissingValFillScaleF = _hlu.NhlNcnMissingValFillScaleF
NhlNcnMissingValPerimColor = _hlu.NhlNcnMissingValPerimColor
NhlNcnMissingValPerimDashPattern = _hlu.NhlNcnMissingValPerimDashPattern
NhlNcnMissingValPerimGridBoundOn = _hlu.NhlNcnMissingValPerimGridBoundOn
NhlNcnMissingValPerimOn = _hlu.NhlNcnMissingValPerimOn
NhlNcnMissingValPerimThicknessF = _hlu.NhlNcnMissingValPerimThicknessF
NhlNcnMonoFillColor = _hlu.NhlNcnMonoFillColor
NhlNcnMonoFillPattern = _hlu.NhlNcnMonoFillPattern
NhlNcnMonoFillScale = _hlu.NhlNcnMonoFillScale
NhlNcnMonoLevelFlag = _hlu.NhlNcnMonoLevelFlag
NhlNcnMonoLineColor = _hlu.NhlNcnMonoLineColor
NhlNcnMonoLineDashPattern = _hlu.NhlNcnMonoLineDashPattern
NhlNcnMonoLineLabelFontColor = _hlu.NhlNcnMonoLineLabelFontColor
NhlNcnMonoLineThickness = _hlu.NhlNcnMonoLineThickness
NhlNcnNoDataLabelOn = _hlu.NhlNcnNoDataLabelOn
NhlNcnNoDataLabelString = _hlu.NhlNcnNoDataLabelString
NhlNcnOutOfRangePerimColor = _hlu.NhlNcnOutOfRangePerimColor
NhlNcnOutOfRangePerimDashPattern = _hlu.NhlNcnOutOfRangePerimDashPattern
NhlNcnOutOfRangePerimOn = _hlu.NhlNcnOutOfRangePerimOn
NhlNcnOutOfRangePerimThicknessF = _hlu.NhlNcnOutOfRangePerimThicknessF
NhlNcnRasterCellSizeF = _hlu.NhlNcnRasterCellSizeF
NhlNcnRasterMinCellSizeF = _hlu.NhlNcnRasterMinCellSizeF
NhlNcnRasterModeOn = _hlu.NhlNcnRasterModeOn
NhlNcnRasterSampleFactorF = _hlu.NhlNcnRasterSampleFactorF
NhlNcnRasterSmoothingOn = _hlu.NhlNcnRasterSmoothingOn
NhlNcnScalarFieldData = _hlu.NhlNcnScalarFieldData
NhlNcnSmoothingDistanceF = _hlu.NhlNcnSmoothingDistanceF
NhlNcnSmoothingOn = _hlu.NhlNcnSmoothingOn
NhlNcnSmoothingTensionF = _hlu.NhlNcnSmoothingTensionF
NhlNctCopyTables = _hlu.NhlNctCopyTables
NhlNctXElementSize = _hlu.NhlNctXElementSize
NhlNctXMaxV = _hlu.NhlNctXMaxV
NhlNctXMinV = _hlu.NhlNctXMinV
NhlNctXMissingV = _hlu.NhlNctXMissingV
NhlNctXTable = _hlu.NhlNctXTable
NhlNctXTableLengths = _hlu.NhlNctXTableLengths
NhlNctXTableType = _hlu.NhlNctXTableType
NhlNctYElementSize = _hlu.NhlNctYElementSize
NhlNctYMaxV = _hlu.NhlNctYMaxV
NhlNctYMinV = _hlu.NhlNctYMinV
NhlNctYMissingV = _hlu.NhlNctYMissingV
NhlNctYTable = _hlu.NhlNctYTable
NhlNctYTableLengths = _hlu.NhlNctYTableLengths
NhlNctYTableType = _hlu.NhlNctYTableType
NhlNdcDelayCompute = _hlu.NhlNdcDelayCompute
NhlNerrBuffer = _hlu.NhlNerrBuffer
NhlNerrFileName = _hlu.NhlNerrFileName
NhlNerrLevel = _hlu.NhlNerrLevel
NhlNerrPrint = _hlu.NhlNerrPrint
NhlNgsClipOn = _hlu.NhlNgsClipOn
NhlNgsEdgeColor = _hlu.NhlNgsEdgeColor
NhlNgsEdgeDashPattern = _hlu.NhlNgsEdgeDashPattern
NhlNgsEdgeDashSegLenF = _hlu.NhlNgsEdgeDashSegLenF
NhlNgsEdgeThicknessF = _hlu.NhlNgsEdgeThicknessF
NhlNgsEdgesOn = _hlu.NhlNgsEdgesOn
NhlNgsFillBackgroundColor = _hlu.NhlNgsFillBackgroundColor
NhlNgsFillColor = _hlu.NhlNgsFillColor
NhlNgsFillIndex = _hlu.NhlNgsFillIndex
NhlNgsFillLineThicknessF = _hlu.NhlNgsFillLineThicknessF
NhlNgsFillScaleF = _hlu.NhlNgsFillScaleF
NhlNgsFont = _hlu.NhlNgsFont
NhlNgsFontAspectF = _hlu.NhlNgsFontAspectF
NhlNgsFontColor = _hlu.NhlNgsFontColor
NhlNgsFontHeightF = _hlu.NhlNgsFontHeightF
NhlNgsFontQuality = _hlu.NhlNgsFontQuality
NhlNgsFontThicknessF = _hlu.NhlNgsFontThicknessF
NhlNgsLineColor = _hlu.NhlNgsLineColor
NhlNgsLineDashPattern = _hlu.NhlNgsLineDashPattern
NhlNgsLineDashSegLenF = _hlu.NhlNgsLineDashSegLenF
NhlNgsLineLabelConstantSpacingF = _hlu.NhlNgsLineLabelConstantSpacingF
NhlNgsLineLabelFont = _hlu.NhlNgsLineLabelFont
NhlNgsLineLabelFontAspectF = _hlu.NhlNgsLineLabelFontAspectF
NhlNgsLineLabelFontColor = _hlu.NhlNgsLineLabelFontColor
NhlNgsLineLabelFontHeightF = _hlu.NhlNgsLineLabelFontHeightF
NhlNgsLineLabelFontQuality = _hlu.NhlNgsLineLabelFontQuality
NhlNgsLineLabelFontThicknessF = _hlu.NhlNgsLineLabelFontThicknessF
NhlNgsLineLabelFuncCode = _hlu.NhlNgsLineLabelFuncCode
NhlNgsLineLabelString = _hlu.NhlNgsLineLabelString
NhlNgsLineThicknessF = _hlu.NhlNgsLineThicknessF
NhlNgsMarkerColor = _hlu.NhlNgsMarkerColor
NhlNgsMarkerIndex = _hlu.NhlNgsMarkerIndex
NhlNgsMarkerSizeF = _hlu.NhlNgsMarkerSizeF
NhlNgsMarkerThicknessF = _hlu.NhlNgsMarkerThicknessF
NhlNgsTextAngleF = _hlu.NhlNgsTextAngleF
NhlNgsTextConstantSpacingF = _hlu.NhlNgsTextConstantSpacingF
NhlNgsTextDirection = _hlu.NhlNgsTextDirection
NhlNgsTextFuncCode = _hlu.NhlNgsTextFuncCode
NhlNgsTextJustification = _hlu.NhlNgsTextJustification
NhlNlbAutoManage = _hlu.NhlNlbAutoManage
NhlNlbBottomMarginF = _hlu.NhlNlbBottomMarginF
NhlNlbBoxCount = _hlu.NhlNlbBoxCount
NhlNlbBoxFractions = _hlu.NhlNlbBoxFractions
NhlNlbBoxLineColor = _hlu.NhlNlbBoxLineColor
NhlNlbBoxLineDashPattern = _hlu.NhlNlbBoxLineDashPattern
NhlNlbBoxLineDashSegLenF = _hlu.NhlNlbBoxLineDashSegLenF
NhlNlbBoxLineThicknessF = _hlu.NhlNlbBoxLineThicknessF
NhlNlbBoxLinesOn = _hlu.NhlNlbBoxLinesOn
NhlNlbBoxMajorExtentF = _hlu.NhlNlbBoxMajorExtentF
NhlNlbBoxMinorExtentF = _hlu.NhlNlbBoxMinorExtentF
NhlNlbBoxSizing = _hlu.NhlNlbBoxSizing
NhlNlbFillBackground = _hlu.NhlNlbFillBackground
NhlNlbFillColor = _hlu.NhlNlbFillColor
NhlNlbFillColors = _hlu.NhlNlbFillColors
NhlNlbFillLineThicknessF = _hlu.NhlNlbFillLineThicknessF
NhlNlbFillPattern = _hlu.NhlNlbFillPattern
NhlNlbFillPatterns = _hlu.NhlNlbFillPatterns
NhlNlbFillScaleF = _hlu.NhlNlbFillScaleF
NhlNlbFillScales = _hlu.NhlNlbFillScales
NhlNlbJustification = _hlu.NhlNlbJustification
NhlNlbLabelAlignment = _hlu.NhlNlbLabelAlignment
NhlNlbLabelAngleF = _hlu.NhlNlbLabelAngleF
NhlNlbLabelAutoStride = _hlu.NhlNlbLabelAutoStride
NhlNlbLabelBarOn = _hlu.NhlNlbLabelBarOn
NhlNlbLabelConstantSpacingF = _hlu.NhlNlbLabelConstantSpacingF
NhlNlbLabelDirection = _hlu.NhlNlbLabelDirection
NhlNlbLabelFont = _hlu.NhlNlbLabelFont
NhlNlbLabelFontAspectF = _hlu.NhlNlbLabelFontAspectF
NhlNlbLabelFontColor = _hlu.NhlNlbLabelFontColor
NhlNlbLabelFontHeightF = _hlu.NhlNlbLabelFontHeightF
NhlNlbLabelFontQuality = _hlu.NhlNlbLabelFontQuality
NhlNlbLabelFontThicknessF = _hlu.NhlNlbLabelFontThicknessF
NhlNlbLabelFuncCode = _hlu.NhlNlbLabelFuncCode
NhlNlbLabelJust = _hlu.NhlNlbLabelJust
NhlNlbLabelOffsetF = _hlu.NhlNlbLabelOffsetF
NhlNlbLabelPosition = _hlu.NhlNlbLabelPosition
NhlNlbLabelStride = _hlu.NhlNlbLabelStride
NhlNlbLabelStrings = _hlu.NhlNlbLabelStrings
NhlNlbLabelsOn = _hlu.NhlNlbLabelsOn
NhlNlbLeftMarginF = _hlu.NhlNlbLeftMarginF
NhlNlbMaxLabelLenF = _hlu.NhlNlbMaxLabelLenF
NhlNlbMinLabelSpacingF = _hlu.NhlNlbMinLabelSpacingF
NhlNlbMonoFillColor = _hlu.NhlNlbMonoFillColor
NhlNlbMonoFillPattern = _hlu.NhlNlbMonoFillPattern
NhlNlbMonoFillScale = _hlu.NhlNlbMonoFillScale
NhlNlbOrientation = _hlu.NhlNlbOrientation
NhlNlbPerimColor = _hlu.NhlNlbPerimColor
NhlNlbPerimDashPattern = _hlu.NhlNlbPerimDashPattern
NhlNlbPerimDashSegLenF = _hlu.NhlNlbPerimDashSegLenF
NhlNlbPerimFill = _hlu.NhlNlbPerimFill
NhlNlbPerimFillColor = _hlu.NhlNlbPerimFillColor
NhlNlbPerimOn = _hlu.NhlNlbPerimOn
NhlNlbPerimThicknessF = _hlu.NhlNlbPerimThicknessF
NhlNlbRightMarginF = _hlu.NhlNlbRightMarginF
NhlNlbTitleAngleF = _hlu.NhlNlbTitleAngleF
NhlNlbTitleConstantSpacingF = _hlu.NhlNlbTitleConstantSpacingF
NhlNlbTitleDirection = _hlu.NhlNlbTitleDirection
NhlNlbTitleExtentF = _hlu.NhlNlbTitleExtentF
NhlNlbTitleFont = _hlu.NhlNlbTitleFont
NhlNlbTitleFontAspectF = _hlu.NhlNlbTitleFontAspectF
NhlNlbTitleFontColor = _hlu.NhlNlbTitleFontColor
NhlNlbTitleFontHeightF = _hlu.NhlNlbTitleFontHeightF
NhlNlbTitleFontQuality = _hlu.NhlNlbTitleFontQuality
NhlNlbTitleFontThicknessF = _hlu.NhlNlbTitleFontThicknessF
NhlNlbTitleFuncCode = _hlu.NhlNlbTitleFuncCode
NhlNlbTitleJust = _hlu.NhlNlbTitleJust
NhlNlbTitleOffsetF = _hlu.NhlNlbTitleOffsetF
NhlNlbTitleOn = _hlu.NhlNlbTitleOn
NhlNlbTitlePosition = _hlu.NhlNlbTitlePosition
NhlNlbTitleString = _hlu.NhlNlbTitleString
NhlNlbTopMarginF = _hlu.NhlNlbTopMarginF
NhlNlgAutoManage = _hlu.NhlNlgAutoManage
NhlNlgBottomMarginF = _hlu.NhlNlgBottomMarginF
NhlNlgBoxBackground = _hlu.NhlNlgBoxBackground
NhlNlgBoxLineColor = _hlu.NhlNlgBoxLineColor
NhlNlgBoxLineDashPattern = _hlu.NhlNlgBoxLineDashPattern
NhlNlgBoxLineDashSegLenF = _hlu.NhlNlgBoxLineDashSegLenF
NhlNlgBoxLineThicknessF = _hlu.NhlNlgBoxLineThicknessF
NhlNlgBoxLinesOn = _hlu.NhlNlgBoxLinesOn
NhlNlgBoxMajorExtentF = _hlu.NhlNlgBoxMajorExtentF
NhlNlgBoxMinorExtentF = _hlu.NhlNlgBoxMinorExtentF
NhlNlgDashIndex = _hlu.NhlNlgDashIndex
NhlNlgDashIndexes = _hlu.NhlNlgDashIndexes
NhlNlgItemCount = _hlu.NhlNlgItemCount
NhlNlgItemPlacement = _hlu.NhlNlgItemPlacement
NhlNlgItemPositions = _hlu.NhlNlgItemPositions
NhlNlgItemType = _hlu.NhlNlgItemType
NhlNlgItemTypes = _hlu.NhlNlgItemTypes
NhlNlgJustification = _hlu.NhlNlgJustification
NhlNlgLabelAlignment = _hlu.NhlNlgLabelAlignment
NhlNlgLabelAngleF = _hlu.NhlNlgLabelAngleF
NhlNlgLabelAutoStride = _hlu.NhlNlgLabelAutoStride
NhlNlgLabelConstantSpacingF = _hlu.NhlNlgLabelConstantSpacingF
NhlNlgLabelDirection = _hlu.NhlNlgLabelDirection
NhlNlgLabelFont = _hlu.NhlNlgLabelFont
NhlNlgLabelFontAspectF = _hlu.NhlNlgLabelFontAspectF
NhlNlgLabelFontColor = _hlu.NhlNlgLabelFontColor
NhlNlgLabelFontHeightF = _hlu.NhlNlgLabelFontHeightF
NhlNlgLabelFontQuality = _hlu.NhlNlgLabelFontQuality
NhlNlgLabelFontThicknessF = _hlu.NhlNlgLabelFontThicknessF
NhlNlgLabelFuncCode = _hlu.NhlNlgLabelFuncCode
NhlNlgLabelJust = _hlu.NhlNlgLabelJust
NhlNlgLabelOffsetF = _hlu.NhlNlgLabelOffsetF
NhlNlgLabelPosition = _hlu.NhlNlgLabelPosition
NhlNlgLabelStride = _hlu.NhlNlgLabelStride
NhlNlgLabelStrings = _hlu.NhlNlgLabelStrings
NhlNlgLabelsOn = _hlu.NhlNlgLabelsOn
NhlNlgLeftMarginF = _hlu.NhlNlgLeftMarginF
NhlNlgLegendOn = _hlu.NhlNlgLegendOn
NhlNlgLineColor = _hlu.NhlNlgLineColor
NhlNlgLineColors = _hlu.NhlNlgLineColors
NhlNlgLineDashSegLenF = _hlu.NhlNlgLineDashSegLenF
NhlNlgLineDashSegLens = _hlu.NhlNlgLineDashSegLens
NhlNlgLineLabelConstantSpacingF = _hlu.NhlNlgLineLabelConstantSpacingF
NhlNlgLineLabelFont = _hlu.NhlNlgLineLabelFont
NhlNlgLineLabelFontAspectF = _hlu.NhlNlgLineLabelFontAspectF
NhlNlgLineLabelFontColor = _hlu.NhlNlgLineLabelFontColor
NhlNlgLineLabelFontColors = _hlu.NhlNlgLineLabelFontColors
NhlNlgLineLabelFontHeightF = _hlu.NhlNlgLineLabelFontHeightF
NhlNlgLineLabelFontHeights = _hlu.NhlNlgLineLabelFontHeights
NhlNlgLineLabelFontQuality = _hlu.NhlNlgLineLabelFontQuality
NhlNlgLineLabelFontThicknessF = _hlu.NhlNlgLineLabelFontThicknessF
NhlNlgLineLabelFuncCode = _hlu.NhlNlgLineLabelFuncCode
NhlNlgLineLabelStrings = _hlu.NhlNlgLineLabelStrings
NhlNlgLineLabelsOn = _hlu.NhlNlgLineLabelsOn
NhlNlgLineThicknessF = _hlu.NhlNlgLineThicknessF
NhlNlgLineThicknesses = _hlu.NhlNlgLineThicknesses
NhlNlgMarkerColor = _hlu.NhlNlgMarkerColor
NhlNlgMarkerColors = _hlu.NhlNlgMarkerColors
NhlNlgMarkerIndex = _hlu.NhlNlgMarkerIndex
NhlNlgMarkerIndexes = _hlu.NhlNlgMarkerIndexes
NhlNlgMarkerSizeF = _hlu.NhlNlgMarkerSizeF
NhlNlgMarkerSizes = _hlu.NhlNlgMarkerSizes
NhlNlgMarkerThicknessF = _hlu.NhlNlgMarkerThicknessF
NhlNlgMarkerThicknesses = _hlu.NhlNlgMarkerThicknesses
NhlNlgMonoDashIndex = _hlu.NhlNlgMonoDashIndex
NhlNlgMonoItemType = _hlu.NhlNlgMonoItemType
NhlNlgMonoLineColor = _hlu.NhlNlgMonoLineColor
NhlNlgMonoLineDashSegLen = _hlu.NhlNlgMonoLineDashSegLen
NhlNlgMonoLineLabelFontColor = _hlu.NhlNlgMonoLineLabelFontColor
NhlNlgMonoLineLabelFontHeight = _hlu.NhlNlgMonoLineLabelFontHeight
NhlNlgMonoLineThickness = _hlu.NhlNlgMonoLineThickness
NhlNlgMonoMarkerColor = _hlu.NhlNlgMonoMarkerColor
NhlNlgMonoMarkerIndex = _hlu.NhlNlgMonoMarkerIndex
NhlNlgMonoMarkerSize = _hlu.NhlNlgMonoMarkerSize
NhlNlgMonoMarkerThickness = _hlu.NhlNlgMonoMarkerThickness
NhlNlgOrientation = _hlu.NhlNlgOrientation
NhlNlgPerimColor = _hlu.NhlNlgPerimColor
NhlNlgPerimDashPattern = _hlu.NhlNlgPerimDashPattern
NhlNlgPerimDashSegLenF = _hlu.NhlNlgPerimDashSegLenF
NhlNlgPerimFill = _hlu.NhlNlgPerimFill
NhlNlgPerimFillColor = _hlu.NhlNlgPerimFillColor
NhlNlgPerimOn = _hlu.NhlNlgPerimOn
NhlNlgPerimThicknessF = _hlu.NhlNlgPerimThicknessF
NhlNlgRightMarginF = _hlu.NhlNlgRightMarginF
NhlNlgTitleAngleF = _hlu.NhlNlgTitleAngleF
NhlNlgTitleConstantSpacingF = _hlu.NhlNlgTitleConstantSpacingF
NhlNlgTitleDirection = _hlu.NhlNlgTitleDirection
NhlNlgTitleExtentF = _hlu.NhlNlgTitleExtentF
NhlNlgTitleFont = _hlu.NhlNlgTitleFont
NhlNlgTitleFontAspectF = _hlu.NhlNlgTitleFontAspectF
NhlNlgTitleFontColor = _hlu.NhlNlgTitleFontColor
NhlNlgTitleFontHeightF = _hlu.NhlNlgTitleFontHeightF
NhlNlgTitleFontQuality = _hlu.NhlNlgTitleFontQuality
NhlNlgTitleFontThicknessF = _hlu.NhlNlgTitleFontThicknessF
NhlNlgTitleFuncCode = _hlu.NhlNlgTitleFuncCode
NhlNlgTitleJust = _hlu.NhlNlgTitleJust
NhlNlgTitleOffsetF = _hlu.NhlNlgTitleOffsetF
NhlNlgTitleOn = _hlu.NhlNlgTitleOn
NhlNlgTitlePosition = _hlu.NhlNlgTitlePosition
NhlNlgTitleString = _hlu.NhlNlgTitleString
NhlNlgTopMarginF = _hlu.NhlNlgTopMarginF
NhlNmpAreaGroupCount = _hlu.NhlNmpAreaGroupCount
NhlNmpAreaMaskingOn = _hlu.NhlNmpAreaMaskingOn
NhlNmpAreaNames = _hlu.NhlNmpAreaNames
NhlNmpAreaTypes = _hlu.NhlNmpAreaTypes
NhlNmpDataBaseVersion = _hlu.NhlNmpDataBaseVersion
NhlNmpDataResolution = _hlu.NhlNmpDataResolution
NhlNmpDataSetName = _hlu.NhlNmpDataSetName
NhlNmpDefaultFillColor = _hlu.NhlNmpDefaultFillColor
NhlNmpDefaultFillPattern = _hlu.NhlNmpDefaultFillPattern
NhlNmpDefaultFillScaleF = _hlu.NhlNmpDefaultFillScaleF
NhlNmpDynamicAreaGroups = _hlu.NhlNmpDynamicAreaGroups
NhlNmpFillAreaSpecifiers = _hlu.NhlNmpFillAreaSpecifiers
NhlNmpFillBoundarySets = _hlu.NhlNmpFillBoundarySets
NhlNmpFillColor = _hlu.NhlNmpFillColor
NhlNmpFillColors = _hlu.NhlNmpFillColors
NhlNmpFillDrawOrder = _hlu.NhlNmpFillDrawOrder
NhlNmpFillOn = _hlu.NhlNmpFillOn
NhlNmpFillPattern = _hlu.NhlNmpFillPattern
NhlNmpFillPatternBackground = _hlu.NhlNmpFillPatternBackground
NhlNmpFillPatterns = _hlu.NhlNmpFillPatterns
NhlNmpFillScaleF = _hlu.NhlNmpFillScaleF
NhlNmpFillScales = _hlu.NhlNmpFillScales
NhlNmpFixedAreaGroups = _hlu.NhlNmpFixedAreaGroups
NhlNmpGeophysicalLineColor = _hlu.NhlNmpGeophysicalLineColor
NhlNmpGeophysicalLineDashPattern = _hlu.NhlNmpGeophysicalLineDashPattern
NhlNmpGeophysicalLineDashSegLenF = _hlu.NhlNmpGeophysicalLineDashSegLenF
NhlNmpGeophysicalLineThicknessF = _hlu.NhlNmpGeophysicalLineThicknessF
NhlNmpGridAndLimbDrawOrder = _hlu.NhlNmpGridAndLimbDrawOrder
NhlNmpGridAndLimbOn = _hlu.NhlNmpGridAndLimbOn
NhlNmpGridLatSpacingF = _hlu.NhlNmpGridLatSpacingF
NhlNmpGridLineColor = _hlu.NhlNmpGridLineColor
NhlNmpGridLineDashPattern = _hlu.NhlNmpGridLineDashPattern
NhlNmpGridLineDashSegLenF = _hlu.NhlNmpGridLineDashSegLenF
NhlNmpGridLineThicknessF = _hlu.NhlNmpGridLineThicknessF
NhlNmpGridLonSpacingF = _hlu.NhlNmpGridLonSpacingF
NhlNmpGridMaskMode = _hlu.NhlNmpGridMaskMode
NhlNmpGridMaxLatF = _hlu.NhlNmpGridMaxLatF
NhlNmpGridPolarLonSpacingF = _hlu.NhlNmpGridPolarLonSpacingF
NhlNmpGridSpacingF = _hlu.NhlNmpGridSpacingF
NhlNmpInlandWaterFillColor = _hlu.NhlNmpInlandWaterFillColor
NhlNmpInlandWaterFillPattern = _hlu.NhlNmpInlandWaterFillPattern
NhlNmpInlandWaterFillScaleF = _hlu.NhlNmpInlandWaterFillScaleF
NhlNmpLabelDrawOrder = _hlu.NhlNmpLabelDrawOrder
NhlNmpLabelFontColor = _hlu.NhlNmpLabelFontColor
NhlNmpLabelFontHeightF = _hlu.NhlNmpLabelFontHeightF
NhlNmpLabelsOn = _hlu.NhlNmpLabelsOn
NhlNmpLandFillColor = _hlu.NhlNmpLandFillColor
NhlNmpLandFillPattern = _hlu.NhlNmpLandFillPattern
NhlNmpLandFillScaleF = _hlu.NhlNmpLandFillScaleF
NhlNmpLimbLineColor = _hlu.NhlNmpLimbLineColor
NhlNmpLimbLineDashPattern = _hlu.NhlNmpLimbLineDashPattern
NhlNmpLimbLineDashSegLenF = _hlu.NhlNmpLimbLineDashSegLenF
NhlNmpLimbLineThicknessF = _hlu.NhlNmpLimbLineThicknessF
NhlNmpMaskAreaSpecifiers = _hlu.NhlNmpMaskAreaSpecifiers
NhlNmpMonoFillColor = _hlu.NhlNmpMonoFillColor
NhlNmpMonoFillPattern = _hlu.NhlNmpMonoFillPattern
NhlNmpMonoFillScale = _hlu.NhlNmpMonoFillScale
NhlNmpNationalLineColor = _hlu.NhlNmpNationalLineColor
NhlNmpNationalLineDashPattern = _hlu.NhlNmpNationalLineDashPattern
NhlNmpNationalLineDashSegLenF = _hlu.NhlNmpNationalLineDashSegLenF
NhlNmpNationalLineThicknessF = _hlu.NhlNmpNationalLineThicknessF
NhlNmpOceanFillColor = _hlu.NhlNmpOceanFillColor
NhlNmpOceanFillPattern = _hlu.NhlNmpOceanFillPattern
NhlNmpOceanFillScaleF = _hlu.NhlNmpOceanFillScaleF
NhlNmpOutlineBoundarySets = _hlu.NhlNmpOutlineBoundarySets
NhlNmpOutlineDrawOrder = _hlu.NhlNmpOutlineDrawOrder
NhlNmpOutlineOn = _hlu.NhlNmpOutlineOn
NhlNmpOutlineSpecifiers = _hlu.NhlNmpOutlineSpecifiers
NhlNmpPerimDrawOrder = _hlu.NhlNmpPerimDrawOrder
NhlNmpPerimLineColor = _hlu.NhlNmpPerimLineColor
NhlNmpPerimLineDashPattern = _hlu.NhlNmpPerimLineDashPattern
NhlNmpPerimLineDashSegLenF = _hlu.NhlNmpPerimLineDashSegLenF
NhlNmpPerimLineThicknessF = _hlu.NhlNmpPerimLineThicknessF
NhlNmpPerimOn = _hlu.NhlNmpPerimOn
NhlNmpShapeMode = _hlu.NhlNmpShapeMode
NhlNmpSpecifiedFillColors = _hlu.NhlNmpSpecifiedFillColors
NhlNmpSpecifiedFillDirectIndexing = _hlu.NhlNmpSpecifiedFillDirectIndexing
NhlNmpSpecifiedFillPatterns = _hlu.NhlNmpSpecifiedFillPatterns
NhlNmpSpecifiedFillPriority = _hlu.NhlNmpSpecifiedFillPriority
NhlNmpSpecifiedFillScales = _hlu.NhlNmpSpecifiedFillScales
NhlNmpUSStateLineColor = _hlu.NhlNmpUSStateLineColor
NhlNmpUSStateLineDashPattern = _hlu.NhlNmpUSStateLineDashPattern
NhlNmpUSStateLineDashSegLenF = _hlu.NhlNmpUSStateLineDashSegLenF
NhlNmpUSStateLineThicknessF = _hlu.NhlNmpUSStateLineThicknessF
NhlNmpBottomAngleF = _hlu.NhlNmpBottomAngleF
NhlNmpBottomMapPosF = _hlu.NhlNmpBottomMapPosF
NhlNmpBottomNDCF = _hlu.NhlNmpBottomNDCF
NhlNmpBottomNPCF = _hlu.NhlNmpBottomNPCF
NhlNmpBottomPointLatF = _hlu.NhlNmpBottomPointLatF
NhlNmpBottomPointLonF = _hlu.NhlNmpBottomPointLonF
NhlNmpBottomWindowF = _hlu.NhlNmpBottomWindowF
NhlNmpCenterLatF = _hlu.NhlNmpCenterLatF
NhlNmpCenterLonF = _hlu.NhlNmpCenterLonF
NhlNmpCenterRotF = _hlu.NhlNmpCenterRotF
NhlNmpEllipticalBoundary = _hlu.NhlNmpEllipticalBoundary
NhlNmpGreatCircleLinesOn = _hlu.NhlNmpGreatCircleLinesOn
NhlNmpLambertMeridianF = _hlu.NhlNmpLambertMeridianF
NhlNmpLambertParallel1F = _hlu.NhlNmpLambertParallel1F
NhlNmpLambertParallel2F = _hlu.NhlNmpLambertParallel2F
NhlNmpLeftAngleF = _hlu.NhlNmpLeftAngleF
NhlNmpLeftCornerLatF = _hlu.NhlNmpLeftCornerLatF
NhlNmpLeftCornerLonF = _hlu.NhlNmpLeftCornerLonF
NhlNmpLeftMapPosF = _hlu.NhlNmpLeftMapPosF
NhlNmpLeftNDCF = _hlu.NhlNmpLeftNDCF
NhlNmpLeftNPCF = _hlu.NhlNmpLeftNPCF
NhlNmpLeftPointLatF = _hlu.NhlNmpLeftPointLatF
NhlNmpLeftPointLonF = _hlu.NhlNmpLeftPointLonF
NhlNmpLeftWindowF = _hlu.NhlNmpLeftWindowF
NhlNmpLimitMode = _hlu.NhlNmpLimitMode
NhlNmpMaxLatF = _hlu.NhlNmpMaxLatF
NhlNmpMaxLonF = _hlu.NhlNmpMaxLonF
NhlNmpMinLatF = _hlu.NhlNmpMinLatF
NhlNmpMinLonF = _hlu.NhlNmpMinLonF
NhlNmpProjection = _hlu.NhlNmpProjection
NhlNmpRelativeCenterLat = _hlu.NhlNmpRelativeCenterLat
NhlNmpRelativeCenterLon = _hlu.NhlNmpRelativeCenterLon
NhlNmpRightAngleF = _hlu.NhlNmpRightAngleF
NhlNmpRightCornerLatF = _hlu.NhlNmpRightCornerLatF
NhlNmpRightCornerLonF = _hlu.NhlNmpRightCornerLonF
NhlNmpRightMapPosF = _hlu.NhlNmpRightMapPosF
NhlNmpRightNDCF = _hlu.NhlNmpRightNDCF
NhlNmpRightNPCF = _hlu.NhlNmpRightNPCF
NhlNmpRightPointLatF = _hlu.NhlNmpRightPointLatF
NhlNmpRightPointLonF = _hlu.NhlNmpRightPointLonF
NhlNmpRightWindowF = _hlu.NhlNmpRightWindowF
NhlNmpSatelliteAngle1F = _hlu.NhlNmpSatelliteAngle1F
NhlNmpSatelliteAngle2F = _hlu.NhlNmpSatelliteAngle2F
NhlNmpSatelliteDistF = _hlu.NhlNmpSatelliteDistF
NhlNmpTopAngleF = _hlu.NhlNmpTopAngleF
NhlNmpTopMapPosF = _hlu.NhlNmpTopMapPosF
NhlNmpTopNDCF = _hlu.NhlNmpTopNDCF
NhlNmpTopNPCF = _hlu.NhlNmpTopNPCF
NhlNmpTopPointLatF = _hlu.NhlNmpTopPointLatF
NhlNmpTopPointLonF = _hlu.NhlNmpTopPointLonF
NhlNmpTopWindowF = _hlu.NhlNmpTopWindowF
NhlNpmAnnoManagers = _hlu.NhlNpmAnnoManagers
NhlNpmAnnoViews = _hlu.NhlNpmAnnoViews
NhlNpmLabelBarDisplayMode = _hlu.NhlNpmLabelBarDisplayMode
NhlNpmLabelBarHeightF = _hlu.NhlNpmLabelBarHeightF
NhlNpmLabelBarKeepAspect = _hlu.NhlNpmLabelBarKeepAspect
NhlNpmLabelBarOrthogonalPosF = _hlu.NhlNpmLabelBarOrthogonalPosF
NhlNpmLabelBarParallelPosF = _hlu.NhlNpmLabelBarParallelPosF
NhlNpmLabelBarSide = _hlu.NhlNpmLabelBarSide
NhlNpmLabelBarWidthF = _hlu.NhlNpmLabelBarWidthF
NhlNpmLabelBarZone = _hlu.NhlNpmLabelBarZone
NhlNpmLegendDisplayMode = _hlu.NhlNpmLegendDisplayMode
NhlNpmLegendHeightF = _hlu.NhlNpmLegendHeightF
NhlNpmLegendKeepAspect = _hlu.NhlNpmLegendKeepAspect
NhlNpmLegendOrthogonalPosF = _hlu.NhlNpmLegendOrthogonalPosF
NhlNpmLegendParallelPosF = _hlu.NhlNpmLegendParallelPosF
NhlNpmLegendSide = _hlu.NhlNpmLegendSide
NhlNpmLegendWidthF = _hlu.NhlNpmLegendWidthF
NhlNpmLegendZone = _hlu.NhlNpmLegendZone
NhlNpmOverlaySequenceIds = _hlu.NhlNpmOverlaySequenceIds
NhlNpmTickMarkDisplayMode = _hlu.NhlNpmTickMarkDisplayMode
NhlNpmTickMarkZone = _hlu.NhlNpmTickMarkZone
NhlNpmTitleDisplayMode = _hlu.NhlNpmTitleDisplayMode
NhlNpmTitleZone = _hlu.NhlNpmTitleZone
NhlNprGraphicStyle = _hlu.NhlNprGraphicStyle
NhlNprPolyType = _hlu.NhlNprPolyType
NhlNprXArray = _hlu.NhlNprXArray
NhlNprYArray = _hlu.NhlNprYArray
NhlNsfCopyData = _hlu.NhlNsfCopyData
NhlNsfDataArray = _hlu.NhlNsfDataArray
NhlNsfDataMaxV = _hlu.NhlNsfDataMaxV
NhlNsfDataMinV = _hlu.NhlNsfDataMinV
NhlNsfExchangeDimensions = _hlu.NhlNsfExchangeDimensions
NhlNsfMissingValueV = _hlu.NhlNsfMissingValueV
NhlNsfXArray = _hlu.NhlNsfXArray
NhlNsfXCActualEndF = _hlu.NhlNsfXCActualEndF
NhlNsfXCActualStartF = _hlu.NhlNsfXCActualStartF
NhlNsfXCEndIndex = _hlu.NhlNsfXCEndIndex
NhlNsfXCEndSubsetV = _hlu.NhlNsfXCEndSubsetV
NhlNsfXCEndV = _hlu.NhlNsfXCEndV
NhlNsfXCStartIndex = _hlu.NhlNsfXCStartIndex
NhlNsfXCStartSubsetV = _hlu.NhlNsfXCStartSubsetV
NhlNsfXCStartV = _hlu.NhlNsfXCStartV
NhlNsfXCStride = _hlu.NhlNsfXCStride
NhlNsfYArray = _hlu.NhlNsfYArray
NhlNsfYCActualEndF = _hlu.NhlNsfYCActualEndF
NhlNsfYCActualStartF = _hlu.NhlNsfYCActualStartF
NhlNsfYCEndIndex = _hlu.NhlNsfYCEndIndex
NhlNsfYCEndSubsetV = _hlu.NhlNsfYCEndSubsetV
NhlNsfYCEndV = _hlu.NhlNsfYCEndV
NhlNsfYCStartIndex = _hlu.NhlNsfYCStartIndex
NhlNsfYCStartSubsetV = _hlu.NhlNsfYCStartSubsetV
NhlNsfYCStartV = _hlu.NhlNsfYCStartV
NhlNsfYCStride = _hlu.NhlNsfYCStride
NhlNstArrowLengthF = _hlu.NhlNstArrowLengthF
NhlNstArrowStride = _hlu.NhlNstArrowStride
NhlNstCrossoverCheckCount = _hlu.NhlNstCrossoverCheckCount
NhlNstExplicitLabelBarLabelsOn = _hlu.NhlNstExplicitLabelBarLabelsOn
NhlNstLabelBarEndLabelsOn = _hlu.NhlNstLabelBarEndLabelsOn
NhlNstLabelFormat = _hlu.NhlNstLabelFormat
NhlNstLengthCheckCount = _hlu.NhlNstLengthCheckCount
NhlNstLevelColors = _hlu.NhlNstLevelColors
NhlNstLevelCount = _hlu.NhlNstLevelCount
NhlNstLevelSelectionMode = _hlu.NhlNstLevelSelectionMode
NhlNstLevelSpacingF = _hlu.NhlNstLevelSpacingF
NhlNstLevels = _hlu.NhlNstLevels
NhlNstLineColor = _hlu.NhlNstLineColor
NhlNstLineStartStride = _hlu.NhlNstLineStartStride
NhlNstLineThicknessF = _hlu.NhlNstLineThicknessF
NhlNstMapDirection = _hlu.NhlNstMapDirection
NhlNstMaxLevelCount = _hlu.NhlNstMaxLevelCount
NhlNstMaxLevelValF = _hlu.NhlNstMaxLevelValF
NhlNstMinArrowSpacingF = _hlu.NhlNstMinArrowSpacingF
NhlNstMinDistanceF = _hlu.NhlNstMinDistanceF
NhlNstMinLevelValF = _hlu.NhlNstMinLevelValF
NhlNstMinLineSpacingF = _hlu.NhlNstMinLineSpacingF
NhlNstMinStepFactorF = _hlu.NhlNstMinStepFactorF
NhlNstMonoLineColor = _hlu.NhlNstMonoLineColor
NhlNstNoDataLabelOn = _hlu.NhlNstNoDataLabelOn
NhlNstNoDataLabelString = _hlu.NhlNstNoDataLabelString
NhlNstScalarFieldData = _hlu.NhlNstScalarFieldData
NhlNstScalarMissingValColor = _hlu.NhlNstScalarMissingValColor
NhlNstStepSizeF = _hlu.NhlNstStepSizeF
NhlNstStreamlineDrawOrder = _hlu.NhlNstStreamlineDrawOrder
NhlNstUseScalarArray = _hlu.NhlNstUseScalarArray
NhlNstVectorFieldData = _hlu.NhlNstVectorFieldData
NhlNvfDataArray = _hlu.NhlNvfDataArray
NhlNvfUDataArray = _hlu.NhlNvfUDataArray
NhlNvfVDataArray = _hlu.NhlNvfVDataArray
NhlNvfXArray = _hlu.NhlNvfXArray
NhlNvfYArray = _hlu.NhlNvfYArray
NhlNvfGridType = _hlu.NhlNvfGridType
NhlNvfPolarData = _hlu.NhlNvfPolarData
NhlNvfSubsetByIndex = _hlu.NhlNvfSubsetByIndex
NhlNvfCopyData = _hlu.NhlNvfCopyData
NhlNvfExchangeDimensions = _hlu.NhlNvfExchangeDimensions
NhlNvfExchangeUVData = _hlu.NhlNvfExchangeUVData
NhlNvfSingleMissingValue = _hlu.NhlNvfSingleMissingValue
NhlNvfMissingUValueV = _hlu.NhlNvfMissingUValueV
NhlNvfMissingVValueV = _hlu.NhlNvfMissingVValueV
NhlNvfMagMinV = _hlu.NhlNvfMagMinV
NhlNvfMagMaxV = _hlu.NhlNvfMagMaxV
NhlNvfUMinV = _hlu.NhlNvfUMinV
NhlNvfUMaxV = _hlu.NhlNvfUMaxV
NhlNvfVMinV = _hlu.NhlNvfVMinV
NhlNvfVMaxV = _hlu.NhlNvfVMaxV
NhlNvfXCStartV = _hlu.NhlNvfXCStartV
NhlNvfXCEndV = _hlu.NhlNvfXCEndV
NhlNvfYCStartV = _hlu.NhlNvfYCStartV
NhlNvfYCEndV = _hlu.NhlNvfYCEndV
NhlNvfXCStartSubsetV = _hlu.NhlNvfXCStartSubsetV
NhlNvfXCEndSubsetV = _hlu.NhlNvfXCEndSubsetV
NhlNvfYCStartSubset = _hlu.NhlNvfYCStartSubset
NhlNvfYCEndSubsetV = _hlu.NhlNvfYCEndSubsetV
NhlNvfXCStartIndex = _hlu.NhlNvfXCStartIndex
NhlNvfXCEndIndex = _hlu.NhlNvfXCEndIndex
NhlNvfYCStartIndex = _hlu.NhlNvfYCStartIndex
NhlNvfYCEndIndex = _hlu.NhlNvfYCEndIndex
NhlNvfXCStride = _hlu.NhlNvfXCStride
NhlNvfYCStride = _hlu.NhlNvfYCStride
NhlNvfXCActualStartF = _hlu.NhlNvfXCActualStartF
NhlNvfXCActualEndF = _hlu.NhlNvfXCActualEndF
NhlNvfXCElementCount = _hlu.NhlNvfXCElementCount
NhlNvfYCActualStartF = _hlu.NhlNvfYCActualStartF
NhlNvfYCActualEndF = _hlu.NhlNvfYCActualEndF
NhlNvfYCElementCount = _hlu.NhlNvfYCElementCount
NhlCvfDataArray = _hlu.NhlCvfDataArray
NhlCvfUDataArray = _hlu.NhlCvfUDataArray
NhlCvfVDataArray = _hlu.NhlCvfVDataArray
NhlCvfXArray = _hlu.NhlCvfXArray
NhlCvfYArray = _hlu.NhlCvfYArray
NhlCvfGridType = _hlu.NhlCvfGridType
NhlCvfPolarData = _hlu.NhlCvfPolarData
NhlCvfSubsetByIndex = _hlu.NhlCvfSubsetByIndex
NhlCvfCopyData = _hlu.NhlCvfCopyData
NhlCvfExchangeDimensions = _hlu.NhlCvfExchangeDimensions
NhlCvfExchangeUVData = _hlu.NhlCvfExchangeUVData
NhlCvfSingleMissingValue = _hlu.NhlCvfSingleMissingValue
NhlCvfMissingUValueV = _hlu.NhlCvfMissingUValueV
NhlCvfMissingVValueV = _hlu.NhlCvfMissingVValueV
NhlCvfMagMinV = _hlu.NhlCvfMagMinV
NhlCvfMagMaxV = _hlu.NhlCvfMagMaxV
NhlCvfUMinV = _hlu.NhlCvfUMinV
NhlCvfUMaxV = _hlu.NhlCvfUMaxV
NhlCvfVMinV = _hlu.NhlCvfVMinV
NhlCvfVMaxV = _hlu.NhlCvfVMaxV
NhlCvfXCStartV = _hlu.NhlCvfXCStartV
NhlCvfXCEndV = _hlu.NhlCvfXCEndV
NhlCvfYCStartV = _hlu.NhlCvfYCStartV
NhlCvfYCEndV = _hlu.NhlCvfYCEndV
NhlCvfXCStartSubsetV = _hlu.NhlCvfXCStartSubsetV
NhlCvfXCEndSubsetV = _hlu.NhlCvfXCEndSubsetV
NhlCvfYCStartSubsetV = _hlu.NhlCvfYCStartSubsetV
NhlCvfYCEndSubsetV = _hlu.NhlCvfYCEndSubsetV
NhlCvfXCStartIndex = _hlu.NhlCvfXCStartIndex
NhlCvfXCEndIndex = _hlu.NhlCvfXCEndIndex
NhlCvfYCStartIndex = _hlu.NhlCvfYCStartIndex
NhlCvfYCEndIndex = _hlu.NhlCvfYCEndIndex
NhlCvfXCStride = _hlu.NhlCvfXCStride
NhlCvfYCStride = _hlu.NhlCvfYCStride
NhlCvfXCActualStartF = _hlu.NhlCvfXCActualStartF
NhlCvfXCActualEndF = _hlu.NhlCvfXCActualEndF
NhlCvfXCElementCount = _hlu.NhlCvfXCElementCount
NhlCvfYCActualStartF = _hlu.NhlCvfYCActualStartF
NhlCvfYCActualEndF = _hlu.NhlCvfYCActualEndF
NhlCvfYCElementCount = _hlu.NhlCvfYCElementCount
NhlNstZeroFLabelAngleF = _hlu.NhlNstZeroFLabelAngleF
NhlNstZeroFLabelBackgroundColor = _hlu.NhlNstZeroFLabelBackgroundColor
NhlNstZeroFLabelConstantSpacingF = _hlu.NhlNstZeroFLabelConstantSpacingF
NhlNstZeroFLabelFont = _hlu.NhlNstZeroFLabelFont
NhlNstZeroFLabelFontAspectF = _hlu.NhlNstZeroFLabelFontAspectF
NhlNstZeroFLabelFontColor = _hlu.NhlNstZeroFLabelFontColor
NhlNstZeroFLabelFontHeightF = _hlu.NhlNstZeroFLabelFontHeightF
NhlNstZeroFLabelFontQuality = _hlu.NhlNstZeroFLabelFontQuality
NhlNstZeroFLabelFontThicknessF = _hlu.NhlNstZeroFLabelFontThicknessF
NhlNstZeroFLabelFuncCode = _hlu.NhlNstZeroFLabelFuncCode
NhlNstZeroFLabelJust = _hlu.NhlNstZeroFLabelJust
NhlNstZeroFLabelOn = _hlu.NhlNstZeroFLabelOn
NhlNstZeroFLabelOrthogonalPosF = _hlu.NhlNstZeroFLabelOrthogonalPosF
NhlNstZeroFLabelParallelPosF = _hlu.NhlNstZeroFLabelParallelPosF
NhlNstZeroFLabelPerimColor = _hlu.NhlNstZeroFLabelPerimColor
NhlNstZeroFLabelPerimOn = _hlu.NhlNstZeroFLabelPerimOn
NhlNstZeroFLabelPerimSpaceF = _hlu.NhlNstZeroFLabelPerimSpaceF
NhlNstZeroFLabelPerimThicknessF = _hlu.NhlNstZeroFLabelPerimThicknessF
NhlNstZeroFLabelSide = _hlu.NhlNstZeroFLabelSide
NhlNstZeroFLabelString = _hlu.NhlNstZeroFLabelString
NhlNstZeroFLabelTextDirection = _hlu.NhlNstZeroFLabelTextDirection
NhlNstZeroFLabelZone = _hlu.NhlNstZeroFLabelZone
NhlNtfDoNDCOverlay = _hlu.NhlNtfDoNDCOverlay
NhlNtfPlotManagerOn = _hlu.NhlNtfPlotManagerOn
NhlNtfPolyDrawList = _hlu.NhlNtfPolyDrawList
NhlNtfPolyDrawOrder = _hlu.NhlNtfPolyDrawOrder
NhlNtiDeltaF = _hlu.NhlNtiDeltaF
NhlNtiMainAngleF = _hlu.NhlNtiMainAngleF
NhlNtiMainConstantSpacingF = _hlu.NhlNtiMainConstantSpacingF
NhlNtiMainDirection = _hlu.NhlNtiMainDirection
NhlNtiMainFont = _hlu.NhlNtiMainFont
NhlNtiMainFontAspectF = _hlu.NhlNtiMainFontAspectF
NhlNtiMainFontColor = _hlu.NhlNtiMainFontColor
NhlNtiMainFontHeightF = _hlu.NhlNtiMainFontHeightF
NhlNtiMainFontQuality = _hlu.NhlNtiMainFontQuality
NhlNtiMainFontThicknessF = _hlu.NhlNtiMainFontThicknessF
NhlNtiMainFuncCode = _hlu.NhlNtiMainFuncCode
NhlNtiMainJust = _hlu.NhlNtiMainJust
NhlNtiMainOffsetXF = _hlu.NhlNtiMainOffsetXF
NhlNtiMainOffsetYF = _hlu.NhlNtiMainOffsetYF
NhlNtiMainOn = _hlu.NhlNtiMainOn
NhlNtiMainPosition = _hlu.NhlNtiMainPosition
NhlNtiMainSide = _hlu.NhlNtiMainSide
NhlNtiMainString = _hlu.NhlNtiMainString
NhlNtiUseMainAttributes = _hlu.NhlNtiUseMainAttributes
NhlNtiXAxisAngleF = _hlu.NhlNtiXAxisAngleF
NhlNtiXAxisConstantSpacingF = _hlu.NhlNtiXAxisConstantSpacingF
NhlNtiXAxisDirection = _hlu.NhlNtiXAxisDirection
NhlNtiXAxisFont = _hlu.NhlNtiXAxisFont
NhlNtiXAxisFontAspectF = _hlu.NhlNtiXAxisFontAspectF
NhlNtiXAxisFontColor = _hlu.NhlNtiXAxisFontColor
NhlNtiXAxisFontHeightF = _hlu.NhlNtiXAxisFontHeightF
NhlNtiXAxisFontQuality = _hlu.NhlNtiXAxisFontQuality
NhlNtiXAxisFontThicknessF = _hlu.NhlNtiXAxisFontThicknessF
NhlNtiXAxisFuncCode = _hlu.NhlNtiXAxisFuncCode
NhlNtiXAxisJust = _hlu.NhlNtiXAxisJust
NhlNtiXAxisOffsetXF = _hlu.NhlNtiXAxisOffsetXF
NhlNtiXAxisOffsetYF = _hlu.NhlNtiXAxisOffsetYF
NhlNtiXAxisOn = _hlu.NhlNtiXAxisOn
NhlNtiXAxisPosition = _hlu.NhlNtiXAxisPosition
NhlNtiXAxisSide = _hlu.NhlNtiXAxisSide
NhlNtiXAxisString = _hlu.NhlNtiXAxisString
NhlNtiYAxisAngleF = _hlu.NhlNtiYAxisAngleF
NhlNtiYAxisConstantSpacingF = _hlu.NhlNtiYAxisConstantSpacingF
NhlNtiYAxisDirection = _hlu.NhlNtiYAxisDirection
NhlNtiYAxisFont = _hlu.NhlNtiYAxisFont
NhlNtiYAxisFontAspectF = _hlu.NhlNtiYAxisFontAspectF
NhlNtiYAxisFontColor = _hlu.NhlNtiYAxisFontColor
NhlNtiYAxisFontHeightF = _hlu.NhlNtiYAxisFontHeightF
NhlNtiYAxisFontQuality = _hlu.NhlNtiYAxisFontQuality
NhlNtiYAxisFontThicknessF = _hlu.NhlNtiYAxisFontThicknessF
NhlNtiYAxisFuncCode = _hlu.NhlNtiYAxisFuncCode
NhlNtiYAxisJust = _hlu.NhlNtiYAxisJust
NhlNtiYAxisOffsetXF = _hlu.NhlNtiYAxisOffsetXF
NhlNtiYAxisOffsetYF = _hlu.NhlNtiYAxisOffsetYF
NhlNtiYAxisOn = _hlu.NhlNtiYAxisOn
NhlNtiYAxisPosition = _hlu.NhlNtiYAxisPosition
NhlNtiYAxisSide = _hlu.NhlNtiYAxisSide
NhlNtiYAxisString = _hlu.NhlNtiYAxisString
NhlNtmBorderLineColor = _hlu.NhlNtmBorderLineColor
NhlNtmBorderThicknessF = _hlu.NhlNtmBorderThicknessF
NhlNtmEqualizeXYSizes = _hlu.NhlNtmEqualizeXYSizes
NhlNtmLabelAutoStride = _hlu.NhlNtmLabelAutoStride
NhlNtmSciNoteCutoff = _hlu.NhlNtmSciNoteCutoff
NhlNtmXBAutoPrecision = _hlu.NhlNtmXBAutoPrecision
NhlNtmXBBorderOn = _hlu.NhlNtmXBBorderOn
NhlNtmXBDataLeftF = _hlu.NhlNtmXBDataLeftF
NhlNtmXBDataRightF = _hlu.NhlNtmXBDataRightF
NhlNtmXBFormat = _hlu.NhlNtmXBFormat
NhlNtmXBIrrTensionF = _hlu.NhlNtmXBIrrTensionF
NhlNtmXBIrregularPoints = _hlu.NhlNtmXBIrregularPoints
NhlNtmXBLabelAngleF = _hlu.NhlNtmXBLabelAngleF
NhlNtmXBLabelConstantSpacingF = _hlu.NhlNtmXBLabelConstantSpacingF
NhlNtmXBLabelDeltaF = _hlu.NhlNtmXBLabelDeltaF
NhlNtmXBLabelDirection = _hlu.NhlNtmXBLabelDirection
NhlNtmXBLabelFont = _hlu.NhlNtmXBLabelFont
NhlNtmXBLabelFontAspectF = _hlu.NhlNtmXBLabelFontAspectF
NhlNtmXBLabelFontColor = _hlu.NhlNtmXBLabelFontColor
NhlNtmXBLabelFontHeightF = _hlu.NhlNtmXBLabelFontHeightF
NhlNtmXBLabelFontQuality = _hlu.NhlNtmXBLabelFontQuality
NhlNtmXBLabelFontThicknessF = _hlu.NhlNtmXBLabelFontThicknessF
NhlNtmXBLabelFuncCode = _hlu.NhlNtmXBLabelFuncCode
NhlNtmXBLabelJust = _hlu.NhlNtmXBLabelJust
NhlNtmXBLabelStride = _hlu.NhlNtmXBLabelStride
NhlNtmXBLabels = _hlu.NhlNtmXBLabels
NhlNtmXBLabelsOn = _hlu.NhlNtmXBLabelsOn
NhlNtmXBMajorLengthF = _hlu.NhlNtmXBMajorLengthF
NhlNtmXBMajorLineColor = _hlu.NhlNtmXBMajorLineColor
NhlNtmXBMajorOutwardLengthF = _hlu.NhlNtmXBMajorOutwardLengthF
NhlNtmXBMajorThicknessF = _hlu.NhlNtmXBMajorThicknessF
NhlNtmXBMaxLabelLenF = _hlu.NhlNtmXBMaxLabelLenF
NhlNtmXBMaxTicks = _hlu.NhlNtmXBMaxTicks
NhlNtmXBMinLabelSpacingF = _hlu.NhlNtmXBMinLabelSpacingF
NhlNtmXBMinorLengthF = _hlu.NhlNtmXBMinorLengthF
NhlNtmXBMinorLineColor = _hlu.NhlNtmXBMinorLineColor
NhlNtmXBMinorOn = _hlu.NhlNtmXBMinorOn
NhlNtmXBMinorOutwardLengthF = _hlu.NhlNtmXBMinorOutwardLengthF
NhlNtmXBMinorPerMajor = _hlu.NhlNtmXBMinorPerMajor
NhlNtmXBMinorThicknessF = _hlu.NhlNtmXBMinorThicknessF
NhlNtmXBMinorValues = _hlu.NhlNtmXBMinorValues
NhlNtmXBMode = _hlu.NhlNtmXBMode
NhlNtmXBOn = _hlu.NhlNtmXBOn
NhlNtmXBPrecision = _hlu.NhlNtmXBPrecision
NhlNtmXBStyle = _hlu.NhlNtmXBStyle
NhlNtmXBTickEndF = _hlu.NhlNtmXBTickEndF
NhlNtmXBTickSpacingF = _hlu.NhlNtmXBTickSpacingF
NhlNtmXBTickStartF = _hlu.NhlNtmXBTickStartF
NhlNtmXBValues = _hlu.NhlNtmXBValues
NhlNtmXMajorGrid = _hlu.NhlNtmXMajorGrid
NhlNtmXMajorGridLineColor = _hlu.NhlNtmXMajorGridLineColor
NhlNtmXMajorGridLineDashPattern = _hlu.NhlNtmXMajorGridLineDashPattern
NhlNtmXMajorGridThicknessF = _hlu.NhlNtmXMajorGridThicknessF
NhlNtmXMinorGrid = _hlu.NhlNtmXMinorGrid
NhlNtmXMinorGridLineColor = _hlu.NhlNtmXMinorGridLineColor
NhlNtmXMinorGridLineDashPattern = _hlu.NhlNtmXMinorGridLineDashPattern
NhlNtmXMinorGridThicknessF = _hlu.NhlNtmXMinorGridThicknessF
NhlNtmXTAutoPrecision = _hlu.NhlNtmXTAutoPrecision
NhlNtmXTBorderOn = _hlu.NhlNtmXTBorderOn
NhlNtmXTDataLeftF = _hlu.NhlNtmXTDataLeftF
NhlNtmXTDataRightF = _hlu.NhlNtmXTDataRightF
NhlNtmXTFormat = _hlu.NhlNtmXTFormat
NhlNtmXTIrrTensionF = _hlu.NhlNtmXTIrrTensionF
NhlNtmXTIrregularPoints = _hlu.NhlNtmXTIrregularPoints
NhlNtmXTLabelAngleF = _hlu.NhlNtmXTLabelAngleF
NhlNtmXTLabelConstantSpacingF = _hlu.NhlNtmXTLabelConstantSpacingF
NhlNtmXTLabelDeltaF = _hlu.NhlNtmXTLabelDeltaF
NhlNtmXTLabelDirection = _hlu.NhlNtmXTLabelDirection
NhlNtmXTLabelFont = _hlu.NhlNtmXTLabelFont
NhlNtmXTLabelFontAspectF = _hlu.NhlNtmXTLabelFontAspectF
NhlNtmXTLabelFontColor = _hlu.NhlNtmXTLabelFontColor
NhlNtmXTLabelFontHeightF = _hlu.NhlNtmXTLabelFontHeightF
NhlNtmXTLabelFontQuality = _hlu.NhlNtmXTLabelFontQuality
NhlNtmXTLabelFontThicknessF = _hlu.NhlNtmXTLabelFontThicknessF
NhlNtmXTLabelFuncCode = _hlu.NhlNtmXTLabelFuncCode
NhlNtmXTLabelJust = _hlu.NhlNtmXTLabelJust
NhlNtmXTLabelStride = _hlu.NhlNtmXTLabelStride
NhlNtmXTLabels = _hlu.NhlNtmXTLabels
NhlNtmXTLabelsOn = _hlu.NhlNtmXTLabelsOn
NhlNtmXTMajorLengthF = _hlu.NhlNtmXTMajorLengthF
NhlNtmXTMajorLineColor = _hlu.NhlNtmXTMajorLineColor
NhlNtmXTMajorOutwardLengthF = _hlu.NhlNtmXTMajorOutwardLengthF
NhlNtmXTMajorThicknessF = _hlu.NhlNtmXTMajorThicknessF
NhlNtmXTMaxLabelLenF = _hlu.NhlNtmXTMaxLabelLenF
NhlNtmXTMaxTicks = _hlu.NhlNtmXTMaxTicks
NhlNtmXTMinLabelSpacingF = _hlu.NhlNtmXTMinLabelSpacingF
NhlNtmXTMinorLengthF = _hlu.NhlNtmXTMinorLengthF
NhlNtmXTMinorLineColor = _hlu.NhlNtmXTMinorLineColor
NhlNtmXTMinorOn = _hlu.NhlNtmXTMinorOn
NhlNtmXTMinorOutwardLengthF = _hlu.NhlNtmXTMinorOutwardLengthF
NhlNtmXTMinorPerMajor = _hlu.NhlNtmXTMinorPerMajor
NhlNtmXTMinorThicknessF = _hlu.NhlNtmXTMinorThicknessF
NhlNtmXTMinorValues = _hlu.NhlNtmXTMinorValues
NhlNtmXTMode = _hlu.NhlNtmXTMode
NhlNtmXTOn = _hlu.NhlNtmXTOn
NhlNtmXTPrecision = _hlu.NhlNtmXTPrecision
NhlNtmXTStyle = _hlu.NhlNtmXTStyle
NhlNtmXTTickEndF = _hlu.NhlNtmXTTickEndF
NhlNtmXTTickSpacingF = _hlu.NhlNtmXTTickSpacingF
NhlNtmXTTickStartF = _hlu.NhlNtmXTTickStartF
NhlNtmXTValues = _hlu.NhlNtmXTValues
NhlNtmXUseBottom = _hlu.NhlNtmXUseBottom
NhlNtmYLAutoPrecision = _hlu.NhlNtmYLAutoPrecision
NhlNtmYLBorderOn = _hlu.NhlNtmYLBorderOn
NhlNtmYLDataBottomF = _hlu.NhlNtmYLDataBottomF
NhlNtmYLDataTopF = _hlu.NhlNtmYLDataTopF
NhlNtmYLFormat = _hlu.NhlNtmYLFormat
NhlNtmYLIrrTensionF = _hlu.NhlNtmYLIrrTensionF
NhlNtmYLIrregularPoints = _hlu.NhlNtmYLIrregularPoints
NhlNtmYLLabelAngleF = _hlu.NhlNtmYLLabelAngleF
NhlNtmYLLabelConstantSpacingF = _hlu.NhlNtmYLLabelConstantSpacingF
NhlNtmYLLabelDeltaF = _hlu.NhlNtmYLLabelDeltaF
NhlNtmYLLabelDirection = _hlu.NhlNtmYLLabelDirection
NhlNtmYLLabelFont = _hlu.NhlNtmYLLabelFont
NhlNtmYLLabelFontAspectF = _hlu.NhlNtmYLLabelFontAspectF
NhlNtmYLLabelFontColor = _hlu.NhlNtmYLLabelFontColor
NhlNtmYLLabelFontHeightF = _hlu.NhlNtmYLLabelFontHeightF
NhlNtmYLLabelFontQuality = _hlu.NhlNtmYLLabelFontQuality
NhlNtmYLLabelFontThicknessF = _hlu.NhlNtmYLLabelFontThicknessF
NhlNtmYLLabelFuncCode = _hlu.NhlNtmYLLabelFuncCode
NhlNtmYLLabelJust = _hlu.NhlNtmYLLabelJust
NhlNtmYLLabelStride = _hlu.NhlNtmYLLabelStride
NhlNtmYLLabels = _hlu.NhlNtmYLLabels
NhlNtmYLLabelsOn = _hlu.NhlNtmYLLabelsOn
NhlNtmYLMajorLengthF = _hlu.NhlNtmYLMajorLengthF
NhlNtmYLMajorLineColor = _hlu.NhlNtmYLMajorLineColor
NhlNtmYLMajorOutwardLengthF = _hlu.NhlNtmYLMajorOutwardLengthF
NhlNtmYLMajorThicknessF = _hlu.NhlNtmYLMajorThicknessF
NhlNtmYLMaxLabelLenF = _hlu.NhlNtmYLMaxLabelLenF
NhlNtmYLMaxTicks = _hlu.NhlNtmYLMaxTicks
NhlNtmYLMinLabelSpacingF = _hlu.NhlNtmYLMinLabelSpacingF
NhlNtmYLMinorLengthF = _hlu.NhlNtmYLMinorLengthF
NhlNtmYLMinorLineColor = _hlu.NhlNtmYLMinorLineColor
NhlNtmYLMinorOn = _hlu.NhlNtmYLMinorOn
NhlNtmYLMinorOutwardLengthF = _hlu.NhlNtmYLMinorOutwardLengthF
NhlNtmYLMinorPerMajor = _hlu.NhlNtmYLMinorPerMajor
NhlNtmYLMinorThicknessF = _hlu.NhlNtmYLMinorThicknessF
NhlNtmYLMinorValues = _hlu.NhlNtmYLMinorValues
NhlNtmYLMode = _hlu.NhlNtmYLMode
NhlNtmYLOn = _hlu.NhlNtmYLOn
NhlNtmYLPrecision = _hlu.NhlNtmYLPrecision
NhlNtmYLStyle = _hlu.NhlNtmYLStyle
NhlNtmYLTickEndF = _hlu.NhlNtmYLTickEndF
NhlNtmYLTickSpacingF = _hlu.NhlNtmYLTickSpacingF
NhlNtmYLTickStartF = _hlu.NhlNtmYLTickStartF
NhlNtmYLValues = _hlu.NhlNtmYLValues
NhlNtmYMajorGrid = _hlu.NhlNtmYMajorGrid
NhlNtmYMajorGridLineColor = _hlu.NhlNtmYMajorGridLineColor
NhlNtmYMajorGridLineDashPattern = _hlu.NhlNtmYMajorGridLineDashPattern
NhlNtmYMajorGridThicknessF = _hlu.NhlNtmYMajorGridThicknessF
NhlNtmYMinorGrid = _hlu.NhlNtmYMinorGrid
NhlNtmYMinorGridLineColor = _hlu.NhlNtmYMinorGridLineColor
NhlNtmYMinorGridLineDashPattern = _hlu.NhlNtmYMinorGridLineDashPattern
NhlNtmYMinorGridThicknessF = _hlu.NhlNtmYMinorGridThicknessF
NhlNtmYRAutoPrecision = _hlu.NhlNtmYRAutoPrecision
NhlNtmYRBorderOn = _hlu.NhlNtmYRBorderOn
NhlNtmYRDataBottomF = _hlu.NhlNtmYRDataBottomF
NhlNtmYRDataTopF = _hlu.NhlNtmYRDataTopF
NhlNtmYRFormat = _hlu.NhlNtmYRFormat
NhlNtmYRIrrTensionF = _hlu.NhlNtmYRIrrTensionF
NhlNtmYRIrregularPoints = _hlu.NhlNtmYRIrregularPoints
NhlNtmYRLabelAngleF = _hlu.NhlNtmYRLabelAngleF
NhlNtmYRLabelConstantSpacingF = _hlu.NhlNtmYRLabelConstantSpacingF
NhlNtmYRLabelDeltaF = _hlu.NhlNtmYRLabelDeltaF
NhlNtmYRLabelDirection = _hlu.NhlNtmYRLabelDirection
NhlNtmYRLabelFont = _hlu.NhlNtmYRLabelFont
NhlNtmYRLabelFontAspectF = _hlu.NhlNtmYRLabelFontAspectF
NhlNtmYRLabelFontColor = _hlu.NhlNtmYRLabelFontColor
NhlNtmYRLabelFontHeightF = _hlu.NhlNtmYRLabelFontHeightF
NhlNtmYRLabelFontQuality = _hlu.NhlNtmYRLabelFontQuality
NhlNtmYRLabelFontThicknessF = _hlu.NhlNtmYRLabelFontThicknessF
NhlNtmYRLabelFuncCode = _hlu.NhlNtmYRLabelFuncCode
NhlNtmYRLabelJust = _hlu.NhlNtmYRLabelJust
NhlNtmYRLabelStride = _hlu.NhlNtmYRLabelStride
NhlNtmYRLabels = _hlu.NhlNtmYRLabels
NhlNtmYRLabelsOn = _hlu.NhlNtmYRLabelsOn
NhlNtmYRMajorLengthF = _hlu.NhlNtmYRMajorLengthF
NhlNtmYRMajorLineColor = _hlu.NhlNtmYRMajorLineColor
NhlNtmYRMajorOutwardLengthF = _hlu.NhlNtmYRMajorOutwardLengthF
NhlNtmYRMajorThicknessF = _hlu.NhlNtmYRMajorThicknessF
NhlNtmYRMaxLabelLenF = _hlu.NhlNtmYRMaxLabelLenF
NhlNtmYRMaxTicks = _hlu.NhlNtmYRMaxTicks
NhlNtmYRMinLabelSpacingF = _hlu.NhlNtmYRMinLabelSpacingF
NhlNtmYRMinorLengthF = _hlu.NhlNtmYRMinorLengthF
NhlNtmYRMinorLineColor = _hlu.NhlNtmYRMinorLineColor
NhlNtmYRMinorOn = _hlu.NhlNtmYRMinorOn
NhlNtmYRMinorOutwardLengthF = _hlu.NhlNtmYRMinorOutwardLengthF
NhlNtmYRMinorPerMajor = _hlu.NhlNtmYRMinorPerMajor
NhlNtmYRMinorThicknessF = _hlu.NhlNtmYRMinorThicknessF
NhlNtmYRMinorValues = _hlu.NhlNtmYRMinorValues
NhlNtmYRMode = _hlu.NhlNtmYRMode
NhlNtmYROn = _hlu.NhlNtmYROn
NhlNtmYRPrecision = _hlu.NhlNtmYRPrecision
NhlNtmYRStyle = _hlu.NhlNtmYRStyle
NhlNtmYRTickEndF = _hlu.NhlNtmYRTickEndF
NhlNtmYRTickSpacingF = _hlu.NhlNtmYRTickSpacingF
NhlNtmYRTickStartF = _hlu.NhlNtmYRTickStartF
NhlNtmYRValues = _hlu.NhlNtmYRValues
NhlNtmYUseLeft = _hlu.NhlNtmYUseLeft
NhlNtrXAxisType = _hlu.NhlNtrXAxisType
NhlNtrXCoordPoints = _hlu.NhlNtrXCoordPoints
NhlNtrXInterPoints = _hlu.NhlNtrXInterPoints
NhlNtrXSamples = _hlu.NhlNtrXSamples
NhlNtrXTensionF = _hlu.NhlNtrXTensionF
NhlNtrYAxisType = _hlu.NhlNtrYAxisType
NhlNtrYCoordPoints = _hlu.NhlNtrYCoordPoints
NhlNtrYInterPoints = _hlu.NhlNtrYInterPoints
NhlNtrYSamples = _hlu.NhlNtrYSamples
NhlNtrYTensionF = _hlu.NhlNtrYTensionF
NhlNtrXLog = _hlu.NhlNtrXLog
NhlNtrYLog = _hlu.NhlNtrYLog
NhlNtrLineInterpolationOn = _hlu.NhlNtrLineInterpolationOn
NhlNtrXMaxF = _hlu.NhlNtrXMaxF
NhlNtrXMinF = _hlu.NhlNtrXMinF
NhlNtrXReverse = _hlu.NhlNtrXReverse
NhlNtrYMaxF = _hlu.NhlNtrYMaxF
NhlNtrYMinF = _hlu.NhlNtrYMinF
NhlNtrYReverse = _hlu.NhlNtrYReverse
NhlNtxAngleF = _hlu.NhlNtxAngleF
NhlNtxBackgroundFillColor = _hlu.NhlNtxBackgroundFillColor
NhlNtxConstantSpacingF = _hlu.NhlNtxConstantSpacingF
NhlNtxDirection = _hlu.NhlNtxDirection
NhlNtxFont = _hlu.NhlNtxFont
NhlNtxFontAspectF = _hlu.NhlNtxFontAspectF
NhlNtxFontColor = _hlu.NhlNtxFontColor
NhlNtxFontHeightF = _hlu.NhlNtxFontHeightF
NhlNtxFontQuality = _hlu.NhlNtxFontQuality
NhlNtxFontThicknessF = _hlu.NhlNtxFontThicknessF
NhlNtxFuncCode = _hlu.NhlNtxFuncCode
NhlNtxJust = _hlu.NhlNtxJust
NhlNtxPerimColor = _hlu.NhlNtxPerimColor
NhlNtxPerimDashLengthF = _hlu.NhlNtxPerimDashLengthF
NhlNtxPerimDashPattern = _hlu.NhlNtxPerimDashPattern
NhlNtxPerimOn = _hlu.NhlNtxPerimOn
NhlNtxPerimSpaceF = _hlu.NhlNtxPerimSpaceF
NhlNtxPerimThicknessF = _hlu.NhlNtxPerimThicknessF
NhlNtxPosXF = _hlu.NhlNtxPosXF
NhlNtxPosYF = _hlu.NhlNtxPosYF
NhlNtxString = _hlu.NhlNtxString
NhlNvcExplicitLabelBarLabelsOn = _hlu.NhlNvcExplicitLabelBarLabelsOn
NhlNvcFillArrowEdgeColor = _hlu.NhlNvcFillArrowEdgeColor
NhlNvcFillArrowEdgeThicknessF = _hlu.NhlNvcFillArrowEdgeThicknessF
NhlNvcFillArrowFillColor = _hlu.NhlNvcFillArrowFillColor
NhlNvcFillArrowHeadInteriorXF = _hlu.NhlNvcFillArrowHeadInteriorXF
NhlNvcFillArrowHeadMinFracXF = _hlu.NhlNvcFillArrowHeadMinFracXF
NhlNvcFillArrowHeadMinFracYF = _hlu.NhlNvcFillArrowHeadMinFracYF
NhlNvcFillArrowHeadXF = _hlu.NhlNvcFillArrowHeadXF
NhlNvcFillArrowHeadYF = _hlu.NhlNvcFillArrowHeadYF
NhlNvcFillArrowMinFracWidthF = _hlu.NhlNvcFillArrowMinFracWidthF
NhlNvcFillArrowWidthF = _hlu.NhlNvcFillArrowWidthF
NhlNvcFillArrowsOn = _hlu.NhlNvcFillArrowsOn
NhlNvcFillOverEdge = _hlu.NhlNvcFillOverEdge
NhlNvcGlyphStyle = _hlu.NhlNvcGlyphStyle
NhlNvcLabelBarEndLabelsOn = _hlu.NhlNvcLabelBarEndLabelsOn
NhlNvcLabelFontColor = _hlu.NhlNvcLabelFontColor
NhlNvcLabelFontHeightF = _hlu.NhlNvcLabelFontHeightF
NhlNvcLabelsOn = _hlu.NhlNvcLabelsOn
NhlNvcLabelsUseVectorColor = _hlu.NhlNvcLabelsUseVectorColor
NhlNvcLevelColors = _hlu.NhlNvcLevelColors
NhlNvcLevelCount = _hlu.NhlNvcLevelCount
NhlNvcLevelSelectionMode = _hlu.NhlNvcLevelSelectionMode
NhlNvcLevelSpacingF = _hlu.NhlNvcLevelSpacingF
NhlNvcLevels = _hlu.NhlNvcLevels
NhlNvcLineArrowHeadMaxSizeF = _hlu.NhlNvcLineArrowHeadMaxSizeF
NhlNvcLineArrowHeadMinSizeF = _hlu.NhlNvcLineArrowHeadMinSizeF
NhlNvcLineArrowThicknessF = _hlu.NhlNvcLineArrowThicknessF
NhlNvcMagnitudeFormat = _hlu.NhlNvcMagnitudeFormat
NhlNvcMagnitudeScaleFactorF = _hlu.NhlNvcMagnitudeScaleFactorF
NhlNvcMagnitudeScaleValueF = _hlu.NhlNvcMagnitudeScaleValueF
NhlNvcMagnitudeScalingMode = _hlu.NhlNvcMagnitudeScalingMode
NhlNvcMapDirection = _hlu.NhlNvcMapDirection
NhlNvcMaxLevelCount = _hlu.NhlNvcMaxLevelCount
NhlNvcMaxLevelValF = _hlu.NhlNvcMaxLevelValF
NhlNvcMaxMagnitudeF = _hlu.NhlNvcMaxMagnitudeF
NhlNvcMinAnnoAngleF = _hlu.NhlNvcMinAnnoAngleF
NhlNvcMinAnnoArrowAngleF = _hlu.NhlNvcMinAnnoArrowAngleF
NhlNvcMinAnnoArrowEdgeColor = _hlu.NhlNvcMinAnnoArrowEdgeColor
NhlNvcMinAnnoArrowFillColor = _hlu.NhlNvcMinAnnoArrowFillColor
NhlNvcMinAnnoArrowLineColor = _hlu.NhlNvcMinAnnoArrowLineColor
NhlNvcMinAnnoArrowMinOffsetF = _hlu.NhlNvcMinAnnoArrowMinOffsetF
NhlNvcMinAnnoArrowSpaceF = _hlu.NhlNvcMinAnnoArrowSpaceF
NhlNvcMinAnnoArrowUseVecColor = _hlu.NhlNvcMinAnnoArrowUseVecColor
NhlNvcMinAnnoBackgroundColor = _hlu.NhlNvcMinAnnoBackgroundColor
NhlNvcMinAnnoConstantSpacingF = _hlu.NhlNvcMinAnnoConstantSpacingF
NhlNvcMinAnnoExplicitMagnitudeF = _hlu.NhlNvcMinAnnoExplicitMagnitudeF
NhlNvcMinAnnoFont = _hlu.NhlNvcMinAnnoFont
NhlNvcMinAnnoFontAspectF = _hlu.NhlNvcMinAnnoFontAspectF
NhlNvcMinAnnoFontColor = _hlu.NhlNvcMinAnnoFontColor
NhlNvcMinAnnoFontHeightF = _hlu.NhlNvcMinAnnoFontHeightF
NhlNvcMinAnnoFontQuality = _hlu.NhlNvcMinAnnoFontQuality
NhlNvcMinAnnoFontThicknessF = _hlu.NhlNvcMinAnnoFontThicknessF
NhlNvcMinAnnoFuncCode = _hlu.NhlNvcMinAnnoFuncCode
NhlNvcMinAnnoJust = _hlu.NhlNvcMinAnnoJust
NhlNvcMinAnnoOn = _hlu.NhlNvcMinAnnoOn
NhlNvcMinAnnoOrientation = _hlu.NhlNvcMinAnnoOrientation
NhlNvcMinAnnoOrthogonalPosF = _hlu.NhlNvcMinAnnoOrthogonalPosF
NhlNvcMinAnnoParallelPosF = _hlu.NhlNvcMinAnnoParallelPosF
NhlNvcMinAnnoPerimColor = _hlu.NhlNvcMinAnnoPerimColor
NhlNvcMinAnnoPerimOn = _hlu.NhlNvcMinAnnoPerimOn
NhlNvcMinAnnoPerimSpaceF = _hlu.NhlNvcMinAnnoPerimSpaceF
NhlNvcMinAnnoPerimThicknessF = _hlu.NhlNvcMinAnnoPerimThicknessF
NhlNvcMinAnnoSide = _hlu.NhlNvcMinAnnoSide
NhlNvcMinAnnoString1 = _hlu.NhlNvcMinAnnoString1
NhlNvcMinAnnoString1On = _hlu.NhlNvcMinAnnoString1On
NhlNvcMinAnnoString2 = _hlu.NhlNvcMinAnnoString2
NhlNvcMinAnnoString2On = _hlu.NhlNvcMinAnnoString2On
NhlNvcMinAnnoTextDirection = _hlu.NhlNvcMinAnnoTextDirection
NhlNvcMinAnnoZone = _hlu.NhlNvcMinAnnoZone
NhlNvcMinDistanceF = _hlu.NhlNvcMinDistanceF
NhlNvcMinFracLengthF = _hlu.NhlNvcMinFracLengthF
NhlNvcMinLevelValF = _hlu.NhlNvcMinLevelValF
NhlNvcMinMagnitudeF = _hlu.NhlNvcMinMagnitudeF
NhlNvcMonoFillArrowEdgeColor = _hlu.NhlNvcMonoFillArrowEdgeColor
NhlNvcMonoFillArrowFillColor = _hlu.NhlNvcMonoFillArrowFillColor
NhlNvcMonoLineArrowColor = _hlu.NhlNvcMonoLineArrowColor
NhlNvcMonoWindBarbColor = _hlu.NhlNvcMonoWindBarbColor
NhlNvcNoDataLabelOn = _hlu.NhlNvcNoDataLabelOn
NhlNvcNoDataLabelString = _hlu.NhlNvcNoDataLabelString
NhlNvcPositionMode = _hlu.NhlNvcPositionMode
NhlNvcRefAnnoAngleF = _hlu.NhlNvcRefAnnoAngleF
NhlNvcRefAnnoArrowAngleF = _hlu.NhlNvcRefAnnoArrowAngleF
NhlNvcRefAnnoArrowEdgeColor = _hlu.NhlNvcRefAnnoArrowEdgeColor
NhlNvcRefAnnoArrowFillColor = _hlu.NhlNvcRefAnnoArrowFillColor
NhlNvcRefAnnoArrowLineColor = _hlu.NhlNvcRefAnnoArrowLineColor
NhlNvcRefAnnoArrowMinOffsetF = _hlu.NhlNvcRefAnnoArrowMinOffsetF
NhlNvcRefAnnoArrowSpaceF = _hlu.NhlNvcRefAnnoArrowSpaceF
NhlNvcRefAnnoArrowUseVecColor = _hlu.NhlNvcRefAnnoArrowUseVecColor
NhlNvcRefAnnoBackgroundColor = _hlu.NhlNvcRefAnnoBackgroundColor
NhlNvcRefAnnoConstantSpacingF = _hlu.NhlNvcRefAnnoConstantSpacingF
NhlNvcRefAnnoExplicitMagnitudeF = _hlu.NhlNvcRefAnnoExplicitMagnitudeF
NhlNvcRefAnnoFont = _hlu.NhlNvcRefAnnoFont
NhlNvcRefAnnoFontAspectF = _hlu.NhlNvcRefAnnoFontAspectF
NhlNvcRefAnnoFontColor = _hlu.NhlNvcRefAnnoFontColor
NhlNvcRefAnnoFontHeightF = _hlu.NhlNvcRefAnnoFontHeightF
NhlNvcRefAnnoFontQuality = _hlu.NhlNvcRefAnnoFontQuality
NhlNvcRefAnnoFontThicknessF = _hlu.NhlNvcRefAnnoFontThicknessF
NhlNvcRefAnnoFuncCode = _hlu.NhlNvcRefAnnoFuncCode
NhlNvcRefAnnoJust = _hlu.NhlNvcRefAnnoJust
NhlNvcRefAnnoOn = _hlu.NhlNvcRefAnnoOn
NhlNvcRefAnnoOrientation = _hlu.NhlNvcRefAnnoOrientation
NhlNvcRefAnnoOrthogonalPosF = _hlu.NhlNvcRefAnnoOrthogonalPosF
NhlNvcRefAnnoParallelPosF = _hlu.NhlNvcRefAnnoParallelPosF
NhlNvcRefAnnoPerimColor = _hlu.NhlNvcRefAnnoPerimColor
NhlNvcRefAnnoPerimOn = _hlu.NhlNvcRefAnnoPerimOn
NhlNvcRefAnnoPerimSpaceF = _hlu.NhlNvcRefAnnoPerimSpaceF
NhlNvcRefAnnoPerimThicknessF = _hlu.NhlNvcRefAnnoPerimThicknessF
NhlNvcRefAnnoSide = _hlu.NhlNvcRefAnnoSide
NhlNvcRefAnnoString1 = _hlu.NhlNvcRefAnnoString1
NhlNvcRefAnnoString1On = _hlu.NhlNvcRefAnnoString1On
NhlNvcRefAnnoString2 = _hlu.NhlNvcRefAnnoString2
NhlNvcRefAnnoString2On = _hlu.NhlNvcRefAnnoString2On
NhlNvcRefAnnoTextDirection = _hlu.NhlNvcRefAnnoTextDirection
NhlNvcRefAnnoZone = _hlu.NhlNvcRefAnnoZone
NhlNvcRefLengthF = _hlu.NhlNvcRefLengthF
NhlNvcRefMagnitudeF = _hlu.NhlNvcRefMagnitudeF
NhlNvcScalarFieldData = _hlu.NhlNvcScalarFieldData
NhlNvcScalarMissingValColor = _hlu.NhlNvcScalarMissingValColor
NhlNvcScalarValueFormat = _hlu.NhlNvcScalarValueFormat
NhlNvcScalarValueScaleFactorF = _hlu.NhlNvcScalarValueScaleFactorF
NhlNvcScalarValueScaleValueF = _hlu.NhlNvcScalarValueScaleValueF
NhlNvcScalarValueScalingMode = _hlu.NhlNvcScalarValueScalingMode
NhlNvcUseRefAnnoRes = _hlu.NhlNvcUseRefAnnoRes
NhlNvcUseScalarArray = _hlu.NhlNvcUseScalarArray
NhlNvcVectorDrawOrder = _hlu.NhlNvcVectorDrawOrder
NhlNvcVectorFieldData = _hlu.NhlNvcVectorFieldData
NhlNvcWindBarbCalmCircleSizeF = _hlu.NhlNvcWindBarbCalmCircleSizeF
NhlNvcWindBarbColor = _hlu.NhlNvcWindBarbColor
NhlNvcWindBarbLineThicknessF = _hlu.NhlNvcWindBarbLineThicknessF
NhlNvcWindBarbScaleFactorF = _hlu.NhlNvcWindBarbScaleFactorF
NhlNvcWindBarbTickAngleF = _hlu.NhlNvcWindBarbTickAngleF
NhlNvcWindBarbTickLengthF = _hlu.NhlNvcWindBarbTickLengthF
NhlNvcWindBarbTickSpacingF = _hlu.NhlNvcWindBarbTickSpacingF
NhlNvcZeroFLabelAngleF = _hlu.NhlNvcZeroFLabelAngleF
NhlNvcZeroFLabelBackgroundColor = _hlu.NhlNvcZeroFLabelBackgroundColor
NhlNvcZeroFLabelConstantSpacingF = _hlu.NhlNvcZeroFLabelConstantSpacingF
NhlNvcZeroFLabelFont = _hlu.NhlNvcZeroFLabelFont
NhlNvcZeroFLabelFontAspectF = _hlu.NhlNvcZeroFLabelFontAspectF
NhlNvcZeroFLabelFontColor = _hlu.NhlNvcZeroFLabelFontColor
NhlNvcZeroFLabelFontHeightF = _hlu.NhlNvcZeroFLabelFontHeightF
NhlNvcZeroFLabelFontQuality = _hlu.NhlNvcZeroFLabelFontQuality
NhlNvcZeroFLabelFontThicknessF = _hlu.NhlNvcZeroFLabelFontThicknessF
NhlNvcZeroFLabelFuncCode = _hlu.NhlNvcZeroFLabelFuncCode
NhlNvcZeroFLabelJust = _hlu.NhlNvcZeroFLabelJust
NhlNvcZeroFLabelOn = _hlu.NhlNvcZeroFLabelOn
NhlNvcZeroFLabelOrthogonalPosF = _hlu.NhlNvcZeroFLabelOrthogonalPosF
NhlNvcZeroFLabelParallelPosF = _hlu.NhlNvcZeroFLabelParallelPosF
NhlNvcZeroFLabelPerimColor = _hlu.NhlNvcZeroFLabelPerimColor
NhlNvcZeroFLabelPerimOn = _hlu.NhlNvcZeroFLabelPerimOn
NhlNvcZeroFLabelPerimSpaceF = _hlu.NhlNvcZeroFLabelPerimSpaceF
NhlNvcZeroFLabelPerimThicknessF = _hlu.NhlNvcZeroFLabelPerimThicknessF
NhlNvcZeroFLabelSide = _hlu.NhlNvcZeroFLabelSide
NhlNvcZeroFLabelString = _hlu.NhlNvcZeroFLabelString
NhlNvcZeroFLabelTextDirection = _hlu.NhlNvcZeroFLabelTextDirection
NhlNvcZeroFLabelZone = _hlu.NhlNvcZeroFLabelZone
NhlNvfYCStartSubsetV = _hlu.NhlNvfYCStartSubsetV
NhlNvpAnnoManagerId = _hlu.NhlNvpAnnoManagerId
NhlNvpHeightF = _hlu.NhlNvpHeightF
NhlNvpKeepAspect = _hlu.NhlNvpKeepAspect
NhlNvpOn = _hlu.NhlNvpOn
NhlNvpUseSegments = _hlu.NhlNvpUseSegments
NhlNvpWidthF = _hlu.NhlNvpWidthF
NhlNvpXF = _hlu.NhlNvpXF
NhlNvpYF = _hlu.NhlNvpYF
NhlNwkMetaName = _hlu.NhlNwkMetaName
NhlNwkDeviceLowerX = _hlu.NhlNwkDeviceLowerX
NhlNwkDeviceLowerY = _hlu.NhlNwkDeviceLowerY
NhlNwkDeviceUpperX = _hlu.NhlNwkDeviceUpperX
NhlNwkDeviceUpperY = _hlu.NhlNwkDeviceUpperY
NhlNwkPSFileName = _hlu.NhlNwkPSFileName
NhlNwkPSFormat = _hlu.NhlNwkPSFormat
NhlNwkPSResolution = _hlu.NhlNwkPSResolution
NhlNwkPDFFileName = _hlu.NhlNwkPDFFileName
NhlNwkPDFFormat = _hlu.NhlNwkPDFFormat
NhlNwkPDFResolution = _hlu.NhlNwkPDFResolution
NhlNwkVisualType = _hlu.NhlNwkVisualType
NhlNwkColorModel = _hlu.NhlNwkColorModel
NhlNwkBackgroundColor = _hlu.NhlNwkBackgroundColor
NhlNwkColorMap = _hlu.NhlNwkColorMap
NhlNwkColorMapLen = _hlu.NhlNwkColorMapLen
NhlNwkDashTableLength = _hlu.NhlNwkDashTableLength
NhlNwkDefGraphicStyleId = _hlu.NhlNwkDefGraphicStyleId
NhlNwkFillTableLength = _hlu.NhlNwkFillTableLength
NhlNwkForegroundColor = _hlu.NhlNwkForegroundColor
NhlNwkGksWorkId = _hlu.NhlNwkGksWorkId
NhlNwkMarkerTableLength = _hlu.NhlNwkMarkerTableLength
NhlNwkTopLevelViews = _hlu.NhlNwkTopLevelViews
NhlNwkViews = _hlu.NhlNwkViews
NhlNwkPause = _hlu.NhlNwkPause
NhlNwkWindowId = _hlu.NhlNwkWindowId
NhlNwkXColorMode = _hlu.NhlNwkXColorMode
NhlNwsCurrentSize = _hlu.NhlNwsCurrentSize
NhlNwsMaximumSize = _hlu.NhlNwsMaximumSize
NhlNwsThresholdSize = _hlu.NhlNwsThresholdSize
NhlNxyComputeXMax = _hlu.NhlNxyComputeXMax
NhlNxyComputeXMin = _hlu.NhlNxyComputeXMin
NhlNxyComputeYMax = _hlu.NhlNxyComputeYMax
NhlNxyComputeYMin = _hlu.NhlNxyComputeYMin
NhlNxyCoordData = _hlu.NhlNxyCoordData
NhlNxyCoordDataSpec = _hlu.NhlNxyCoordDataSpec
NhlNxyCurveDrawOrder = _hlu.NhlNxyCurveDrawOrder
NhlNxyDashPattern = _hlu.NhlNxyDashPattern
NhlNxyDashPatterns = _hlu.NhlNxyDashPatterns
NhlNxyExplicitLabels = _hlu.NhlNxyExplicitLabels
NhlNxyExplicitLegendLabels = _hlu.NhlNxyExplicitLegendLabels
NhlNxyLabelMode = _hlu.NhlNxyLabelMode
NhlNxyLineColor = _hlu.NhlNxyLineColor
NhlNxyLineColors = _hlu.NhlNxyLineColors
NhlNxyLineDashSegLenF = _hlu.NhlNxyLineDashSegLenF
NhlNxyLineLabelConstantSpacingF = _hlu.NhlNxyLineLabelConstantSpacingF
NhlNxyLineLabelFont = _hlu.NhlNxyLineLabelFont
NhlNxyLineLabelFontAspectF = _hlu.NhlNxyLineLabelFontAspectF
NhlNxyLineLabelFontColor = _hlu.NhlNxyLineLabelFontColor
NhlNxyLineLabelFontColors = _hlu.NhlNxyLineLabelFontColors
NhlNxyLineLabelFontHeightF = _hlu.NhlNxyLineLabelFontHeightF
NhlNxyLineLabelFontQuality = _hlu.NhlNxyLineLabelFontQuality
NhlNxyLineLabelFontThicknessF = _hlu.NhlNxyLineLabelFontThicknessF
NhlNxyLineLabelFuncCode = _hlu.NhlNxyLineLabelFuncCode
NhlNxyLineThicknessF = _hlu.NhlNxyLineThicknessF
NhlNxyLineThicknesses = _hlu.NhlNxyLineThicknesses
NhlNxyMarkLineMode = _hlu.NhlNxyMarkLineMode
NhlNxyMarkLineModes = _hlu.NhlNxyMarkLineModes
NhlNxyMarker = _hlu.NhlNxyMarker
NhlNxyMarkerColor = _hlu.NhlNxyMarkerColor
NhlNxyMarkerColors = _hlu.NhlNxyMarkerColors
NhlNxyMarkerSizeF = _hlu.NhlNxyMarkerSizeF
NhlNxyMarkerSizes = _hlu.NhlNxyMarkerSizes
NhlNxyMarkerThicknessF = _hlu.NhlNxyMarkerThicknessF
NhlNxyMarkerThicknesses = _hlu.NhlNxyMarkerThicknesses
NhlNxyMarkers = _hlu.NhlNxyMarkers
NhlNxyMonoDashPattern = _hlu.NhlNxyMonoDashPattern
NhlNxyMonoLineColor = _hlu.NhlNxyMonoLineColor
NhlNxyMonoLineLabelFontColor = _hlu.NhlNxyMonoLineLabelFontColor
NhlNxyMonoLineThickness = _hlu.NhlNxyMonoLineThickness
NhlNxyMonoMarkLineMode = _hlu.NhlNxyMonoMarkLineMode
NhlNxyMonoMarker = _hlu.NhlNxyMonoMarker
NhlNxyMonoMarkerColor = _hlu.NhlNxyMonoMarkerColor
NhlNxyMonoMarkerSize = _hlu.NhlNxyMonoMarkerSize
NhlNxyMonoMarkerThickness = _hlu.NhlNxyMonoMarkerThickness
NhlNxyXIrrTensionF = _hlu.NhlNxyXIrrTensionF
NhlNxyXIrregularPoints = _hlu.NhlNxyXIrregularPoints
NhlNxyXStyle = _hlu.NhlNxyXStyle
NhlNxyYIrrTensionF = _hlu.NhlNxyYIrrTensionF
NhlNxyYIrregularPoints = _hlu.NhlNxyYIrregularPoints
NhlNxyYStyle = _hlu.NhlNxyYStyle
NhlTFillIndexFullEnum = _hlu.NhlTFillIndexFullEnum
NhlTFillIndexFullEnumGenArray = _hlu.NhlTFillIndexFullEnumGenArray
NhlUNSPECIFIEDFILL = _hlu.NhlUNSPECIFIEDFILL
NhlTFillIndex = _hlu.NhlTFillIndex
NhlTFillIndexGenArray = _hlu.NhlTFillIndexGenArray
NhlHOLLOWFILL = _hlu.NhlHOLLOWFILL
NhlNULLFILL = _hlu.NhlNULLFILL
NhlSOLIDFILL = _hlu.NhlSOLIDFILL
NhlWK_INITIAL_FILL_BUFSIZE = _hlu.NhlWK_INITIAL_FILL_BUFSIZE
new_intp = _hlu.new_intp

copy_intp = _hlu.copy_intp

delete_intp = _hlu.delete_intp

intp_assign = _hlu.intp_assign

intp_value = _hlu.intp_value

new_floatArray = _hlu.new_floatArray

delete_floatArray = _hlu.delete_floatArray

floatArray_getitem = _hlu.floatArray_getitem

floatArray_setitem = _hlu.floatArray_setitem

NhlSETRL = _hlu.NhlSETRL
NhlGETRL = _hlu.NhlGETRL
NhlFATAL = _hlu.NhlFATAL
NhlWARNING = _hlu.NhlWARNING
NhlINFO = _hlu.NhlINFO
NhlNOERROR = _hlu.NhlNOERROR
NhlNOLINE = _hlu.NhlNOLINE
NhlLINEONLY = _hlu.NhlLINEONLY
NhlLABELONLY = _hlu.NhlLABELONLY
NhlLINEANDLABEL = _hlu.NhlLINEANDLABEL
NhlPOLYLINE = _hlu.NhlPOLYLINE
NhlPOLYMARKER = _hlu.NhlPOLYMARKER
NhlPOLYGON = _hlu.NhlPOLYGON
NhlDEFAULT_APP = _hlu.NhlDEFAULT_APP
False = _hlu.False
True = _hlu.True
NhlAUTOMATIC = _hlu.NhlAUTOMATIC
NhlMANUAL = _hlu.NhlMANUAL
NhlEXPLICIT = _hlu.NhlEXPLICIT
NhlLOG = _hlu.NhlLOG
NhlLINEAR = _hlu.NhlLINEAR
NhlIRREGULAR = _hlu.NhlIRREGULAR
NhlGEOGRAPHIC = _hlu.NhlGEOGRAPHIC
NhlTIME = _hlu.NhlTIME
NhlEUNKNOWN = _hlu.NhlEUNKNOWN
NhlENODATA = _hlu.NhlENODATA
NhlECONSTFIELD = _hlu.NhlECONSTFIELD
NhlEZEROFIELD = _hlu.NhlEZEROFIELD
NhlEZEROSPAN = _hlu.NhlEZEROSPAN
_NGGetNCARGEnv = _hlu._NGGetNCARGEnv

NhlInitialize = _hlu.NhlInitialize

NhlClose = _hlu.NhlClose

NhlRLClear = _hlu.NhlRLClear

NhlSetValues = _hlu.NhlSetValues

NhlRLSetString = _hlu.NhlRLSetString

NhlRLSetFloat = _hlu.NhlRLSetFloat

NhlRLSetDouble = _hlu.NhlRLSetDouble

NhlRLSetInteger = _hlu.NhlRLSetInteger

NhlNDCPolyline = _hlu.NhlNDCPolyline

NhlNDCPolymarker = _hlu.NhlNDCPolymarker

NhlNDCPolygon = _hlu.NhlNDCPolygon

NhlDataPolyline = _hlu.NhlDataPolyline

NhlDataPolymarker = _hlu.NhlDataPolymarker

NhlDataPolygon = _hlu.NhlDataPolygon

NhlDraw = _hlu.NhlDraw

NhlFreeColor = _hlu.NhlFreeColor

NhlGetGksCi = _hlu.NhlGetGksCi

NhlGetWorkspaceObjectId = _hlu.NhlGetWorkspaceObjectId

NhlIsAllocatedColor = _hlu.NhlIsAllocatedColor

NhlIsApp = _hlu.NhlIsApp

NhlIsDataComm = _hlu.NhlIsDataComm

NhlIsDataItem = _hlu.NhlIsDataItem

NhlIsDataSpec = _hlu.NhlIsDataSpec

NhlRLIsSet = _hlu.NhlRLIsSet

NhlRLUnSet = _hlu.NhlRLUnSet

NhlIsTransform = _hlu.NhlIsTransform

NhlIsView = _hlu.NhlIsView

NhlIsWorkstation = _hlu.NhlIsWorkstation

NhlName = _hlu.NhlName

NhlNewColor = _hlu.NhlNewColor

NhlNewDashPattern = _hlu.NhlNewDashPattern

NhlNewMarker = _hlu.NhlNewMarker

NhlSetColor = _hlu.NhlSetColor

NhlUpdateData = _hlu.NhlUpdateData

NhlUpdateWorkstation = _hlu.NhlUpdateWorkstation

NhlOpen = _hlu.NhlOpen

NhlCreate = _hlu.NhlCreate

NhlRLCreate = _hlu.NhlRLCreate

NhlFrame = _hlu.NhlFrame

NhlDestroy = _hlu.NhlDestroy

NhlRLSetMDIntegerArray = _hlu.NhlRLSetMDIntegerArray

NhlRLSetMDDoubleArray = _hlu.NhlRLSetMDDoubleArray

NhlRLSetMDFloatArray = _hlu.NhlRLSetMDFloatArray

NhlRLSetFloatArray = _hlu.NhlRLSetFloatArray

NhlRLSetIntegerArray = _hlu.NhlRLSetIntegerArray

NhlRLSetStringArray = _hlu.NhlRLSetStringArray

NhlGetValues = _hlu.NhlGetValues

NhlGetFloat = _hlu.NhlGetFloat

NhlGetFloatArray = _hlu.NhlGetFloatArray

NhlGetInteger = _hlu.NhlGetInteger

NhlGetIntegerArray = _hlu.NhlGetIntegerArray

NhlGetDouble = _hlu.NhlGetDouble

NhlGetDoubleArray = _hlu.NhlGetDoubleArray

NhlAddOverlay = _hlu.NhlAddOverlay

NhlClearWorkstation = _hlu.NhlClearWorkstation

NhlRemoveAnnotation = _hlu.NhlRemoveAnnotation

NhlAddAnnotation = _hlu.NhlAddAnnotation

NhlAppGetDefaultParentId = _hlu.NhlAppGetDefaultParentId

NhlGetParentWorkstation = _hlu.NhlGetParentWorkstation

NhlClassName = _hlu.NhlClassName

NhlGetString = _hlu.NhlGetString

NhlAddData = _hlu.NhlAddData

NhlRemoveData = _hlu.NhlRemoveData

NhlRemoveOverlay = _hlu.NhlRemoveOverlay

NhlGetStringArray = _hlu.NhlGetStringArray

NhlRLDestroy = _hlu.NhlRLDestroy

NhlGetNamedColorIndex = _hlu.NhlGetNamedColorIndex

NhlGetBB = _hlu.NhlGetBB

NhlChangeWorkstation = _hlu.NhlChangeWorkstation

NhlPGetBB = _hlu.NhlPGetBB

NhlPNDCToData = _hlu.NhlPNDCToData

NhlPDataToNDC = _hlu.NhlPDataToNDC

NhlGetMDFloatArray = _hlu.NhlGetMDFloatArray

NhlGetMDDoubleArray = _hlu.NhlGetMDDoubleArray

NhlGetMDIntegerArray = _hlu.NhlGetMDIntegerArray

NhlPAppClass = _hlu.NhlPAppClass

NhlPNcgmWorkstationClass = _hlu.NhlPNcgmWorkstationClass

NhlPXWorkstationClass = _hlu.NhlPXWorkstationClass

NhlPPSWorkstationClass = _hlu.NhlPPSWorkstationClass

NhlPPDFWorkstationClass = _hlu.NhlPPDFWorkstationClass

NhlPLogLinPlotClass = _hlu.NhlPLogLinPlotClass

NhlPGraphicStyleClass = _hlu.NhlPGraphicStyleClass

NhlPScalarFieldClass = _hlu.NhlPScalarFieldClass

NhlPContourPlotClass = _hlu.NhlPContourPlotClass

NhlPtextItemClass = _hlu.NhlPtextItemClass

NhlPscalarFieldClass = _hlu.NhlPscalarFieldClass

NhlPmapPlotClass = _hlu.NhlPmapPlotClass

NhlPcoordArraysClass = _hlu.NhlPcoordArraysClass

NhlPxyPlotClass = _hlu.NhlPxyPlotClass

NhlPtickMarkClass = _hlu.NhlPtickMarkClass

NhlPtitleClass = _hlu.NhlPtitleClass

NhlPlabelBarClass = _hlu.NhlPlabelBarClass

NhlPlegendClass = _hlu.NhlPlegendClass

NhlPvectorFieldClass = _hlu.NhlPvectorFieldClass

NhlPvectorPlotClass = _hlu.NhlPvectorPlotClass

NhlPstreamlinePlotClass = _hlu.NhlPstreamlinePlotClass

NGGetNCARGEnv = _hlu.NGGetNCARGEnv

set_PCMP04 = _hlu.set_PCMP04

gendat = _hlu.gendat

gactivate_ws = _hlu.gactivate_ws

gdeactivate_ws = _hlu.gdeactivate_ws

bndary = _hlu.bndary

c_plotif = _hlu.c_plotif

c_cpseti = _hlu.c_cpseti

c_cpsetr = _hlu.c_cpsetr

c_pcseti = _hlu.c_pcseti

c_pcsetr = _hlu.c_pcsetr

c_set = _hlu.c_set

c_cprect = _hlu.c_cprect

c_cpcldr = _hlu.c_cpcldr

c_plchhq = _hlu.c_plchhq

open_wks_wrap = _hlu.open_wks_wrap

labelbar_ndc_wrap = _hlu.labelbar_ndc_wrap

legend_ndc_wrap = _hlu.legend_ndc_wrap

contour_wrap = _hlu.contour_wrap

map_wrap = _hlu.map_wrap

contour_map_wrap = _hlu.contour_map_wrap

xy_wrap = _hlu.xy_wrap

y_wrap = _hlu.y_wrap

vector_wrap = _hlu.vector_wrap

vector_map_wrap = _hlu.vector_map_wrap

vector_scalar_wrap = _hlu.vector_scalar_wrap

vector_scalar_map_wrap = _hlu.vector_scalar_map_wrap

streamline_wrap = _hlu.streamline_wrap

streamline_map_wrap = _hlu.streamline_map_wrap

text_ndc_wrap = _hlu.text_ndc_wrap

text_wrap = _hlu.text_wrap

add_text_wrap = _hlu.add_text_wrap

maximize_plots = _hlu.maximize_plots

poly_wrap = _hlu.poly_wrap

add_poly_wrap = _hlu.add_poly_wrap

panel_wrap = _hlu.panel_wrap

mapgci = _hlu.mapgci

dcapethermo = _hlu.dcapethermo

draw_colormap_wrap = _hlu.draw_colormap_wrap

natgridc = _hlu.natgridc

ftcurvc = _hlu.ftcurvc

ftcurvpc = _hlu.ftcurvpc

ftcurvpic = _hlu.ftcurvpic

c_rgbhls = _hlu.c_rgbhls

c_hlsrgb = _hlu.c_hlsrgb

c_rgbhsv = _hlu.c_rgbhsv

c_hsvrgb = _hlu.c_hsvrgb

c_rgbyiq = _hlu.c_rgbyiq

c_yiqrgb = _hlu.c_yiqrgb

c_wmbarbp = _hlu.c_wmbarbp

c_wmsetip = _hlu.c_wmsetip

c_wmsetrp = _hlu.c_wmsetrp

c_wmsetcp = _hlu.c_wmsetcp

c_wmgetip = _hlu.c_wmgetip

c_wmgetrp = _hlu.c_wmgetrp

c_wmgetcp = _hlu.c_wmgetcp

c_nnseti = _hlu.c_nnseti

c_nnsetrd = _hlu.c_nnsetrd

c_nnsetc = _hlu.c_nnsetc

c_nngeti = _hlu.c_nngeti

c_nngetrd = _hlu.c_nngetrd

c_nngetcp = _hlu.c_nngetcp

c_dgcdist = _hlu.c_dgcdist

c_dcapethermo = _hlu.c_dcapethermo

c_dptlclskewt = _hlu.c_dptlclskewt

c_dtmrskewt = _hlu.c_dtmrskewt

c_dtdaskewt = _hlu.c_dtdaskewt

c_dsatlftskewt = _hlu.c_dsatlftskewt

c_dshowalskewt = _hlu.c_dshowalskewt

c_dpwskewt = _hlu.c_dpwskewt

pvoid = _hlu.pvoid

set_nglRes_i = _hlu.set_nglRes_i

get_nglRes_i = _hlu.get_nglRes_i

set_nglRes_f = _hlu.set_nglRes_f

get_nglRes_f = _hlu.get_nglRes_f

set_nglRes_c = _hlu.set_nglRes_c

get_nglRes_c = _hlu.get_nglRes_c

set_nglRes_s = _hlu.set_nglRes_s

get_nglRes_s = _hlu.get_nglRes_s

NglGaus_p = _hlu.NglGaus_p

NglVinth2p = _hlu.NglVinth2p


