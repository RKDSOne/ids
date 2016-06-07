Sub Top10()
    For Row = 2 To 12
        Dim irow As String
        irow = "B" & Row & ":" & "F" & Row
        Range(irow).Select
        Selection.FormatConditions.AddTop10
        Selection.FormatConditions(Selection.FormatConditions.Count).SetFirstPriority
        With Selection.FormatConditions(1)
            .TopBottom = xlTop10Top
            .Rank = 10
            .Percent = True
        End With
        With Selection.FormatConditions(1).Font
            .Color = -16383844
            .TintAndShade = 0
        End With
        With Selection.FormatConditions(1).Interior
            .PatternColorIndex = xlAutomatic
            .Color = 13551615
            .TintAndShade = 0
        End With
        Selection.FormatConditions(1).StopIfTrue = False
    Next
End Sub