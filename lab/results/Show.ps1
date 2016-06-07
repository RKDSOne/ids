function Get-Curdir
{
    return (Get-Item -Path ".\" -Verbose).FullName
}

Copy-Item pre.csv table.csv
$curdir = Get-Curdir
$fname = ("{0}\table.csv" -f $curdir)

# convert to macro-enabled xls
$xl = New-Object -ComObject Excel.Application
$wb = $xl.Workbooks.Open($fname)
$wb.SaveAs($fname.Replace('.csv', '.xlsm'), 52)
$xl.Quit()

# run macro to find top10
$xl = New-Object -ComObject Excel.Application
# $xl.Visible = $true
$wb = $xl.Workbooks.Open($fname.Replace('.csv', '.xlsm'))
$xlmodule = $wb.VBProject.VBComponents.Add(1)

$code = @"
Sub Macro1()
    For i = 2 To 12
        Dim irow
        irow = "B" & i & ":" & "F" & i
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
"@

$xlmodule.CodeModule.AddFromString($code)
$app = $xl.Application
$app.Run("Macro1")
$wb.Save()
$xl.Visible = $true
