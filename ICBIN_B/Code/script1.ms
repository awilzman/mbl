-- Save OBJ File
fn saveOBJ obj filePath =
(
    filePath = filePath + ".obj"

    -- Export the file as OBJ
    local objExport = exportFile filePath #noPrompt selectedOnly:true  -- Exports only the selected object
    
    if objExport == false then
    (
        print ("Error: Unable to export OBJ to " + filePath)
        return false
    )
    
    print ("OBJ exported successfully to: " + filePath)
    return true
)

fn ensureEditablePoly obj =
(
    if isValidNode obj and not (isKindOf obj Editable_Poly) do convertToPoly obj
    return obj
)

-- Refine Mesh
fn refineMesh obj maxEdgeLength =
(
    obj = ensureEditablePoly obj
    local subdivMod = Subdivide()
    subdivMod.size = maxEdgeLength
    addModifier obj subdivMod
    return obj
)

fn applyRelaxation obj iterations =
(
    obj = ensureEditablePoly obj
    if obj == undefined then
    (
        print "Error: Object is not valid for relaxation!"
        return false
    )

    local relaxMod = Relax()
    relaxMod.iterations = iterations
    addModifier obj relaxMod
    return obj
)

-- Process OBJ Files
fn processOBJFiles inputFolder modelName outputFolder relaxIterations maxEdgeLength =
(
    -- Construct the full path to the model folder
    local modelFolderPath = inputFolder + "\\" + modelName
    local objFiles = getFiles (modelFolderPath + "\\*.stl")  -- Still importing STL but exporting as OBJ
    
    if objFiles.count == 0 then
    (
        print ("No STL files found in folder: " + modelFolderPath)
        return false
    )

    for filePath in objFiles do
    (
        print ("Processing: " + filePath)

        -- Import STL
        obj = importFile filePath #noPrompt
        obj = selection[1]
		
		obj = ensureEditablePoly obj

        -- Apply Relaxation and Refinement
        obj = applyRelaxation obj relaxIterations
        obj = refineMesh obj maxEdgeLength

        -- Save the refined mesh as OBJ
        local outputFileName = getFileNameFile filePath
        local outputFilePath = outputFolder + "\\" + modelName + "\\" + outputFileName
        makeDir (outputFolder + "\\" + modelName)
		obj.material = undefined
        saveOBJ obj outputFilePath

        -- Cleanup
        delete obj
    )
    return true
)

-- Parameters
inputFolderPath = @"Z:\_PROJECTS\Deep_Learning_HRpQCT\ICBIN_B\Data\Generated\unprocessed"
outputFolderPath = @"Z:\_PROJECTS\Deep_Learning_HRpQCT\ICBIN_B\Data\Generated\tet4"
modelName = "gangen_lrg_fold_512_128_256"  -- The name of the nested model folder
relaxIterations = 5
maxEdgeLength = 2

-- Process OBJ Files
processOBJFiles inputFolderPath modelName outputFolderPath relaxIterations maxEdgeLength
