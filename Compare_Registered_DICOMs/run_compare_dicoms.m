clear;  % Clear workspace

% Initial Setup
subs = ["15"];  % List of subjects
directory = '';  % Directory of DICOM files (end slash is important)
res = 82;  % Image resolution (in micrometers)
calibrate_slope = 0.000357;
calibrate_int = -0.0012625;
excelFileName = 'output.xlsx';  % Output file name

columns = ["Anterior", "Posterior", "Medial", "Lateral"]; 
rows = [""; "Scan 1"; "Scan 2"; "Difference"];
labels = cellstr(["File name"; "Total Volume [cm^3]"; "Bone Volume [cm^3]"; ...
    "Bone Mineral Content [g]"; "Bone Mineral Density [g/cm^3]"]);

% Initialize
output = cell(length(subs), 2, 5);  % Preallocate for output
mask1s = cell(length(subs), 2);  % Preallocate for mask1s
mask2s = cell(length(subs), 2);  % Preallocate for mask2s

% Loop through subjects
medial_left = 0;
angle_rot = 0;
for s = 1:length(subs)
    subjects = strcat(subs(s), [" 4 ", " 30 "]);
    
    % Prepare file names and compare DICOMs
    for i = 1:length(subjects)
        output{s, i, 1} = strcat(subjects(i), 'lcv');  % Scan name
        mask1s{s, i} = strcat(subjects(i), 'scan1');   % Mask 1 name
        mask2s{s, i} = strcat(subjects(i), 'scan2');   % Mask 2 name

        disp("Running " + subjects(i));  % Log current subject
        
        % Compare DICOMs and store results
        [output{s, i, 2}, output{s, i, 3}, output{s, i, 4}, output{s, i, 5}] = ...
            compare_dicoms(directory, res, output{s, i, 1}, mask1s{s, i}, mask2s{s, i}, ...
            calibrate_slope, calibrate_int, medial_left, angle_rot);
    end
end

allData = {};  % Initialize allData for Excel output

% Save results to Excel
for s = 1:length(subs)  % Loop over subjects
    for i = 1:size(output, 2)  % Number of scans
        baseRow = 36 * (s - 1) + 18 * (i - 1) + 1; 
        for j = 2:size(output, 3)  % Loop over features
            startRow = baseRow + 4 * (j - 2);
            allData(startRow, 2:5) = labels(2:end);  % Store labels
            dataMatrix = output{s, i, j};  % Access the data 
            [numDataRows, numDataCols] = size(dataMatrix); 
            for r = 1:numDataRows
                allData(startRow + r, 2:5) = num2cell(dataMatrix(r, :));  % Fill in data
            end
        end
        allData{baseRow, 1} = output{s, i, 1};  % Store subject info
        allData{baseRow + 1} = 'Anterior';
        allData{baseRow + 5} = 'Posterior';
        allData{baseRow + 9} = 'Medial';
        allData{baseRow + 13} = 'Lateral';
    end
end

% Write the data to Excel, overwriting if it exists
writecell(allData, excelFileName, 'Sheet', 1, 'WriteMode', 'overwrite');  
