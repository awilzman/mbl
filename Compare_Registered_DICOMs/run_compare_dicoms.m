clear %save data before starting
directory = ''; %end slash is important
res = 82;
calibrate_slope = 0.000357;
calibrate_int = -0.0012625; 

subjects = ["MARA14-"];
LCV_name = 'LCV';
mask1_name = 'SCAN1';
mask2_name = 'SCAN2';
labels = ["File name";"Total Volume [cm^3]";"Bone Volume [cm^3]";"Bone Mineral Content [g]";"Bone Mineral Density [g/cm^3]"];
labels = cellstr(labels);

% this code requires each lcv, mask1, and mask2 to be named the same with 
% an incrementing suffix. this could be changed to feed from a list of file
% names
for i=1:length(subjects)
    output{i} = strcat(subjects(i),LCV_name);
    mask1s{i} = strcat(subjects(i),mask1_name);
    mask2s{i} = strcat(subjects(i),mask2_name);
end
columns = ["Anterior","Posterior","Medial","Lateral"];
rows = ["";"Scan 1";"Scan 2";"Difference"];
for i=1:length(output)
    [output{2,i},output{3,i},output{4,i},output{5,i}] = compare_dicoms(directory, ...
        res,output{1,i},mask1s{i},mask2s{i},calibrate_slope,calibrate_int);
    for j=2:5
        output{j,i} = cat(1,columns,output{j,i});
        output{j,i} = cat(2,rows,output{j,i});
        output{j,i}(1,1) = labels(j);
    end
end
output = [labels,output];

% Export to Excel
excelFileName = 'output.xlsx';
% Loop through each subject
for j = 2:length(subjects)+1
    % Write the file name
    writematrix(output{1, 2}, excelFileName, 'Sheet', 1, 'Range', ['A' num2str(18*(j-2)+1)]);
    for i = 2:5
        % Calculate the starting row for this data block
        startRow = (18 * (j - 2)) + (i - 1) * 4;
        label = output{i,1};
        dataMatrix = output{i,2};
        writematrix(label, excelFileName, 'Sheet', 1, 'Range', ['A' num2str(startRow)]);
        writematrix(dataMatrix, excelFileName, 'Sheet', 1, 'Range', ['B' num2str(startRow)]);
    end
end