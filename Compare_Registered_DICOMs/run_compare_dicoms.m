clear %save data before starting
directory = 'C:\Users\arwilzman\Documents\'; %end slash is important
res = 82;
calibrate_slope = 0.000357;
calibrate_int = -0.0012625; 

subjects = ["MARA14-"];
LCV_name = 'LCV';
mask1_name = 'SCAN1';
mask2_name = 'SCAN2';
labels = ["File name";"Bone Volume [cm^3]";"Bone Mineral Content [g]";"Bone Mineral Density [g/cm^3]"];
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
    [output{2,i},output{3,i},output{4,i}] = compare_dicoms(directory, ...
        res,output{1,i},mask1s{i},mask2s{i},calibrate_slope,calibrate_int);
    for j=2:4
        output{j,i} = cat(1,columns,output{j,i});
        output{j,i} = cat(2,rows,output{j,i});
        output{j,i}(1,1) = labels(j);
    end
end
output = [labels,output];

% Export to Excel
excelFileName = 'output.xlsx';
for j = 2:length(subjects)+1
    writematrix(output{1, j}, excelFileName, 'Sheet', 1, 'Range', ['A' num2str(6*j-5)]);
    for i = 2:4
        writematrix(output{i, j}, excelFileName, 'Sheet', 1, 'Range', ['A' num2str(6*j-7+(i-1)*4) ':' 'E' num2str(6*j-3+(i-1)*4)]);
    end
end