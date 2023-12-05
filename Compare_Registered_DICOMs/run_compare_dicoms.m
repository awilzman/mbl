directory = 'C:\Users\Andrew\Desktop\Compare_Registered_DICOMs\'; %end slash is important
res = 82;
calibrate_slope = 0.000357;
calibrate_int = -0.0012625;
% if I had 10 files 
numfiles=1;

LCV_name = 'LCV_example';
mask1_name = 'RAW1_example';
mask2_name = 'RAWROT2_example';
labels = ["File name";"Bone Volume [cm^3]";"Bone Mineral Content [g]";"Bone Mineral Density [g/cm^3]"];
labels = cellstr(labels);

for i=1:numfiles
    output{i} = strcat(LCV_name,num2str(i));
    mask1s{i} = strcat(mask1_name,num2str(i));
    mask2s{i} = strcat(mask2_name,num2str(i));
end
columns = ["Anterior-Medial","Anterior-Lateral","Posterior-Medial","Posterior-Lateral"];
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