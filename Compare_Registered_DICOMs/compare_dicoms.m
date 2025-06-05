% Read DICOM stacks of LCV, Time1, and Time2, and raw image of time1
% raw image time1 is reference for registering time2 earlier
% Rotate scans on user-defined AP axis and
% calculate intensity [unit] differences between t1 and t2 of
% bone volume (bv) [cm^3], bone mineral content (bmc) [g], and 
% bone mineral density [g/cm^3] (bmd) 
%               % Anterior          % Posterior         % Medial           % Lateral
% Scan 1        % (1,1)             % (1,2)             % (1,3)            % (1,4)
% Scan 2        % (2,1)             % (2,2)             % (2,3)            % (2,4)
% Difference    % (3,1)             % (3,2)             % (3,3)            % (3,4)
%
% Written by Andrew Wilzman and Karen Troy 06/2023
% Example run:
% calibrate_slope = 0.00035619;
% calibrate_int = -0.00365584; 
%res in um, voxel edge length
% [bv,bmc,bmd,medial_left,angle_rot] = compare_dicoms(default_directory,res,LCV_name,
% mask1_name,mask2_name,calibrate_slope,calibrate_int,
% medial_left,angle_rot,first_full_slice)

% T. Hildebrand, A. Laib, R. Müller, J. Dequecker, P. Rüegsegger. 
% Direct 3-D morphometric analysis of human cancellous bone: 
% microstructural data from spine, femur, iliac crest and calcaneus. 
% J Bone Miner Res 1999;14(7):1167-74.
function [tv, bv, bmc, bmd, medial_left, angle_rot] = compare_dicoms(default_directory,res, ...
    LCV_name,mask1_name,mask2_name,calibrate_slope,calibrate_int, ...
    medial_left,angle_rot)

    if nargin < 9
        angle_rot = 0;
    end
    if nargin < 8
        medial_left = 0;
    end

    mask_LCV = get_mask(LCV_name,default_directory);
    mask_1 = get_mask(mask1_name,default_directory);
    mask_2 = get_mask(mask2_name,default_directory);
    
    % un-Pad matrices
    mask_LCV = pad_3dmat(mask_LCV);
    mask_1 = pad_3dmat(mask_1);
    mask_2 = pad_3dmat(mask_2);
    
    idx = mask_LCV == 0;
    mask_1(idx) = 0;
    mask_2(idx) = 0;
    tangle = pi()/4;

    if angle_rot==0
        %threshold image
        %classify tibia and fibula by size (larger = tibia)
        %define centers of each as above
        %calculate the angle of the line with respect to the X axis CCW+
        dicom_files = dir(fullfile(strcat(chdir, '\', mask1_name, '_raw\'), '*.dcm'));
        dir_path = strcat(chdir,'\',mask1_name,'_raw\');
        % Check if any DICOM files exist and select the first one
        if ~isempty(dicom_files)
            d_name = dicom_files(1).name;  % First DICOM file
            full_dicom_path = fullfile(dir_path, d_name);  % Full path to DICOM file
        else
            error('No DICOM files found in the directory.');
        end
        
        raw_image = dicomread(full_dicom_path);
        raw_image_b = raw_image > 2 / calibrate_slope;
        raw_image_b = logical(raw_image_b);
        stats = regionprops(raw_image_b, 'Area', 'Centroid');
        min_area_threshold = 500;
        stats = stats([stats.Area] > min_area_threshold);
        areas = [stats.Area];
        [~, idx_tibia] = max(areas);
        [~, idx_fibula] = min(areas);
        tibia_centroid = stats(idx_tibia).Centroid; 
        fibula_centroid = stats(idx_fibula).Centroid;
        delta_y = tibia_centroid(2) - fibula_centroid(2);  % Y difference (rows)
        delta_x = tibia_centroid(1) - fibula_centroid(1);  % X difference (columns)
        % Angle calculation
        angle_rot = atan2(delta_y, delta_x)-tangle+pi();
        % Medial side calculation
        if tibia_centroid(1) < fibula_centroid(1)
            medial_left = true;
        else
            medial_left = false;
        end
    end

    z_center = round(size(mask_LCV, 3) / 2); % Middle slice index
    center_slice = mask_LCV(:, :, z_center); % 2D slice along Z axis
    non_zero_voxels_2D = center_slice > 0;
    filled_slice = imfill(non_zero_voxels_2D, 'holes');
    [rows, cols] = find(filled_slice); 
    
    x = mean(cols);
    y = mean(rows);
    circle_center = [x, y];
    % Rotate mask to set anterior in quadrant 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                    #
    %                    #
    %                    #
    % Medial/Lateral     #  Anterior
    % ########################################
    %         Posterior  #  Medial/Lateral
    %                    #
    %                    #
    %                    #
    %                    #
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    mask_1 = rotate_mask(mask_1,angle_rot,circle_center);
    mask_2 = rotate_mask(mask_2,angle_rot,circle_center);
    
    mask_1 = int16(mask_1);
    mask_2 = int16(mask_2);
    mask_1 = pad_3dmat(mask_1);
    mask_2 = pad_3dmat(mask_2);

    slice_image = mask_1(:,:,round(size(mask_1,3)/2));
    [rows, cols] = size(slice_image);
    [xGrid, yGrid] = meshgrid(1:cols, 1:rows);
    ap_slope = tan(tangle);
    line_mask = abs(yGrid - (ap_slope * (xGrid - x) + y)) < 1;
    slice_image(line_mask) = max(slice_image(:));
    line_mask = abs(yGrid - ((1/(-ap_slope + 1e-6)) * (xGrid - x) + y)) < 1;
    slice_image(line_mask) = max(slice_image(:));

    % Calculate metrics
    % 4 sections, anterior/posterior and medial/lateral columns
    % 3 masks, time 1, 2, and difference (in order) rows
    tv = zeros(3,4);
    bv = zeros(3,4);
    bmc = zeros(3,4);
    bmd = zeros(3,4);
    mask_1_med = zeros(size(mask_1, 1),size(mask_1, 2),size(mask_1,3));
    mask_1_ant = zeros(size(mask_1, 1),size(mask_1, 2),size(mask_1,3));
    mask_1_post = zeros(size(mask_1, 1),size(mask_1, 2),size(mask_1,3));
    mask_1_lat = zeros(size(mask_1, 1),size(mask_1, 2),size(mask_1,3));
    mask_2_med = zeros(size(mask_2, 1),size(mask_2, 2),size(mask_1,3));
    mask_2_ant = zeros(size(mask_2, 1),size(mask_2, 2),size(mask_1,3));
    mask_2_post = zeros(size(mask_2, 1),size(mask_2, 2),size(mask_1,3));
    mask_2_lat = zeros(size(mask_2, 1),size(mask_2, 2),size(mask_1,3));
    
    % Split up the masks based on anterior/posterior/medial/lateral
    x = uint16(x);
    y = uint16(y);
    if medial_left %medial in quadrant 2
        for z = 1:size(mask_1,3)
            mask_1_med(1:y,1:x,z) = mask_1(1:y,1:x,z);
            mask_1_ant(1:y,x+1:end,z) = mask_1(1:y,x+1:end,z);
            mask_1_post(1+y:end,1:x,z) = mask_1(1+y:end,1:x,z);
            mask_1_lat(1+y:end,x+1:end,z) = mask_1(1+y:end,x+1:end,z);

            mask_2_med(1:y,1:x,z) = mask_2(1:y,1:x,z);
            mask_2_ant(1:y,x+1:end,z) = mask_2(1:y,x+1:end,z);
            mask_2_post(1+y:end,1:x,z) = mask_2(1+y:end,1:x,z);
            mask_2_lat(1+y:end,x+1:end,z) = mask_2(1+y:end,x+1:end,z);
            
            x_neg = find(mask_2(y+1,:,z) > 0,1,'first');
            x_pos = find(mask_2(y+1,:,z) > 0,1,'last');
            y_top = find(mask_2(:,x+1,z) > 0,1,'first');
            y_bot = find(mask_2(:,x+1,z) > 0,1,'last');
            % Fill 1s so that the area is enclosed by boundary function
            mask_1_med(y+1,x_neg:x,z) = 1;
            mask_1_med(y_top:y,x+1,z) = 1;

            mask_1_ant(y+1,x+1:x_pos,z) = 1;
            mask_1_ant(y_top:y,x,z) = 1;

            mask_1_post(y+1,x_neg:x,z) = 1;
            mask_1_post(1+y:y_bot,x+1,z) = 1;

            mask_1_lat(y+1,1+x:x_pos,z) = 1;
            mask_1_lat(1+y:y_bot,x+1,z) = 1;
            
            mask_2_med(y+1,x_neg:x,z) = 1;
            mask_2_med(y_top:y,x+1,z) = 1;

            mask_2_ant(y+1,x+1:x_pos,z) = 1;
            mask_2_ant(y_top:y,x,z) = 1;

            mask_2_post(y+1,x_neg:x,z) = 1;
            mask_2_post(1+y:y_bot,x+1,z) = 1;

            mask_2_lat(y+1,1+x:x_pos,z) = 1;
            mask_2_lat(1+y:y_bot,x+1,z) = 1;
        end
    else %lateral in quadrant 2
        for z = 1:size(mask_1,3)
            x_neg = find(mask_2(y+1,:,z) > 0,1,'first');
            x_pos = find(mask_2(y+1,:,z) > 0,1,'last');
            y_top = find(mask_2(:,x+1,z) > 0,1,'first');
            y_bot = find(mask_2(:,x+1,z) > 0,1,'last');

            mask_1_ant(1:y,x+1:end,z) = mask_1(1:y,x+1:end,z);
            mask_1_post(1+y:end,1:x,z) = mask_1(1+y:end,1:x,z);
            mask_1_med(1+y:end,x+1:end,z) = mask_1(1+y:end,x+1:end,z);
            mask_1_lat(1:y,1:x,z) = mask_1(1:y,1:x,z);

            mask_2_ant(1:y,x+1:end,z) = mask_2(1:y,x+1:end,z);
            mask_2_post(1+y:end,1:x,z) = mask_2(1+y:end,1:x,z);
            mask_2_med(1+y:end,x+1:end,z) = mask_2(1+y:end,x+1:end,z);
            mask_2_lat(1:y,1:x,z) = mask_2(1:y,1:x,z);

            mask_1_lat(y+1,x_neg:x,z) = 1;
            mask_1_lat(y_top:y,x+1,z) = 1;

            mask_1_ant(y+1,x+1:x_pos,z) = 1;
            mask_1_ant(y_top:y,x,z) = 1;

            mask_1_post(y+1,x_neg:x,z) = 1;
            mask_1_post(1+y:y_bot,x+1,z) = 1;

            mask_1_med(y+1,1+x:x_pos,z) = 1;
            mask_1_med(1+y:y_bot,x+1,z) = 1;

            mask_2_lat(y+1,x_neg:x,z) = 1;
            mask_2_lat(y_top:y,x+1,z) = 1;

            mask_2_ant(y+1,x+1:x_pos,z) = 1;
            mask_2_ant(y_top:y,x,z) = 1;

            mask_2_post(y+1,x_neg:x,z) = 1;
            mask_2_post(1+y:y_bot,x+1,z) = 1;

            mask_2_med(y+1,1+x:x_pos,z) = 1;
            mask_2_med(1+y:y_bot,x+1,z) = 1;
        end
    end
    
    [tv(1,1), bv(1,1), bmc(1,1), bmd(1,1)] = bv_bmc(mask_1_ant,res,calibrate_slope,calibrate_int);
    [tv(1,2), bv(1,2), bmc(1,2), bmd(1,2)] = bv_bmc(mask_1_post,res,calibrate_slope,calibrate_int);
    [tv(1,3), bv(1,3), bmc(1,3), bmd(1,3)] = bv_bmc(mask_1_med,res,calibrate_slope,calibrate_int);
    [tv(1,4), bv(1,4), bmc(1,4), bmd(1,4)] = bv_bmc(mask_1_lat,res,calibrate_slope,calibrate_int);
    [tv(2,1), bv(2,1), bmc(2,1), bmd(2,1)] = bv_bmc(mask_2_ant,res,calibrate_slope,calibrate_int);
    [tv(2,2), bv(2,2), bmc(2,2), bmd(2,2)] = bv_bmc(mask_2_post,res,calibrate_slope,calibrate_int);
    [tv(2,3), bv(2,3), bmc(2,3), bmd(2,3)] = bv_bmc(mask_2_med,res,calibrate_slope,calibrate_int);
    [tv(2,4), bv(2,4), bmc(2,4), bmd(2,4)] = bv_bmc(mask_2_lat,res,calibrate_slope,calibrate_int);
    tv(3,1) = tv(2,1)-tv(1,1);
    tv(3,2) = tv(2,2)-tv(1,2);
    tv(3,3) = tv(2,3)-tv(1,3);
    tv(3,4) = tv(2,4)-tv(1,4);
    bv(3,1) = bv(2,1)-bv(1,1);
    bv(3,2) = bv(2,2)-bv(1,2);
    bv(3,3) = bv(2,3)-bv(1,3);
    bv(3,4) = bv(2,4)-bv(1,4);
    bmc(3,1) = bmc(2,1)-bmc(1,1);
    bmc(3,2) = bmc(2,2)-bmc(1,2);
    bmc(3,3) = bmc(2,3)-bmc(1,3);
    bmc(3,4) = bmc(2,4)-bmc(1,4);
    bmd(3,1) = bmd(2,1)-bmd(1,1);
    bmd(3,2) = bmd(2,2)-bmd(1,2);
    bmd(3,3) = bmd(2,3)-bmd(1,3);
    bmd(3,4) = bmd(2,4)-bmd(1,4);

    slice_image = uint16(slice_image);
    imwrite(slice_image,strcat(default_directory,mask1_name,".png"))

    color_ant = [255, 0, 0];   % Red for anterior
    color_post = [0, 255, 0];  % Green for posterior
    color_med = [0, 0, 255];   % Blue for medial
    color_lat = [255, 255, 0]; % Yellow for lateral
    slice_image_rgb = zeros([size(slice_image), 3], 'uint8');
    
    for z = 1:size(mask_1, 3)
        % Anterior section
        slice_image_rgb(:,:,1) = slice_image_rgb(:,:,1) + uint8(mask_1_ant(:,:,z)) * color_ant(1);
        slice_image_rgb(:,:,2) = slice_image_rgb(:,:,2) + uint8(mask_1_ant(:,:,z)) * color_ant(2);
        slice_image_rgb(:,:,3) = slice_image_rgb(:,:,3) + uint8(mask_1_ant(:,:,z)) * color_ant(3);
    
        % Posterior section
        slice_image_rgb(:,:,1) = slice_image_rgb(:,:,1) + uint8(mask_1_post(:,:,z)) * color_post(1);
        slice_image_rgb(:,:,2) = slice_image_rgb(:,:,2) + uint8(mask_1_post(:,:,z)) * color_post(2);
        slice_image_rgb(:,:,3) = slice_image_rgb(:,:,3) + uint8(mask_1_post(:,:,z)) * color_post(3);
    
        % Medial section
        slice_image_rgb(:,:,1) = slice_image_rgb(:,:,1) + uint8(mask_1_med(:,:,z)) * color_med(1);
        slice_image_rgb(:,:,2) = slice_image_rgb(:,:,2) + uint8(mask_1_med(:,:,z)) * color_med(2);
        slice_image_rgb(:,:,3) = slice_image_rgb(:,:,3) + uint8(mask_1_med(:,:,z)) * color_med(3);
    
        % Lateral section
        slice_image_rgb(:,:,1) = slice_image_rgb(:,:,1) + uint8(mask_1_lat(:,:,z)) * color_lat(1);
        slice_image_rgb(:,:,2) = slice_image_rgb(:,:,2) + uint8(mask_1_lat(:,:,z)) * color_lat(2);
        slice_image_rgb(:,:,3) = slice_image_rgb(:,:,3) + uint8(mask_1_lat(:,:,z)) * color_lat(3);
    end
    
    imwrite(slice_image_rgb, strcat(default_directory, mask1_name, '_segmented.png'));
    % Red anterior
    % Green Posterior
    % Blue Medial
    % Yellow Lateral
end

% Function definitions
function totalArea = calculateFilledArea(array)
    % Find the contour of the binary image
    binary_image = array > 0;
    % Perform morphological operations to connect gaps
    se = strel('disk', 20);  % Adjust the size as needed
    closed_image = imclose(binary_image, se);
    contour = bwboundaries(closed_image);
    % Initialize the total area
    totalArea = 0;
    % Iterate through each contour and add its area to the total
    for k = 1:length(contour)
        boundary = contour{k};
        totalArea = totalArea + polyarea(boundary(:, 2), boundary(:, 1));
    end
end

function [tv, bv, bmc, bmd] = bv_bmc(mask, res, slope, int)
    tv = 0;    
    bv = 0;
    bmd = 0;
    bmc = 0;
    
    vox_ed = res / 10000.0; % um to cm
    for z = 1:size(mask, 3)
        slice = mask(:, :, z);
        area = calculateFilledArea(slice);
        % Check if the area is zero; if it is, skip this slice
        if area == 0
            continue;
        end
        area = area * vox_ed^2;
        vol = area * vox_ed;
        bv_vol = sum(sum(slice>1))*vox_ed^3;
        if (bv_vol == 0)
            continue
        end

        slice_density = mean(mean(slice(slice>1))) * slope + int;
        slice_content = slice_density * bv_vol;
        slice_density = slice_content / vol;

        tv = tv + vol; %Total Volume
        bv = bv + bv_vol; %Bone Volume
        bmd = bmd + slice_density;
        bmc = bmc + slice_content;
    end
    bmd = bmd / size(mask,3);
end

function [new_mask] = rotate_mask(mask, rotationAngle, centroid, interpolationMethod)
    % Input validation
    assert(isnumeric(mask) && ndims(mask) >= 2, 'Input mask must be a numeric 2D or 3D array.');
    assert(isscalar(rotationAngle), 'Rotation angle must be a scalar.');
    
    % Set default interpolation method
    if nargin < 4
        interpolationMethod = 'linear';
    end
    % Precompute values
    max_length = round(1.2 * sqrt(size(mask, 1)^2 + size(mask, 2)^2));
    pad_LR = round((max_length - size(mask, 2)) / 2);
    pad_TB = round((max_length - size(mask, 1)) / 2);
    % Pad the mask
    mask = padarray(mask, [pad_TB, pad_LR], 0, 'both');
    % Initialize the result
    new_mask = zeros(size(mask));    
    centroid(1) = centroid(1) + pad_TB;
    centroid(2) = centroid(2) + pad_LR;
    for channelIndex = 1:size(mask, 3)
        matrix = double(mask(:, :, channelIndex));
        [rows, cols] = size(matrix);        
        [x, y] = meshgrid(1:cols, 1:rows);
        x = x - centroid(1);
        y = y - centroid(2);        
        x_rot = x * cos(rotationAngle) - y * sin(rotationAngle);
        y_rot = x * sin(rotationAngle) + y * cos(rotationAngle);        
        % Perform interpolation with specified method
        new_mask(:, :, channelIndex) = interp2(x, y, matrix, x_rot, y_rot, interpolationMethod);
    end
end

function [red_mask] = pad_3dmat(mask)
    inds = zeros(2,3);
    inds(1,1) = pad_dim(1,mask,1);
    inds(1,2) = pad_dim(2,mask,1);
    inds(1,3) = pad_dim(3,mask,1);
    inds(2,1) = pad_dim(1,mask,-1);
    inds(2,2) = pad_dim(2,mask,-1);
    inds(2,3) = pad_dim(3,mask,-1);
    red_mask = mask(inds(1,1):end-inds(2,1),inds(1,2):end-inds(2,2),inds(1,3):end-inds(2,3));
end

function [mask] = get_mask(name,default_directory)
    chdir = strcat(default_directory,name);
    files = dir(fullfile(chdir,'*.DCM*'));
    slices = length(files);
    mask = dicomread(strcat(chdir,'\',files(1).name));
    mask(end,end,slices)=0; %pre allocate
    for i = 2:slices
        mask(:,:,i) = dicomread(strcat(chdir,'\',files(i).name));
    end
end

function [ind] = pad_dim(dimension,mask,direction)
    if direction > 0
        if dimension == 3
            for i = 1:size(mask,dimension)
                if sum(mask(:,:,i),'all') == 0
                    continue
                end
                ind = i;
                break
            end
        elseif dimension == 2
            for i = 1:size(mask,dimension)
                if sum(mask(:,i,:),'all') == 0
                    continue
                end
                ind = i;
                break
            end 
        elseif dimension == 1
            for i = 1:size(mask,dimension)
                if sum(mask(i,:,:),'all') == 0
                    continue
                end
                ind = i;
                break
            end
        end
    else 
        if dimension == 3
            for i = 1:size(mask,dimension)
                if sum(mask(:,:,end-(i-1)),'all') == 0
                    continue
                end
                ind = i;
                break
            end
        elseif dimension == 2
            for i = 1:size(mask,dimension)
                if sum(mask(:,end-(i-1),:),'all') == 0
                    continue
                end
                ind = i;
                break
            end 
        elseif dimension == 1
            for i = 1:size(mask,dimension)
                if sum(mask(end-(i-1),:,:),'all') == 0
                    continue
                end
                ind = i;
                break
            end
        end
    end
end
