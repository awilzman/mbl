% Read DICOM stacks of LCV, Time1, and Time2
% Rotate scans on user-defined AP axis and
% calculate intensity [unit] differences between t1 and t2 of
% bone volume (bv) [cm^3], bone mineral content (bmc) [g], and 
% bone mineral density [g/cm^3] (bmd) 
%               % Anterior-Medial   % Anterior-Lateral  % Posterior-Medial % Posterior-Lateral
% Scan 1        % (1,1)             % (1,2)             % (1,3)            % (1,4)
% Scan 2        % (2,1)             % (2,2)             % (2,3)            % (2,4)
% Difference    % (3,1)             % (3,2)             % (3,3)            % (3,4)
%
% Written by Andrew Wilzman and Karen Troy 06/2023
% Updated ARW 08/25/23
% Updated ARW 09/29/23
% Updated ARW 11/01/23
% Updated ARW 01/15/24
% eg
% default_directory = 'C:\Users\arwilzman\Documents'; or your user folder
% Z:\_Lab Personnel Folders\Andrew\Codes\Compare_Registered_DICOMs
% res = 82;
% LCV_name = 'LCV_example1';
% mask1_name = 'RAW1_example1';
% mask2_name = 'RAWROT2_example1';
% calibrate_slope = 0.00035619;
% calibrate_int = -0.00365584; % Z:\_SOPs\Scanner\Scanner Density Calibration Info
% [bv,bmc,bmd] = compare_dicoms(default_directory,res,LCV_name,mask1_name,mask2_name,calibrate_slope,calibrate_int)


% T. Hildebrand, A. Laib, R. Müller, J. Dequecker, P. Rüegsegger. 
% Direct 3-D morphometric analysis of human cancellous bone: 
% microstructural data from spine, femur, iliac crest and calcaneus. 
% J Bone Miner Res 1999;14(7):1167-74.
% MARA14 LCV volume = 6.743 cm^3
function [bv, bmc, bmd] = compare_dicoms(default_directory,res, ...
    LCV_name,mask1_name,mask2_name,calibrate_slope,calibrate_int, ...
    baseline,first_full_slice)

    %res in um, voxel edge length
    if nargin < 8
        baseline = 10;
        first_full_slice = 30;
    elseif nargin < 9
        first_full_slice = 30;
    end
    mask_LCV = get_mask(LCV_name,default_directory);
    mask_1 = get_mask(mask1_name,default_directory);
    mask_2 = get_mask(mask2_name,default_directory);
    
    % un-Pad matrices
    mask_LCV = pad_3dmat(mask_LCV);
    mask_1 = pad_3dmat(mask_1);
    mask_2 = pad_3dmat(mask_2);
    
    % Fill LCV by including all values that exist above a threshold in scans 1 and 2
    for i = 1:size(mask_LCV,3) 
        for j = 1:size(mask_LCV,1)
            for k = 1:size(mask_LCV,2)
                if mask_LCV(j,k,i) == 0
                    if mask_1(j,k,i) > baseline || mask_2(j,k,i) > baseline
                        mask_LCV(j,k,i) = 127;
                    end
                end
            end
        end
    end
    
    % View scans and choose directions
    graphfig = uifigure;
    figure(graphfig)
    colormap('hot');
    imagesc(mask_LCV(:,:,round(size(mask_LCV,3)/2)));
    axis image
    title('LCV')
    uialert(graphfig,'Please choose two points that represent the Anterior-Posterior axis', ...
        'Hello friend',Icon='info');
    movegui(graphfig,'east');
    [ap_x,ap_y] = ginput(2);
    
    % global origin is at top left of slice viewer, so ys are negated 
    ap_y = - ap_y;
    ap_slope = (ap_y(2)-ap_y(1))/(ap_x(2)-ap_x(1));
    close(graphfig);
    
    % Set origins, defined by geometric centroid (not density-weighted)
    origins = zeros(3,1);
    count = 0;
    for i = 1:size(mask_LCV,3) %z
        for j = 1:size(mask_LCV,1) %y
            for k = 1:size(mask_LCV,2) %x
                if mask_LCV(j,k,i) > 0
                    origins(1) = origins(1) + k;
                    origins(2) = origins(2) + j;
                    origins(3) = origins(3) + i;
                    count = count + 1;
                end
            end
        end
    end
    origins = origins / count;
    % rotate masks as defined
    angle = acot(ap_slope);
    if ap_y(1) < ap_y(2)
        angle = angle - pi();
    end
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
    angle = angle - (pi()/4); 
    mask_1 = rotate_mask(mask_1,angle,origins);
    mask_2 = rotate_mask(mask_2,angle,origins);
    mask_LCV = rotate_mask(mask_LCV,angle,origins);
    
    % View scan and choose medial side
    graphfig = uifigure;
    figure(graphfig)
    colormap('hot');
    imagesc(mask_LCV(:,:,round(size(mask_LCV,3)/2)));
    title('LCV')
    axis image
    uialert(graphfig,'Now click on the medial side', ...
        'almost there!',Icon='success');
    movegui(graphfig,'east');
    [med_x,] = ginput(1);    
    %Calculate the differences in masks, 
    % set any points outside of LCV to 0
    mask_diff = mask_2 - mask_1;
    min_original = min(mask_diff(:));
    max_original = max(mask_diff(:));
    desired_min = 0;
    desired_max = 255;
    normalized_matrix = (mask_diff - min_original) / (max_original - min_original) * (desired_max - desired_min) + desired_min;
    normalized_matrix(mask_diff == 0) = 127;
    normalized_matrix = max(min(normalized_matrix, 255), 0);
    
    mask_1 = uint16(mask_1);
    mask_2 = uint16(mask_2);
    mask_LCV = uint16(mask_LCV);
    mask_1 = pad_3dmat(mask_1);
    mask_2 = pad_3dmat(mask_2);
    mask_LCV = pad_3dmat(mask_LCV);
    normalized_matrix = pad_3dmat(normalized_matrix);
    
    if med_x < size(mask_LCV,2)
        medial_left = true;
    else medial_left = false; 
    end

    close all force
    figure()
    sliceViewer(mask_1)
    title('Scan 1')
    movegui('north');
    figure()
    sliceViewer(mask_2)
    title('Scan 2')
    movegui('south');
    figure()
    sliceViewer(normalized_matrix)
    title('Difference')
    movegui('east');
    
    % Calculate metrics
    % 4 sections, anterior/posterior and medial/lateral columns
    % 3 masks, time 1, 2, and difference (in order) rows
    bv = zeros(3,4);
    bmc = zeros(3,4);
    bmd = zeros(3,4);
    slices = size(mask_1, 3) - first_full_slice + 1;
    mask_1_med = zeros(size(mask_1, 1),size(mask_1, 2),slices);
    mask_1_ant = zeros(size(mask_1, 1),size(mask_1, 2),slices);
    mask_1_post = zeros(size(mask_1, 1),size(mask_1, 2),slices);
    mask_1_lat = zeros(size(mask_1, 1),size(mask_1, 2),slices);
    mask_2_med = zeros(size(mask_2, 1),size(mask_2, 2),slices);
    mask_2_ant = zeros(size(mask_2, 1),size(mask_2, 2),slices);
    mask_2_post = zeros(size(mask_2, 1),size(mask_2, 2),slices);
    mask_2_lat = zeros(size(mask_2, 1),size(mask_2, 2),slices);

    % Split up the masks based on anterior/posterior/medial/lateral
    if medial_left %medial in quadrant 2
        for z = 1:slices
            zs = z+(size(mask_1,3)-slices);
            y = round(origins(2));
            x = round(origins(1));

            mask_1_med(1:y,1:x,z) = mask_1(1:y,1:x,zs);
            mask_1_ant(1:y,x+1:end,z) = mask_1(1:y,x+1:end,zs);
            mask_1_post(1+y:end,1:x,z) = mask_1(1+y:end,1:x,zs);
            mask_1_lat(1+y:end,x+1:end,z) = mask_1(1+y:end,x+1:end,zs);

            mask_2_med(1:y,1:x,z) = mask_2(1:y,1:x,zs);
            mask_2_ant(1:y,x+1:end,z) = mask_2(1:y,x+1:end,zs);
            mask_2_post(1+y:end,1:x,z) = mask_2(1+y:end,1:x,zs);
            mask_2_lat(1+y:end,x+1:end,z) = mask_2(1+y:end,x+1:end,zs);
            
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
        for z = 1:slices
            zs = z+(size(mask_1,3)-slices);
            y = round(origins(2));
            x = round(origins(1));
            x_neg = find(mask_2(y+1,:,z) > 0,1,'first');
            x_pos = find(mask_2(y+1,:,z) > 0,1,'last');
            y_top = find(mask_2(:,x+1,z) > 0,1,'first');
            y_bot = find(mask_2(:,x+1,z) > 0,1,'last');

            mask_1_ant(1:y,x+1:end,z) = mask_1(1:y,x+1:end,zs);
            mask_1_post(1+y:end,1:x,z) = mask_1(1+y:end,1:x,zs);
            mask_1_med(1+y:end,x+1:end,z) = mask_1(1+y:end,x+1:end,zs);
            mask_1_lat(1:y,1:x,z) = mask_1(1:y,1:x,zs);

            mask_2_ant(1:y,x+1:end,z) = mask_2(1:y,x+1:end,zs);
            mask_2_post(1+y:end,1:x,z) = mask_2(1+y:end,1:x,zs);
            mask_2_med(1+y:end,x+1:end,z) = mask_2(1+y:end,x+1:end,zs);
            mask_2_lat(1:y,1:x,z) = mask_2(1:y,1:x,zs);

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
    
    [bv(1,1), bmc(1,1), bmd(1,1)] = bv_bmc(mask_1_ant,res,calibrate_slope,calibrate_int);
    [bv(1,2), bmc(1,2), bmd(1,2)] = bv_bmc(mask_1_post,res,calibrate_slope,calibrate_int);
    [bv(1,3), bmc(1,3), bmd(1,3)] = bv_bmc(mask_1_med,res,calibrate_slope,calibrate_int);
    [bv(1,4), bmc(1,4), bmd(1,4)] = bv_bmc(mask_1_lat,res,calibrate_slope,calibrate_int);
    [bv(2,1), bmc(2,1), bmd(2,1)] = bv_bmc(mask_2_ant,res,calibrate_slope,calibrate_int);
    [bv(2,2), bmc(2,2), bmd(2,2)] = bv_bmc(mask_2_post,res,calibrate_slope,calibrate_int);
    [bv(2,3), bmc(2,3), bmd(2,3)] = bv_bmc(mask_2_med,res,calibrate_slope,calibrate_int);
    [bv(2,4), bmc(2,4), bmd(2,4)] = bv_bmc(mask_2_lat,res,calibrate_slope,calibrate_int);
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

function [bv, bmc, bmd] = bv_bmc(mask, res, slope, int)
    bv = 0; 
    vox_ed = res / 10000.0; % um to cm
    for z = 1:size(mask, 3)
        slice = mask(:, :, z);
        area = calculateFilledArea(slice);
        % Check if the area is zero; if it is, skip this slice
        if area == 0
            disp('zero area slice detected');
            continue;
        end
        area = area * vox_ed^2;
        vol = area * vox_ed;
        bv = bv + vol; %get BV as well, this is TV
    end
    bmd = mean(mask(mask > 0)); % Mean of all values > 0
                                % can we get the total raw mask? 
                                % intersect raw DICOMs with BLCK instead of
                                % SEG
    count = nnz(mask > 0);
    occupied_vol = count * (vox_ed^3);
    bmd = bmd*slope + int;
    bmc = bmd * occupied_vol;
    bmd = bmc / bv;
end

function [new_mask] = rotate_mask(mask, rotationAngle, origins, interpolationMethod)
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
    origins(1) = origins(1) + pad_TB;
    origins(2) = origins(2) + pad_LR;
    for channelIndex = 1:size(mask, 3)
        matrix = double(mask(:, :, channelIndex));
        [rows, cols] = size(matrix);        
        [x, y] = meshgrid(1:cols, 1:rows);
        x = x - origins(1);
        y = y - origins(2);        
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