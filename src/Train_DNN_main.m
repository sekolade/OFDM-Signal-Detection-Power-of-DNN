% Load and combine multiple OFDM training/validation datasets, then train DNN models

% List of dataset files containing mixed-SNR OFDM feature/target cells
files = {
    'savednets/OFDM_Dataset_SNR_Mixed1(rand,not16,50k-200k).mat', ...
    'savednets/OFDM_Dataset_SNR_Mixed2(rand,not16,50k-200k).mat', ...
    'savednets/OFDM_Dataset_SNR_Mixed3(rand,not16,50k-200k).mat', ...
    'savednets/OFDM_Dataset_SNR_Mixed4(rand,not16,50k-200k).mat'
};
disp("OK: Dataset file list prepared")

% Preallocate empty cell arrays to accumulate training and validation data
Xtraining_combined   = {};  % will hold feature cells for training
Ytraining_combined   = {};  % will hold target cells for training
Xvalidation_combined = {};  % will hold feature cells for validation
Yvalidation_combined = {};  % will hold target cells for validation

% Iterate over each file and append its data to the combined lists
for i = 1:length(files)
    data = load(files{i});  
    % Each .mat file contains cell arrays named:
    %   Xtraining_Cells, Ytraining_regression_cells,
    %   Xvalidation_Cells, Yvalidation_regression
    Xtraining_combined   = [Xtraining_combined;   data.Xtraining_Cells{1}];
    Ytraining_combined   = [Ytraining_combined;   data.Ytraining_regression_cells{1}];
    Xvalidation_combined = [Xvalidation_combined; data.Xvalidation_Cells{1}];
    Yvalidation_combined = [Yvalidation_combined; data.Yvalidation_regression{1}];
end
disp("Combined training and validation cells from all files")

% Prepare containers for the trained networks and training info
DNN_Trained_All = cell(1, 8);  % one network per OFDM block
info_All        = cell(1, 8);  % corresponding training info

% Train one DNN per block (currently configured for block 1 only)
for blk = 1:1
    % Train the DNN using the combined datasets
    % data.Training_set_ratio: fraction originally used for training
    % data.Xtrain_max_values(1), data.Ytrain_max_values(1): scaling maxima
    [DNN_Trained, info] = Train_DNN( ...
        Xtraining_combined, ...
        Ytraining_combined, ...
        Xvalidation_combined, ...
        Yvalidation_combined, ...
        data.Training_set_ratio, ...
        data.Xtrain_max_values(1), ...
        data.Ytrain_max_values(1) ...
    );
    
    % Store the trained network and its training record
    DNN_Trained_All{blk} = DNN_Trained;
    info_All{blk}        = info;
    
    disp(['Block ', num2str(blk), ' training complete'])
end

% Save all trained DNN models and training information to a single .mat file
save('DNN_Trained_ALL_Blocks.mat', 'DNN_Trained_All', 'info_All');
disp("Saved all trained networks and info to DNN_Trained_ALL_Blocks.mat")