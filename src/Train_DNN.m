function [net, info] = Train_DNN(XTrainCell, YTrainCell, XValCell, YValCell, trainRatio, Xmax, Ymax)
%TRAIN_DNN   Train a 5-layer DNN for OFDM-based feature regression

    ValidationFrequency = ceil(1 / (1 - trainRatio));

    %% 1) Stack cell arrays into big matrices
    % Each cell contains observations in columns; concatenate and transpose
    % XTrain: [N_train × inputDim], YTrain: [N_train × outputDim]
    XTrain = [XTrainCell{:}]';    
    YTrain = [YTrainCell{:}]';
    XVal   = [XValCell{:}]';
    YVal   = [YValCell{:}]';

    %% 2) Define network architecture
    % - featureInputLayer: expects 256-dimensional input
    % - Batch normalization + ReLU in each hidden block
    % - Fully connected layers decreasing in size
    % - Final regressionLayer with MSE loss
    layers = [
        featureInputLayer(256, 'Normalization', 'zscore', 'Name', 'input')
        
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        
        fullyConnectedLayer(500, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        
        fullyConnectedLayer(250, 'Name', 'fc3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        
        fullyConnectedLayer(120, 'Name', 'fc4')
        reluLayer('Name', 'relu4')
        
        fullyConnectedLayer(16,  'Name', 'fc5')
        % Optionally, you could add a sigmoidLayer here if outputs need bounding
        regressionLayer('Name', 'output')
    ];

    %% 3) Set training options
    % - Adam optimizer with a small initial learning rate
    % - Up to 100 epochs, mini-batch size of 256
    % - Shuffle data every epoch
    % - Use validation data to monitor performance
    mbs = 256;  % mini-batch size
    options = trainingOptions('adam', ...
        'InitialLearnRate',    1e-4, ...
        'MaxEpochs',           100, ...
        'MiniBatchSize',       mbs, ...
        'Shuffle',             'every-epoch', ...
        'ValidationData',      {XVal, YVal}, ...
        'ValidationFrequency', floor(size(XTrain,1) / mbs), ...
        'Verbose',             1, ...
        'Plots',               'none');

    %% 4) Train the network
    % Returns:
    %   net  - the trained network
    %   info - training progress and final metrics
    [net, info] = trainNetwork(XTrain, YTrain, layers, options);
end


