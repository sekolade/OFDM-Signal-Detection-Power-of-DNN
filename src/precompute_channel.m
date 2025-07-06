%% 0 | Clear workspace and command window
clear; clc;
rng('shuffle');                % Use a different random seed every run

%% 1 | Basic parameters
numSamples   = 50000;          % Number of distinct channel realizations to generate
frameLenSym  = 160;            % Length of channel impulse response in symbols
Rsym         = 1e6;            % Symbol rate [symbols/s]
rollOff      = 0.25;           % Root-raised-cosine (RRC) filter roll-off factor
spanSym      = 10;             % RRC filter span in symbols
fc           = 2.6e9;          % Carrier frequency for WINNER-II channel [Hz]

%% 2 | Antenna array definitions
AA(1) = winner2.AntennaArray('UCA', 16, 0.3);   % Base station: uniform circular array, 16 elements, 0.3λ spacing
AA(2) = winner2.AntennaArray('UCA',  1, 0.05);  % Mobile station: single antenna, 0.05λ spacing

%% 3 | Transmit/receive RRC filter templates
% Create dummy filters with 1 sample/symbol; will clone and adjust L inside parfor
txRRC_tmpl = comm.RaisedCosineTransmitFilter( ...
                 'RolloffFactor',      rollOff, ...
                 'FilterSpanInSymbols', spanSym, ...
                 'OutputSamplesPerSymbol', 1);
rxRRC_tmpl = comm.RaisedCosineReceiveFilter( ...
                 'RolloffFactor',      rollOff, ...
                 'FilterSpanInSymbols', spanSym, ...
                 'InputSamplesPerSymbol',  1, ...
                 'DecimationFactor',   1);  % We will decimate manually later

%% 4 | Initialize parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool('local');         % Launch default number of workers
end

%% 5 | Preallocate cell array for impulse responses
h_eq_cell = cell(numSamples, 1);

%% 6 | Channel generation loop (parallel)
parfor ii = 1:numSamples

    % 6.1 | Configure layout and channel object for WINNER-II
    BSIdx = {1}; MSIdx = [2];
    cfgLayout = winner2.layoutparset(MSIdx, BSIdx, 1, AA, 300);
    cfgLayout.Pairing               = [1; 2];    % Pair MS to BS
    cfgLayout.ScenarioVector        = 11;        % Urban macrocell, LOS
    cfgLayout.PropagConditionVector = 0;         % LOS only
    cfgLayout.Stations(1).Pos(1:2)  = [150; 150];% BS position [x; y] in meters
    cfgLayout.Stations(2).Pos(1:2)  = [ 10; 180];% MS position [x; y] in meters

    % Assign random velocity vector to mobile station
    numBSSect = sum(cfgLayout.NofSect);
    for k = numBSSect + 1 : numBSSect + numel(MSIdx)
        cfgLayout.Stations(k).Velocity = rand(3,1) - 0.5;  % Uniform in [-0.5, 0.5] m/s per axis
    end

    % Channel parameter settings
    cfgWim = winner2.wimparset;
    cfgWim.NumTimeSamples      = frameLenSym * 40;  
    cfgWim.CenterFrequency     = fc;
    cfgWim.IntraClusterDsUsed  = 'yes';     % Include intra-cluster delay spread
    cfgWim.UniformTimeSampling = 'no';      % Non-uniform time sampling
    cfgWim.ShadowingModelUsed  = 'yes';     % Enable log-normal shadowing
    cfgWim.PathLossModelUsed   = 'yes';     % Enable path loss
    cfgWim.RandomSeed          = randi([0 2^31 - 1]); % Per-realization seed

    % Create channel System object
    WINNERChan = comm.WINNER2Channel(cfgWim, cfgLayout);
    chanInfo   = info(WINNERChan);
    Fs         = chanInfo.SampleRate;             % Channel sampling rate [Hz]
    L          = round(Fs / Rsym);                % Oversampling factor (samples per symbol)

    %% 6.2 | Clone and configure RRC filters for this worker
    txRRC = clone(txRRC_tmpl);
    rxRRC = clone(rxRRC_tmpl);
    txRRC.OutputSamplesPerSymbol = L;              % Upsample by L
    rxRRC.InputSamplesPerSymbol  = L;              % Downsample by L after filtering

    %% 6.3 | Generate unit impulse at symbol rate
    impSym = [1; zeros(frameLenSym-1,1)];          % length = frameLenSym
    impUp  = upsample(impSym, L);                  % length = frameLenSym * L

    %% 6.4 | Transmit through filter, channel, then receive filter
    txSig = txRRC(impUp);  
    % Replicate transmit signal per BS element
    txSig = cellfun(@(x) ones(1,x) .* txSig, ...
        num2cell(chanInfo.NumBSElements)', 'UniformOutput', false);
    chCell = WINNERChan(txSig);                    % Pass through channel
    rxSig  = rxRRC(chCell{1});                     % Receive filter output

    %% 6.5 | Group delay compensation and downsampling
    grpDelay = spanSym * L/2;                      % Total filter group delay in samples
    rxSync   = rxSig(grpDelay+1 : end);            % Remove preamble zeros
    h_eq     = rxSync(1 : L : end);                % Take one sample per symbol

    %% 6.6 | Truncate and store impulse response
    h_eq_cell{ii} = h_eq(1 : frameLenSym);          % Ensure fixed length

end

%% 7 | Save results and clean up
fprintf('>> %d channel impulse responses generated; saving to file...\n', numSamples);
save('savedchans.mat', 'h_eq_cell', '-v7.3');      % Use v7.3 for large variables
delete(gcp('nocreate'));                          % Shut down parallel pool

fprintf('>> Done. savedchans.mat is ready.\n');
