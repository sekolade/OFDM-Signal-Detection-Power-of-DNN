%%

%%
rng('shuffle')
Rsym         = 1e5;            % Symbol rate [symbols/s] – adjust as needed
rollOff      = 0.25;           % Root-raised-cosine (RRC) filter roll-off factor
spanSym      = 10;             % RRC filter length in symbols (filter span)
fc           = 2.6e9;          % Carrier frequency for WINNER-II channel [Hz]
max_delay_period = 16;         % Maximum delay of 16 symbol periods as requiered in the paper

SNR_values = 5:5:25;
SNR_Range = SNR_values; 
%
pilot_num = 32; %Impact of Pilot Numbers
%
clip = 0;%Impact of Clipping , %1 enable clipping %0 disable
clip_ratio = 4.5; % Impact of Clipping
%
cyclic_prefix = 0; % Impact of CP, %1 enable CP, %0 disable
nonrobust=0; %Impact of mismatch between training and deployment stage , %1 enable mismatch %0 disable


% Preallocate arrays to store BER and SER results for each detector over SNR
MMSE_BER_over_SNR = zeros(length(SNR_Range), 1);
MMSE_SER_over_SNR = zeros(length(SNR_Range), 1);
ZF_BER_over_SNR   = zeros(length(SNR_Range), 1);
ZF_SER_over_SNR   = zeros(length(SNR_Range), 1);
DNN_BER_over_SNR  = zeros(length(SNR_Range), 1);
DNN_SER_over_SNR  = zeros(length(SNR_Range), 1);

%% Load pre-trained deep neural networks for all OFDM blocks
load('savednets/DNN_Trained_ALL_Blocks4.mat');  % Contains DNN_Trained_All{1..8}
load('savedchans.mat','h_eq_cell');

%% Configure WINNER-II channel layout and parameters
AA(1) = winner2.AntennaArray('UCA', 16, 0.3);   % Base station: uniform circular array, 16 elements, 0.3λ spacing
AA(2) = winner2.AntennaArray('UCA', 1,  0.05);  % Mobile station: single antenna, 0.05λ spacing
BSIdx    = {1};
MSIdx    = [2];
numLinks = 1;
range    = 300;                                % Link distance [m]
cfgLayout = winner2.layoutparset(MSIdx, BSIdx, numLinks, AA, range);

cfgLayout.Pairing                = [1; 2];     % Pair MS 2 with BS 1
cfgLayout.PropagConditionVector  = [0];        % Use default propagation condition

% Manually set positions of BS and MS in 2D plane
cfgLayout.Stations(1).Pos(1:2) = [150; 150];   % BS at (150 m, 150 m)
cfgLayout.Stations(2).Pos(1:2) = [10; 180];    % MS at (10 m, 180 m)

% Channel model settings
cfgWim = winner2.wimparset;

% Choose scenario vector based on robustness flag
if nonrobust == 0
    cfgWim.IntraClusterDsUsed  = 'yes';       % Include intra-cluster delay spread
    cfgLayout.ScenarioVector   = [11];        % Urban microcell
else
    cfgWim.IntraClusterDsUsed  = 'no';        % No intra-cluster delay spread
    cfgLayout.ScenarioVector   = [12];        % More challenging scenario
end

% Other channel settings
cfgWim.CenterFrequency     = fc;              % Channel center frequency
cfgWim.UniformTimeSampling = 'no';            % Use random sampling times
cfgWim.ShadowingModelUsed  = 'yes';           % Enable log-normal shadowing
cfgWim.PathLossModelUsed   = 'yes';           % Enable path loss

%% Loop over each SNR point
for SNR = SNR_Range
    disp(['Simulating for SNR = ', num2str(SNR), ' dB'])
    
    % OFDM and modulation parameters
    Baseband_bandwidth  = 20e6;                % Baseband bandwidth [Hz]
    M                   = 4;                   % QPSK modulation order
    k                   = log2(M);             % Bits per QPSK symbol
    
    Num_of_subcarriers  = 63;                  % Data subcarriers (excluding DC)
    Num_of_FFT          = Num_of_subcarriers + 1;  % FFT size including DC bin
    %% Channel mean and covariance estimation
    h_all  = cell2mat(h_eq_cell.');              % (frameLenSym × Nframe)
    H_all  = fft(h_all, Num_of_FFT, 1);          % (Nsc × Nframe)  (Nsc=64)
    H_all_data = H_all(2:end,:);        % 63×N
    mu_H   = mean(H_all_data, 2);                     % (Nsc × 1)
    R_H    = cov(H_all_data.');                       % (Nsc × Nsc)
    %%
    % Set cyclic prefix length based on flag
    if cyclic_prefix == 1
        length_of_CP = 16;                     % CP length in samples
    else
        length_of_CP = 0;                      % No CP
    end
    
    % Frame structure
    Num_of_frame    = 200;                     % Number of OFDM frames to average over
    Num_of_symbols  = 1;                       % Data symbols per OFDM symbol
    Num_of_pilot    = 1;                       % Pilot symbols per OFDM symbol
    Frame_size      = Num_of_symbols + Num_of_pilot;
    
    % Pilot arrangement
    Pilot_interval           = Frame_size / Num_of_pilot;
    Pilot_starting_location  = 1;              % First symbol index for pilot
    
    % Compute lengths
    length_of_symbol = Num_of_FFT + length_of_CP;            % Total OFDM symbol length
    Num_of_QPSK_symbols      = Num_of_subcarriers * Num_of_symbols * Num_of_frame;
    Num_of_bits              = Num_of_QPSK_symbols * k;
    Num_of_QPSK_symbols_DNN  = 64 * Num_of_symbols * Num_of_frame; % For DNN (8 blocks × 8 symbols)
    Num_of_bits_DNN          = Num_of_QPSK_symbols_DNN * k;
    frameLen          = Frame_size*length_of_symbol ;


    % Preallocate error tracking arrays for this SNR
    MMSE_numErrs_sym      = zeros(Num_of_frame, 1);
    MMSE_SER_in_frame     = zeros(Num_of_frame, 1);
    MMSE_numErrs_bit      = zeros(Num_of_frame, 1);
    MMSE_BER_in_frame     = zeros(Num_of_frame, 1);
    
    ZF_numErrs_sym        = zeros(Num_of_frame, 1);
    ZF_SER_in_frame       = zeros(Num_of_frame, 1);
    ZF_numErrs_bit        = zeros(Num_of_frame, 1);
    ZF_BER_in_frame       = zeros(Num_of_frame, 1);
    
    DNN_total_sym_errs    = 0;
    DNN_total_bit_errs    = 0;
    
    %% Loop over each OFDM frame
    for Frame = 1:Num_of_frame
        % Generate random bit sequence for this frame
        N    = Num_of_subcarriers * Num_of_symbols;
        data = randi([0 1], N, k);            % Bits matrix of size N×k
        Data = reshape(data, [], 1);           % Vectorized bit stream
        
        % Map bits to QPSK symbols
        dataSym     = bi2de(data);             % Convert binary to decimal symbol indices
        QPSK_symbol = QPSK_Modulator(dataSym); % Modulate to QPSK complex symbols
        QPSK_signal = reshape(QPSK_symbol, Num_of_subcarriers, Num_of_symbols);
        
        % --- Pilot tone generation ---
        Pilot_value      = zeros(Num_of_FFT - 1, 1);  
        pilot_num        = min(pilot_num, 64);                     
        Pilot_indices    = 1:floor(64/pilot_num):63;               
        Pilot_value(Pilot_indices) = (1 - 1j)/sqrt(2);  % Example 8-PSK pilot symbols
        
        % --- Allocate pilots and data within OFDM frame ---
        Pilot_location = Pilot_starting_location : Pilot_interval : Frame_size;
        data_location  = setdiff(1:Frame_size, Pilot_location);
        
        % Build input matrix for IFFT (exclude DC at index 1)
        data_in_IFFT = zeros(Num_of_FFT - 1, Frame_size);
        data_in_IFFT(:, Pilot_location) = Pilot_value;
        data_in_IFFT(:, data_location)  = QPSK_signal;
        data_in_IFFT = [zeros(1,Frame_size); data_in_IFFT];   % Insert DC subcarrier
        
        % Transmit OFDM signal (IFFT + CP)
        Transmitted_signal = OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);
        

        % --- Configure and generate WINNER-II channel impulse response ---
%         cfgWim.RandomSeed  = randi([0 2^31 - 1]);   % New random seed per frame
%         for i = sum(cfgLayout.NofSect)+(1:length(MSIdx))
%             cfgLayout.Stations(i).Velocity = rand(3,1) - 0.5;  % Random MS velocity vector
%         end

        %% Based on WINNER II Delay distribution, CDF calculation to guarantee inequality below
        % maximum delay < 16 sampling period (specified in original paper) by arranging mobile velocity
        numBSSect = sum(cfgLayout.NofSect);
        p = 99.99 / 100; 
        z = norminv(p, 0, 1);
        logDS_th = -6.63 + 0.32 * z;
        DS_th = 10.^logDS_th;
        max_fs = max_delay_period / DS_th;
        vel_max = max_fs*((2.99792458e8/cfgWim.CenterFrequency/2/2000000));
        for k = numBSSect + 1 : numBSSect + numel(MSIdx)
            cfgLayout.Stations(k).Velocity = (rand(3,1) - 0.5)/0.5 * vel_max;
        end
        
        %%

        WINNERChan = comm.WINNER2Channel(cfgWim, cfgLayout);
        chanInfo   = info(WINNERChan);
        Fs         = chanInfo.SampleRate;          % Channel sampling rate
        L          = round(Fs / Rsym);             % Upsampling factor
        [H1,~,finalCond] = winner2.wim(cfgWim,cfgLayout);

        % --- Pulse-shaping filters for channel sounding ---
        txRRC_tmpl = comm.RaisedCosineTransmitFilter( ...
                         'RolloffFactor',     rollOff, ...
                         'FilterSpanInSymbols', spanSym, ...
                         'OutputSamplesPerSymbol', L);
        rxRRC_tmpl = comm.RaisedCosineReceiveFilter( ...
                         'RolloffFactor',     rollOff, ...
                         'FilterSpanInSymbols', spanSym, ...
                         'InputSamplesPerSymbol',  L, ...
                         'DecimationFactor', 1);  % Manual decimation
        % Send an impulse through transmit filter, channel, and receive filter
        impSym = [1];
        impUp  = upsample(impSym, L);
        txSig  = txRRC_tmpl(impUp);

        % Reconfigure the channel for NumTimeSamples
        cfgWim.NumTimeSamples      = length(txSig); % Number of time samples for channel synthesis
        WINNERChan = comm.WINNER2Channel(cfgWim, cfgLayout);
        chanInfo   = info(WINNERChan);
        Fs         = chanInfo.SampleRate;          % Channel sampling rate
        L          = round(Fs / Rsym);             % Upsampling factor


        %Send an impulse through transmit filter, channel, and receive filter cont.
        txSig  = cellfun(@(x) ones(1,x).*txSig, num2cell(chanInfo.NumBSElements)','UniformOutput',false);
        chCell = WINNERChan(txSig);
        rxSig  = rxRRC_tmpl(chCell{1});
        
        % --- Group delay compensation and channel tap extraction ---
        grpDelay = spanSym * L/2;
        rxSync   = rxSig(grpDelay+1:end);       % Remove preamble zeros
        h_eq     = rxSync(1:L:end);             % Downsample to symbol rate
        
        % With CDF calculation for mobile velocity for channel configuration above, 
        % h_eq has max delay of 16 symbol period. Below just truncates/adds
        % trailing zeros if any
        if length(h_eq) < max_delay_period
            h_eq = [h_eq;zeros(max_delay_period-length(h_eq),1)];
        else
            h_eq = h_eq(1:max_delay_period);
        end
        
        % Convolve transmitted OFDM signal with estimated channel taps
        Multitap_Channel_Signal = conv(Transmitted_signal, h_eq);
        Multitap_Channel_Signal = Multitap_Channel_Signal(1:length(Transmitted_signal));

        % Add AWGN noise with specified SNR
        [Multitap_Channel_Signal,noise_power] = awgn(Multitap_Channel_Signal, SNR, 'measured');
        % --- Optional clipping to control PAPR ---
        if clip == 1
            CR      = clip_ratio;            
            sigma   = rms(Multitap_Channel_Signal);
            A       = CR * sigma;
            phases  = angle(Multitap_Channel_Signal);
            magnitudes = abs(Multitap_Channel_Signal);
            clipped_signal = Multitap_Channel_Signal;
            clipped_signal(magnitudes > A) = A * exp(1j * phases(magnitudes > A));
            Multitap_Channel_Signal = clipped_signal;
        end
        
        % --- OFDM reception (demodulation before equalization) ---
        Channel_signal_when_h_is_known = [1; zeros(size(Multitap_Channel_Signal,1)-1,1)];
        [Unrecovered_signal, Unrecovered_signal_when_h_is_known] = ...
            OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Channel_signal_when_h_is_known);
        
        %% --- Least Squares (LS) and MMSE channel estimation from pilots ---
        Received_pilot = Unrecovered_signal(:, Pilot_location);
        Received_pilot_data   = Received_pilot(2:end);      % Remove DC index
        Y_p = Received_pilot_data(Pilot_indices);           % Recieved pilots, pilot inserted subcarriers
        X_p = Pilot_value(Pilot_indices);                   % Transmitted pilots, pilot inserted subcarriers
        Ep = mean(abs(X_p).^2);                             % Average transmitted pilot power
        
        all_indices   = 1:63;

        H_LS_values   = Y_p ./ X_p;
        H_LS          = interp1(Pilot_indices, H_LS_values, all_indices, 'linear', 'extrap').';
        
        %H_MMSE_values = MMSE_channel_estimator(H_LS_values, noise_power, Pilot_indices, R_H, mu_H);
        %H_MMSE        = interp1(Pilot_indices, H_MMSE_values, all_indices, 'linear', 'extrap').';
        H_MMSE = MMSE_channel_estimator(H_LS_values, noise_power, Pilot_indices, R_H, mu_H);

            % --- Zero-Forcing equalization ---
            Received_Signal_ZF = Unrecovered_signal(2:end, data_location) ./ H_LS(:);
            Received_data_ZF   = Received_Signal_ZF;
        
            % --- MMSE equalization ---  
            Received_Signal_MMSE = Unrecovered_signal(2:end,data_location) ./ H_MMSE(:);
            Received_data_MMSE   = Received_Signal_MMSE;
        
        %% Compute BER and SER for MMSE
        dataSym_Rx          = QPSK_Demodulator(Received_data_MMSE);
        dataBits_Rx         = de2bi(dataSym_Rx, 2);
        Data_bits_Rx        = reshape(dataBits_Rx, [], 1);
        MMSE_numErrs_sym(Frame) = sum(dataSym ~= dataSym_Rx);
        MMSE_SER_in_frame(Frame) = MMSE_numErrs_sym(Frame) / length(dataSym);
        MMSE_numErrs_bit(Frame) = sum(Data ~= Data_bits_Rx);
        MMSE_BER_in_frame(Frame) = MMSE_numErrs_bit(Frame) / length(Data);
        
        %% Compute BER and SER for ZF
        dataSym_Rx_ZF       = QPSK_Demodulator(Received_data_ZF);
        dataBits_Rx_ZF      = de2bi(dataSym_Rx_ZF, 2);
        Data_bits_Rx_ZF     = reshape(dataBits_Rx_ZF, [], 1);
        ZF_numErrs_sym(Frame) = sum(dataSym ~= dataSym_Rx_ZF);
        ZF_SER_in_frame(Frame) = ZF_numErrs_sym(Frame) / length(dataSym);
        ZF_numErrs_bit(Frame) = sum(Data ~= Data_bits_Rx_ZF);
        ZF_BER_in_frame(Frame) = ZF_numErrs_bit(Frame) / length(Data);
        
        %% DNN-based detection for the first block
        % Split OFDM symbol into 8 blocks of 8 subcarriers each
        blk2 = cell(1,8);
        for i = 1:7
            blk2{i} = QPSK_signal((i-1)*8+1 : i*8);
        end
        blk2{8} = [QPSK_signal(57:63); QPSK_signal(57)];
        
        % Split transmitted symbol indices similarly
        blk3 = cell(1,8);
        for i = 1:7
            blk3{i} = dataSym((i-1)*8+1 : i*8);
        end
        blk3{8} = [dataSym(57:63); dataSym(57)];
        
        % Process only the first block in this example
        for blk = 1:1
            idx_start = (blk - 1) * 8 + 1;
            idx_end   = blk * 8;
            
            % Extract features for DNN input from received time-domain signal
            [DNN_feature_signal, ~, ~] = Extract_Feature_OFDM(Unrecovered_signal, dataSym(1:2), M, blk2{blk});
            
            % Predict QPSK symbols using the trained network
            Received_data_DNN = predict(DNN_Trained_All{blk}, DNN_feature_signal.').';
            
            % Reconstruct complex output and demodulate
            DNN_complex_out    = Received_data_DNN(1:2:end,:) + 1j*Received_data_DNN(2:2:end,:);
            DNN_dataSym_Rx     = QPSK_Demodulator(DNN_complex_out);
            DNN_bits_Rx        = de2bi(DNN_dataSym_Rx, 2);
            DNN_bits_vec       = reshape(DNN_bits_Rx, [], 1);
            
            % Accumulate symbol and bit errors
            DNN_total_sym_errs = DNN_total_sym_errs + sum(blk3{blk} ~= DNN_dataSym_Rx);
            DNN_total_bit_errs = DNN_total_bit_errs + sum(reshape(de2bi(blk3{blk},2),[],1) ~= DNN_bits_vec);
        end
    end
    
    %% Aggregate BER and SER over all frames for this SNR
    MMSE_BER = sum(MMSE_numErrs_bit) / Num_of_bits;
    MMSE_SER = sum(MMSE_numErrs_sym) / Num_of_QPSK_symbols;
    MMSE_BER_over_SNR(SNR/5,1) = MMSE_BER;
    MMSE_SER_over_SNR(SNR/5,1) = MMSE_SER;
    
    ZF_BER = sum(ZF_numErrs_bit) / Num_of_bits;
    ZF_SER = sum(ZF_numErrs_sym) / Num_of_QPSK_symbols;
    ZF_BER_over_SNR(SNR/5,1) = ZF_BER;
    ZF_SER_over_SNR(SNR/5,1) = ZF_SER;
    
    DNN_BER_over_SNR(SNR/5,1) = (DNN_total_bit_errs / Num_of_bits_DNN) * 8;
    DNN_SER_over_SNR(SNR/5,1) = (DNN_total_sym_errs / Num_of_QPSK_symbols_DNN) * 8;
    
end

%% Plot BER vs. SNR for ZF, MMSE, and DNN detectors
figure;
plot(SNR_range, ZF_BER_over_SNR,   '-o', 'LineWidth',1.5); hold on;
plot(SNR_range, MMSE_BER_over_SNR, '-s', 'LineWidth',1.5);
plot(SNR_range, DNN_BER_over_SNR,  '-^', 'LineWidth',1.5);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('BER vs. SNR');
legend('ZF', 'MMSE', 'DNN');
xlim([min(SNR_range) max(SNR_range)]);
ylim([1e-5 1]);
set(gca, 'YScale', 'log');

%% Plot SER vs. SNR for ZF, MMSE, and DNN detectors
figure;
plot(SNR_range, ZF_SER_over_SNR,   '-o', 'LineWidth',1.5); hold on;
plot(SNR_range, MMSE_SER_over_SNR, '-s', 'LineWidth',1.5);
plot(SNR_range, DNN_SER_over_SNR,  '-^', 'LineWidth',1.5);
grid on;
xlabel('SNR (dB)');
ylabel('SER');
title('SER vs. SNR');
legend('ZF', 'MMSE', 'DNN');
xlim([min(SNR_range) max(SNR_range)]);
ylim([1e-5 1]);
set(gca, 'YScale', 'log');