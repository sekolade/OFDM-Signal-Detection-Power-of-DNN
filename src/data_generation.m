%% Settings
clear all                             % Clear all variables
rng('shuffle')                       % Seed random number generator based on current time
delete(gcp('nocreate'))              % Shut down any existing parallel pools

Training_set_ratio = 0.8;            % Ratio of data to use for training
Num_of_training_and_validation_frame = 100000;  % Total number of frames for training + validation
SNR_values = 5:5:25;                 % List of SNR values (5, 10, 15, 20, 25)

%% Parameters
Num_of_frame = Num_of_training_and_validation_frame;  
M = 4;                               % QPSK modulation order
k = log2(M);                         % Bits per QPSK symbol
check_chan = 0;                      % Unused flag for channel checking
Num_of_subcarriers = 63;             % Number of OFDM subcarriers
Num_of_FFT = Num_of_subcarriers + 1; % FFT size (including DC)
length_of_CP = 16;                   % Length of cyclic prefix

Num_of_symbols = 1;                  % Number of OFDM symbols per frame (excluding pilot)
Num_of_pilot = 1;                    % Number of pilot symbols per frame
Frame_size = Num_of_symbols + Num_of_pilot;  % Total symbols per OFDM frame

Pilot_interval = Frame_size / Num_of_pilot;  % Interval between pilots
Pilot_starting_location = 1;                 % Starting index for pilot placement

length_of_symbol = Num_of_FFT + length_of_CP;  % Total time-domain symbol length

numTrain = round(Training_set_ratio * Num_of_frame);  % Number of training frames
numVal = Num_of_frame - numTrain;                    % Number of validation frames

%% SNR lists
% Create randomized SNR assignments for each frame
SNR_list = repmat(SNR_values, 1, ceil(Num_of_frame / numel(SNR_values)));
SNR_list = SNR_list(1:Num_of_frame);
SNR_list = SNR_list(randperm(Num_of_frame));

idxs = randperm(Num_of_frame);
trainIdx_total = idxs(1:numTrain);     % Training indices
valIdx_total = idxs(numTrain+1:end);   % Validation indices

%% Preallocate storage cells
Xtraining_Cells = cell(1,8);
Xvalidation_Cells = cell(1,8);
Ytraining_regression_cells = cell(1,8);
Yvalidation_regression = cell(1,8);
Ytraining_categorical_doubles = [];
Yvalidation_categorical_doubles = [];

%% Batch processing setup
batch_size = 10000;
num_batches = ceil(Num_of_frame / batch_size);

% Initialize antenna arrays using WINNER II model
AA(1) = winner2.AntennaArray('UCA', 16, 0.3);
AA(2) = winner2.AntennaArray('UCA', 1,  0.05);

% Load pre-generated channel responses
tmp = load('savedchans.mat');
channel_responses_all = tmp.h_eq_cell;

for batch = 1:num_batches
    fprintf("Batch %d / %d\n", batch, num_batches);

    batch_start = (batch - 1) * batch_size + 1;
    batch_end = min(batch * batch_size, Num_of_frame);
    batch_range = batch_start:batch_end;
    batch_len = length(batch_range);

    % Find which frames in this batch are for training vs. validation
    [~, batch_train_mask] = ismember(batch_range, trainIdx_total);
    [~, batch_val_mask]   = ismember(batch_range, valIdx_total);
    train_mask = batch_train_mask > 0;
    val_mask   = batch_val_mask   > 0;

    % Get corresponding SNR values for this batch
    SNR_batch = SNR_list(batch_range);

    % Temporary storage for this batch
    Xtrain_temp = cell(8, batch_len);
    Xval_temp   = cell(8, batch_len);
    Ytrain_reg_temp = cell(8, batch_len);
    Yval_reg_temp   = cell(8, batch_len);
    Ytrain_cat_temp = zeros(1, batch_len);
    Yval_cat_temp   = zeros(1, batch_len);

    %% Parallel feature extraction loop
    if isempty(gcp('nocreate'))
        parpool('local');  % Start parallel pool if not already running
    end
    parfor idx = 1:batch_len

        Frame = batch_range(idx);
        SNR   = SNR_batch(idx);

        % Generate random QPSK data symbols
        N = Num_of_subcarriers * Num_of_symbols;
        data = randi([0 1], N, k);
        dataSym = bi2de(data);
        QPSK_symbol = QPSK_Modulator(dataSym);
        QPSK_signal = reshape(QPSK_symbol, Num_of_subcarriers, Num_of_symbols);

        % Pilot insertion
        Pilot_value = (1 - 1j)/sqrt(2);
        Pilot_location = Pilot_starting_location : Pilot_interval : Frame_size;
        data_location = 1 : Frame_size;
        data_location(Pilot_location(:)) = [];

        data_in_IFFT = zeros(Num_of_FFT - 1, Frame_size);
        data_in_IFFT(:, Pilot_location) = Pilot_value;
        data_in_IFFT(:, data_location)  = QPSK_signal;
        data_in_IFFT = [zeros(1, Frame_size); data_in_IFFT];

        % OFDM transmitter: IFFT + CP
        Transmitted_signal = OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);

        % Pass through WINNER II channel
        channel_h = channel_responses_all{ mod(Frame - 1, length(channel_responses_all)) + 1 };
        Multitap_Channel_Signal = conv(Transmitted_signal, channel_h);
        Multitap_Channel_Signal = Multitap_Channel_Signal(1 : length(Transmitted_signal));
        es = mean(abs(Multitap_Channel_Signal).^2);
        % Additive white Gaussian noise
        [Multitap_Channel_Signal,noi] = awgn(Multitap_Channel_Signal, SNR, 'measured');
        disp(10*log10(es/noi));
        % OFDM receiver with known channel (idealized)
        Channel_signal_when_h_is_known = [1; zeros(size(Multitap_Channel_Signal,1)-1, 1)];
        [Received_data, ~] = OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Channel_signal_when_h_is_known);

        % Divide QPSK symbols into 8 blocks
        blk = cell(1, 8);
        for b = 1:7
            blk{b} = QPSK_signal((b-1)*8+1 : b*8);
        end
        blk{8} = [QPSK_signal(57:63); QPSK_signal(57)];

        % Extract features and labels for each block (only block 1 used here)
        for b = 1:1
            symIdx = (b-1)*8+1 : b*8;
            [feat_tmp, label_tmp, label_reg_tmp] = Extract_Feature_OFDM(Received_data, dataSym(1:2), M, blk{b});

            if train_mask(idx)
                Xtrain_temp{b, idx}     = feat_tmp;
                Ytrain_reg_temp{b, idx} = label_reg_tmp;
                if b == 1
                    Ytrain_cat_temp(idx) = label_tmp;
                end
            elseif val_mask(idx)
                Xval_temp{b, idx}     = feat_tmp;
                Yval_reg_temp{b, idx} = label_reg_tmp;
                if b == 1
                    Yval_cat_temp(idx) = label_tmp;
                end
            end
        end
    end

    %% Aggregate batch data into full dataset
    for b = 1:1
        Xtraining_Cells{b}          = [Xtraining_Cells{b}; Xtrain_temp(b, train_mask)'];
        Xvalidation_Cells{b}        = [Xvalidation_Cells{b}; Xval_temp(b, val_mask)'];
        Ytraining_regression_cells{b} = [Ytraining_regression_cells{b}; Ytrain_reg_temp(b, train_mask)'];
        Yvalidation_regression{b}     = [Yvalidation_regression{b}; Yval_reg_temp(b, val_mask)'];
    end
    Ytraining_categorical_doubles = [Ytraining_categorical_doubles; Ytrain_cat_temp(train_mask)'];
    Yvalidation_categorical_doubles = [Yvalidation_categorical_doubles; Yval_cat_temp(val_mask)'];

    %% (Optional) Temporary save for checkpointing
    % save(sprintf('OFDM_Dataset_TEMP_%07d.mat', batch_end), ...
    %     'Xtrain_temp', 'Xval_temp', 'Ytrain_reg_temp', 'Yval_reg_temp', ...
    %     'Ytrain_cat_temp', 'Yval_cat_temp', '-v7.3');
end

%% 3. Save final values
% Convert categorical labels and save dataset
Ytraining_categorical = categorical(Ytraining_categorical_doubles);
Yvalidation_categorical = categorical(Yvalidation_categorical_doubles);

%% === GLOBAL MAX NORMALIZATION and saving for use during testing ===
Xtrain_max_values = zeros(1, 8);
Ytrain_max_values = zeros(1, 8);

for b = 1:1
    % --- Compute global maximum for features ---
    all_X_train = cell2mat(Xtraining_Cells{b});
    max_X = max(abs(all_X_train(:))) + eps;
    Xtrain_max_values(b) = max_X;

    % --- Compute global maximum for regression targets ---
    all_Y_train = cell2mat(Ytraining_regression_cells{b});
    max_Y = max(abs(all_Y_train(:))) + eps;
    Ytrain_max_values(b) = max_Y;
end

save('OFDM_Dataset_SNR_Mixed.mat', ...
    'Xtraining_Cells', 'Xvalidation_Cells', ...
    'Ytraining_regression_cells', 'Yvalidation_regression', ...
    'Ytraining_categorical', 'Yvalidation_categorical', ...
    'Xtrain_max_values', 'Ytrain_max_values', ...
    'Training_set_ratio', '-v7.3');

%% Common style settings for later plotting
lw    = 1.5;    % Line width
ms    = 8;      % Marker size
fs    = 14;     % Font size for axes
figW  = 900;    % Figure width in pixels
figH  = 400;    % Figure height in pixels

%% 1) DFT symbols X[k]
%% 1) DFT symbols X[k]
% X = reshape(data_in_IFFT, [], 1);  % 128×1 complex
% 
% figure('Position',[100 100 figW figH]);
% sgtitle(sprintf('DFT symbols X[k] — SNR = %d dB', SNR),'FontSize',fs+2);
% 
% subplot(2,1,1);
% stem(0:127, real(X), 'b', 'LineWidth',lw, 'MarkerSize',ms);
% ylabel('Re\{X[k]\}','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Real Part','FontSize',fs+1);
% 
% subplot(2,1,2);
% stem(0:127, imag(X), 'r', 'LineWidth',lw, 'MarkerSize',ms);
% xlabel('k','FontSize',fs); ylabel('Im\{X[k]\}','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Imaginary Part','FontSize',fs+1);
% 
% %saveas(gcf, sprintf('Xk_subplots_SNR_%ddB.png', SNR));
% 
% %% 2) Transmitted OFDM Time Real / Imag
% n_tx = 0:length(Transmitted_signal)-1;
% 
% figure('Position',[100 100 figW figH]);
% sgtitle(sprintf('Transmitted OFDM Time Domain — SNR = %d dB', SNR),'FontSize',fs+2);
% 
% subplot(2,1,1);
% stem(n_tx, real(Transmitted_signal), 'b', 'LineWidth',lw, 'MarkerSize',ms);
% ylabel('Re','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Real Part','FontSize',fs+1);
% 
% subplot(2,1,2);
% stem(n_tx, imag(Transmitted_signal), 'r', 'LineWidth',lw, 'MarkerSize',ms);
% xlabel('n','FontSize',fs); ylabel('Im','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Imaginary Part','FontSize',fs+1);
% 
% %saveas(gcf, sprintf('TX_subplots_SNR_%ddB.png', SNR));
% 
% %% 3) Channel Output Pre-AWGN Real / Imag
% n_ch = 0:length(preAWGN_signal)-1;
% 
% figure('Position',[100 100 figW figH]);
% sgtitle(sprintf('Channel Output Pre-AWGN — SNR = %d dB', SNR),'FontSize',fs+2);
% 
% subplot(2,1,1);
% stem(n_ch, real(preAWGN_signal), 'b', 'LineWidth',lw, 'MarkerSize',ms);
% ylabel('Re','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Real Part','FontSize',fs+1);
% 
% subplot(2,1,2);
% stem(n_ch, imag(preAWGN_signal), 'r', 'LineWidth',lw, 'MarkerSize',ms);
% xlabel('n','FontSize',fs); ylabel('Im','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Imaginary Part','FontSize',fs+1);
% 
% %saveas(gcf, sprintf('PreAWGN_subplots_SNR_%ddB.png', SNR));
% 
% %% 4) Channel Output After-AWGN Real / Imag
% noisy_signal = Multitap_Channel_Signal;
% n_noisy = 0:length(noisy_signal)-1;
% 
% figure('Position',[100 100 figW figH]);
% sgtitle(sprintf('Channel Output After-AWGN — SNR = %d dB', SNR),'FontSize',fs+2);
% 
% subplot(2,1,1);
% stem(n_noisy, real(noisy_signal), 'b', 'LineWidth',lw, 'MarkerSize',ms);
% ylabel('Re','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Real Part','FontSize',fs+1);
% 
% subplot(2,1,2);
% stem(n_noisy, imag(noisy_signal), 'r', 'LineWidth',lw, 'MarkerSize',ms);
% xlabel('n','FontSize',fs); ylabel('Im','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Imaginary Part','FontSize',fs+1);
% 
% %saveas(gcf, sprintf('PostAWGN_subplots_SNR_%ddB.png', SNR));
% 
% %% 5) Received DFT Real / Imag
% Y = reshape(Received_data, [], 1);
% 
% figure('Position',[100 100 figW figH]);
% sgtitle(sprintf('Received DFT Symbols — SNR = %d dB', SNR),'FontSize',fs+2);
% 
% subplot(2,1,1);
% stem(0:127, real(Y), 'b', 'LineWidth',lw, 'MarkerSize',ms);
% ylabel('Re\{Y[k]\}','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Real Part','FontSize',fs+1);
% 
% subplot(2,1,2);
% stem(0:127, imag(Y), 'r', 'LineWidth',lw, 'MarkerSize',ms);
% xlabel('k','FontSize',fs); ylabel('Im\{Y[k]\}','FontSize',fs);
% grid on; set(gca,'FontSize',fs);
% title('Imaginary Part','FontSize',fs+1);
% 
% %saveas(gcf, sprintf('Received_subplots_SNR_%ddB.png', SNR));
% 
% %% Combined Constellation: Transmitted vs Received
% % ——— Normalized Constellations in Subplots ———
% % Ön koşullar: X, Y, ms, fs, figW, figH tanımlı
% 
% % 1) Maskeler (orijine çok yakın noktaları çıkar)
% %epsVal = 1e-99;
% 
% % 2) Normalize et (maks mutlak değere böl)
% %X = X / max(abs(X));
% %Y =  Y / max(abs(Y));
% Xn = [X(2:64), X(66:end)];
% Yn = [Y(2:64), Y(66:end)];
% c1 = max(max(max(abs(Xn))),max(max(abs(Yn))));
% %Xn = ~(abs(real(Xn)) < epsVal & abs(imag(Xn)) < epsVal);
% %Yn = ~(abs(real(Yn)) < epsVal & abs(imag(Yn)) < epsVal);
% 
% % 3) Subplot’ta yan yana çiz
% figure('Position',[100 100 figW figH]);
% subplot(1,2,1);
% scatter(real(Xn), imag(Xn), ms^2, 'o', 'filled');
% title(', Transmitted Constellation','FontSize',fs);
% xlabel('Re\{X\}','FontSize',fs); ylabel('Im\{X\}','FontSize',fs);
% axis equal; grid on; set(gca,'FontSize',fs);
% %xlim([-c1 c1]); ylim([-c1 c1]);
% 
% subplot(1,2,2);
% scatter(real(Yn), imag(Yn), ms^2, 'x', 'LineWidth',1.5, 'MarkerEdgeColor','b');
% title(', Received Constellation','FontSize',fs);
% xlabel('Re\{Y\}','FontSize',fs); ylabel('Im\{Y\}','FontSize',fs);
% axis equal; grid on; set(gca,'FontSize',fs);
% %xlim([-c1 c1]); ylim([-c1 c1]);
% 
% % 4) Kaydet (isteğe bağlı)
% %saveas(gcf, sprintf('Normalized_Constellations_SNR_%ddB.png', SNR));
% 

