function [H_MMSE_h,rf,rf2] = MMSE_Channel_Tap_Block_Pilot_Demo_1(Pilot_indices,Received_Pilot, Pilot_Value, Nfft, Frame_size, SNR, h)
%MMSE_Channel_Tap_Block_Pilot_DEMO_1  Block-wise MMSE channel estimation
%   Estimates the frequency-domain channel response for a block of OFDM
%   symbols using pilot subcarriers and the true impulse response statistics.
%
%   Inputs:
%     Pilot_indices   - Indices of pilot subcarriers (vector)
%     Received_Pilot  - Received samples at pilot subcarriers (Np×1 vector)
%     Pilot_Value     - Known transmitted pilot symbol values
%     Nfft            - Number of data subcarriers (excl. DC)
%     Frame_size      - Number of OFDM symbols in the block
%     SNR             - Signal-to-noise ratio in dB
%     h               - True channel impulse response (length-L vector)
%
%   Outputs:
%     H_MMSE_h        - MMSE-estimated channel [Nfft×Frame_size]
%     rf              - Cross-covariance matrix [Nfft×Np]
%     rf2             - Auto-covariance matrix [Np×Np]

H_MMSE_h = zeros(Nfft, Frame_size);       % Preallocate output matrix

SNR_HEX = 10^(SNR / 10);                  % Convert SNR from dB to linear scale

Np = Nfft;                                % Number of pilot subcarriers (using all here)

%H_LS = Received_Pilot ./ Pilot_Value;    % Optional direct LS estimate (commented)

%% LS estimate at pilot positions and interpolation
H_LS_pilots = Received_Pilot;                     % Np×1 vector of pilot measurements
H_LS_values = H_LS_pilots(Pilot_indices) ./ Pilot_Value(Pilot_indices);
all_indices = 1:63;                               % All subcarrier indices (1..63)
H_LS = interp1(Pilot_indices, H_LS_values, all_indices, 'linear', 'extrap').';

Nps = 1;                                           % Number of pilot OFDM symbols per block

%% Compute RMS delay spread from true impulse response
k   = 0:length(h)-1;                               % Tap index vector
hh  = h * h';                                      % Total channel power (scalar)
tmp = h .* conj(h) .* k;                           % Weighted tap energies over delay
r   = sum(tmp) / hh;                               % Mean delay
r2  = tmp * k.' / hh;                              % Second moment of delay
tau_rms = sqrt(r2 - r.^2);                         % RMS delay spread

%% Build MMSE correlation matrices in frequency domain
df = 1 / Nfft;                                     % Normalized subcarrier spacing
j2pi_tau_df = 1j * 2 * pi * tau_rms * df;

% Cross-covariance between all subcarriers and pilot subcarriers
K1 = repmat((0:Nfft-1).', 1, Np);
K2 = repmat(0:Np-1, Nfft, 1);
rf  = 1 ./ (1 + j2pi_tau_df * (K1 - K2 * Nps));     % [Nfft×Np]

% Auto-covariance among pilot subcarriers
K3  = repmat((0:Np-1).', 1, Np);
K4  = repmat(0:Np-1, Np, 1);
rf2 = 1 ./ (1 + j2pi_tau_df * Nps * (K3 - K4));      % [Np×Np]

%% Compute MMSE channel estimate and replicate over frame
Rhp    = rf;                                       % Cross-covariance
Rpp    = rf2 + (eye(length(H_LS)) / SNR_HEX);      % Auto-covariance + noise
H_MMSE = Rhp * pinv(Rpp) * H_LS;                   % MMSE estimate [Nfft×1]

for i_MMSE = 1 : Frame_size
    H_MMSE_h(:, i_MMSE) = H_MMSE;                  % Copy estimate to all symbols
end