function H_MMSE = MMSE_channel_estimator(H_LS_values, noise_power, Pilot_indices, ...
                                         R_H, mu_H)

%   H_LS_values    : LS channel estimation result
%   noise_power    : in dB
%   Pilot_indices  : indices of the pilot inserted subcarriers
%   R_H            : Nsc×Nsc freq domain covariance matrix
%   mu_H           : Nsc×1 freq domain mean

%   Nsc is the total number of (used) subcarrier, and DC is removed,
%   column vectors are expected. 
%
%   ˆ ĥ_MMSE = μ_H + R_HP (R_PP + σ_n² I)⁻¹ (Y_eq − μ_P)
%   where Y_eq = Y_p ./ X_p  (one-tap LS estimates at pilot positions).

% --- 1. Noise power σ_n² -------------------------------------------------
sigma_n2  = noise_power;          % noise variance per subcarrier

% --- 2. Extract pilot-related statistics --------------------------------
Np        = numel(Pilot_indices);        % number of pilot tones
R_pp      = R_H(Pilot_indices, Pilot_indices);   % Np×Np pilot-pilot covariance
R_kp      = R_H(:, Pilot_indices);               % Nsc×Np all-to-pilot covariance
mu_H_p    = mu_H(Pilot_indices);                 % Np×1 mean at pilots

% --- 3. MMSE combining matrix -------------------------------------------
C         = R_pp + sigma_n2 * eye(Np);  
W         = R_kp / C;                   

% --- 4. MMSE estimate ----------------------------------------------------
%   ˆ ĥ_MMSE = μ_H + R_HP (R_PP + σ_n² I)⁻¹ (Y_eq − μ_P)
%   where Y_eq = Y_p ./ X_p  (one-tap LS estimates at pilot positions).
Y_eq      = H_LS_values;                 % LS channel estimates at pilots
H_MMSE    = mu_H + W * (Y_eq - mu_H_p); % Nsc×1 final estimate
end