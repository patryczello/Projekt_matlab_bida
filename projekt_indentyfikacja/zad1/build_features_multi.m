function X = build_features_multi(mat, fs, windowLen, hop)
% BUILD_FEATURES_MULTI - ekstrakcja cech z sygnału wielokanałowego (6 kanałów)
% mat: N x C (C=6: ia, ib, ic, ialfa, ibeta, torque)
% fs: częstotliwość próbkowania
% windowLen: długość okna w próbkach
% hop: krok przesuwania okna
% Zwraca: X - macierz NxM cech

n = size(mat,1);
C = size(mat,2);
X = [];

for s = 1:hop:(n - windowLen + 1)
    w = mat(s:s+windowLen-1, :); % windowLen x C
    f_all_channels = [];

    for p = 1:C
        x = w(:,p);

        % --- Statistical features ---
        f_stats = [rms(x), std(x), max(x), min(x), skewness(x), kurtosis(x)];

        % --- Spectral feature (FFT) ---
        Xf = fft(x);
        mag = abs(Xf(1:floor(windowLen/2)));
        if numel(mag) > 2
            [~, idx] = max(mag(2:end)); % skip DC
            domAmp = mag(idx+1);
        else
            domAmp = 0;
        end

        f_all_channels = [f_all_channels, f_stats, domAmp];
    end

    % --- Inter-phase features (ia, ib, ic) ---
    ia = w(:,1); ib = w(:,2); ic = w(:,3);
    zero_seq = mean(ia + ib + ic);
    pos_seq = max(abs(ia + ib*exp(-1j*2*pi/3) + ic*exp(1j*2*pi/3)));

    X = [X; f_all_channels, zero_seq, pos_seq];
end

end
