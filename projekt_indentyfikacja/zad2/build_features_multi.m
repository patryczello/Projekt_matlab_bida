function X = build_features_multi(mat, fs, windowLen, hop)
% BUILD_FEATURES_MULTI - extract features from multi-channel signal (6 channels)
% mat: N x C (C=6: ia, ib, ic, ialfa, ibeta, torque)
% fs: sampling rate
% windowLen: window length in samples
% hop: window shift step
% return: X - NxM feature matrix

n = size(mat,1);
C = size(mat,2);
X = [];

for s = 1:hop:(n - windowLen + 1)
    w = mat(s:s+windowLen-1, :); % windowLen x C
    f_all_channels = [];

    for p = 1:C
        x = w(:,p);

        f_stats = [rms(x), std(x), skewness(x), kurtosis(x), max(x) - min(x), max(abs(x)) / (curr_rms + 1e-6)];

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

    ia = w(:,1); ib = w(:,2); ic = w(:,3);
    zero_seq = mean(ia + ib + ic);
    pos_seq = max(abs(ia + ib*exp(-1j*2*pi/3) + ic*exp(1j*2*pi/3)));

    X = [X; f_all_channels, zero_seq, pos_seq];
end

end
