clc; clear; close all;

%% Ustawienia użytkownika
path_data = './Faults_database/';
errors = [1,2,4,8,16,32];        % zmień tu listę przypadków błędów (np. 1:5)
fs = 20000;          % częstotliwość próbkowania (używana przy cechach)
windowLen = 100;      % długość okna (próbki)
hop = windowLen/2;     % hop (brak overlap). Możesz ustawić mniejszy
tailStart = 20001;   % indeks początku części "fault" w plikach

nCases = length(errors);

%% Wczytaj pierwszy plik żeby ustalić nTail i sprawdzić zgodność
S0 = load(sprintf('%s%s%d%s', path_data, 'fault', errors(1), '_faultType0.mat'));
if numel(S0.ia) < tailStart
    error('Plik ma mniej niż %d próbek.', tailStart);
end
nTail = numel(S0.ia) - (tailStart-1);

% Prealokacja macierzy tail (rows = nTail, cols = nCases)
ia_fault    = zeros(nTail, nCases);
ib_fault    = zeros(nTail, nCases);
ic_fault    = zeros(nTail, nCases);
ialfa_fault = zeros(nTail, nCases);
ibeta_fault = zeros(nTail, nCases);
motor_torque_fault = zeros(nTail, nCases);

% Wczytaj pliki i zapisz tail
for i = 1:nCases
    case_number = errors(i);
    S = load(sprintf('%s%s%d%s', path_data, 'fault', case_number, '_faultType0.mat'));
    % Sprawdź długości
    if numel(S.ia) < tailStart + nTail - 1
        error('Plik fault%d ma za mało próbek.', case_number);
    end
    ia_fault(:,i)    = S.ia(tailStart:end,1);
    ib_fault(:,i)    = S.ib(tailStart:end,1);
    ic_fault(:,i)    = S.ic(tailStart:end,1);
    ialfa_fault(:,i) = S.ialfa(tailStart:end,1);
    ibeta_fault(:,i) = S.ibeta(tailStart:end,1);
    motor_torque_fault(:,i) = S.motor_torque(tailStart:end,1);
end

% "OK" (poprawne) sygnały — przyjmujemy z ostatnio wczytanego pliku S (albo wczytaj osobny plik)
ia_poprawne = S.ia(1:tailStart-1,1);
ib_poprawne = S.ib(1:tailStart-1,1);
ic_poprawne = S.ic(1:tailStart-1,1);
ialfa_poprawne = S.ialfa(1:tailStart-1,1);
ibeta_poprawne = S.ibeta(1:tailStart-1,1);
motor_torque_poprawne = S.motor_torque(1:tailStart-1,1);

%% Przygotowanie macierzy wejściowej do ML
% Scal sygnały: kolumny = [ia ib ic ialfa ibeta motor_torque]
fault = [ia_fault(:), ib_fault(:), ic_fault(:), ialfa_fault(:), ibeta_fault(:), motor_torque_fault(:)];
% Uwaga: powyżej flattenujemy wszystkie kolumny jedna po drugiej — ale zamiast tego
% skorzystamy z funkcji build_features która pracuje na macierzy Nx3. Dostosujemy ją do 6-kanałowej.

% Zbuduj cechy z okien (adaptowana funkcja do 6 kanałów)
function X = build_features_multi(mat, ~, windowLen, hop)
    % mat: N x C (C = 6)
    n = size(mat,1);
    C = size(mat,2);
    X = [];
    for s = 1:hop:(n - windowLen + 1)
        w = mat(s:s+windowLen-1, :); % windowLen x C
        f_phase = [];
        for p = 1:C
            x = w(:,p);
            f_phase = [f_phase, rms(x), std(x), max(x), min(x), skewness(x), kurtosis(x)];
        end
        % złączenia międzyfazowe — dla sygnałów prądowych (pierws 3 kolumn)
        ia = w(:,1); ib = w(:,2); ic = w(:,3);
        zero_seq = mean(ia + ib + ic);
        pos_seq = max(abs(ia + ib*exp(-1j*2*pi/3) + ic*exp(1j*2*pi/3)));
        % dodatkowo: średnia i std dla motor_torque (kol.6) jako cechy globalne
        torque_mean = mean(w(:,6));
        torque_std  = std(w(:,6));
        % spektralna z pierwszego kanału (ia)
        Xf = fft(ia);
        mag = abs(Xf(1:floor(windowLen/2)));
        if numel(mag) > 2
            [~, idx] = max(mag(2:end)); idx = idx+1;
            domAmp = mag(idx);
        else
            domAmp = 0;
        end
        X = [X; f_phase, zero_seq, pos_seq, torque_mean, torque_std, domAmp];
    end
end

% Przygotuj macierze wejściowe dla fault i ok
% Najpierw zrekonstruujemy macierze N x C dla jednego przypadku: kolumny są poskładane w ia_fault(:,i) itd.
% Dla każdego case tworzymy macierz N x 6 i połączymy w długi zbiór (każdy case daje wiele okien)
X_fault = [];
for i = 1:nCases
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);
    X_fault = [X_fault; Xf];
end

% Z ok danych (przyjmujemy pojedynczy komplet poprawnych kanałów)
mat_ok = [ia_poprawne, ib_poprawne, ic_poprawne, ialfa_poprawne, ibeta_poprawne, motor_torque_poprawne];
X_ok = build_features_multi(mat_ok, fs, windowLen, hop);

%% Etykiety nowe
Y_fault = [];
for i = 1:nCases
    Xi = ia_fault(:,i); % tylko do policzenia liczby okien
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ...
           ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);

    % etykieta = numer błędu (np. 1,2,3…)
    Y_fault = [Y_fault; repmat(errors(i), size(Xf,1), 1)];
end

% klasa 0 dla OK
Y_ok = zeros(size(X_ok,1),1);

% łączymy
Yall = [Y_fault; Y_ok];


%% Połącz i przetasuj
Xall = [X_fault; X_ok];
Yall = [Y_fault; Y_ok];
perm = randperm(size(Xall,1));
Xall = Xall(perm,:);
Yall = Yall(perm);

% Podział train/test
cv = cvpartition(Yall,'HoldOut',0.3);
Xtrain = Xall(training(cv),:); Ytrain = Yall(training(cv));
Xtest  = Xall(test(cv),:);     Ytest  = Yall(test(cv));

%% Normalizacja (zachowujemy parametry do późniejszej normalizacji testów)
mu = mean(Xtrain,1);
sigma = std(Xtrain,[],1);
sigma(sigma==0) = 1;
Xtrain_norm = (Xtrain - mu) ./ sigma;
Xtest_norm  = (Xtest  - mu) ./ sigma;

%% Sieć neuronowa (prosty fully connected deep net)
inputSize = size(Xtrain_norm,2);
numClasses = numel(errors) + 1;

layers = [
    featureInputLayer(inputSize)

    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(64)
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',80, ...
    'MiniBatchSize',128, ...
    'InitialLearnRate',1e-3, ...
    'Verbose',false, ...
    'Plots','none');

% Przygotuj dane do trenowania (tabela lub ds)
Ytrain_cat = categorical(Ytrain);
dsTrain = arrayDatastore(Xtrain_norm,'IterationDimension',1);
trainTbl = combine(dsTrain, arrayDatastore(Ytrain_cat));

% Trenuj
net = trainNetwork(Xtrain_norm, Ytrain_cat, layers, options);

%% Ocena na zbiorze testowym
Ypred_probs = predict(net, Xtest_norm); % Nx2 macierz prawdopodobieństw
cats = categories(Ytrain_cat);           % {'0','1','2','4','8','16','32'}
[~, idxmax] = max(Ypred_probs,[],2);     
Ypred = str2double(cats(idxmax));        % zamiana na oryginalne etykiety


confMat = confusionmat(Ytest, Ypred);
disp('Confusion matrix (rows=true, cols=predicted):');
disp(confMat);
acc = sum(Ypred==Ytest)/numel(Ytest);
fprintf('Accuracy = %.2f%%\n', acc*100);

%% Parametry "online"
fprintf('================================== \n')
windowLen_samples = windowLen;    % długość okna w próbkach
hop_samples = 1;            % przesunięcie okna o 1 próbkę
dt = 1/fs;                  % krok czasowy

% -----------------------------
% Losowanie sygnału błędu do sprawdzenia
% -----------------------------
rng('shuffle');
all_cases = [errors, 0];
chosen_case = all_cases(randi(length(all_cases)));

if chosen_case == 0
    fprintf('Wybrano losowo sygnał OK (bez błędu)\n');
    mat_signal = mat_ok;
else
    idx = find(errors == chosen_case);
    fprintf('Wybrano losowo sygnał z błędem numer: %d\n', chosen_case);
    S = load(sprintf('%s%s%d%s', path_data, 'fault', chosen_case, '_faultType0.mat'));
    mat_signal = [S.ia, S.ib, S.ic, S.ialfa, S.ibeta, S.motor_torque];
end

nSamples = size(mat_signal,1);
fault_start_idx = tailStart;

all_cats = [0, errors];
cat_str = cellstr(string(all_cats));

confirmation_window = 5;
recent_preds = [];
found_fault = false;

% wektor przewidywanych klas dla każdej pozycji okna
pred_over_time = NaN(nSamples,1);

for s = 1:hop_samples:(nSamples - windowLen_samples + 1)
    window = mat_signal(s:s+windowLen_samples-1, :);

    % Budowa cech
    Xwin = build_features_multi(window, fs, windowLen_samples, windowLen_samples);

    % Normalizacja
    Xwin_norm = (Xwin - mu) ./ sigma;

    % Predykcja
    Yprob = predict(net, Xwin_norm);
    [~, idxmax] = max(Yprob,[],2);
    Ypred_win = str2double(cat_str(idxmax));  % klasa

    % zapisz predykcję dla czasu końca okna
    pred_over_time(s+windowLen_samples-1) = Ypred_win;

    % Mechanizm potwierdzenia błędu
    recent_preds = [recent_preds, Ypred_win];
    if length(recent_preds) > confirmation_window
        recent_preds = recent_preds(end-confirmation_window+1:end);
    end

    if length(recent_preds) == confirmation_window && numel(unique(recent_preds))==1 ...
            && recent_preds(1) ~= 0
        if ~found_fault && (chosen_case ~= 0 && s >= fault_start_idx)
            windowEnd = s + windowLen_samples - 1;
            windowEnd_time = (windowEnd - fault_start_idx) * dt;
            found_fault = true;
            fprintf('Błąd potwierdzony w oknie kończącym się w próbce %d.\n', windowEnd);
            fprintf('Czas od wystąpienia błędu: %.1f ms\n', windowEnd_time*1000);
            fprintf('Przewidziany numer błędu: %d\n', recent_preds(1));
            break
        end
    end
end

if ~found_fault
    fprintf('Błąd nie został wykryty w sygnale.\n');
end

%% Rysowanie wykresu torque z predykcjami
torque = mat_signal(:,end);
time = (0:nSamples-1) * dt;

figure;
yyaxis left
plot(time, torque, 'b', 'LineWidth', 1.2);
ylabel('Torque')
ylim([min(torque) max(torque)])  % opcjonalnie dopasuj

yyaxis right
plot(time, pred_over_time, 'r', 'LineWidth', 1.2);
ylabel('Predykcja')
ylim([min(pred_over_time) max(pred_over_time)])  % opcjonalnie dopasuj

xlabel('Czas [s]')
title('Predykcje nałożone na torque (dwie osie Y)')
grid on
legend({'Torque','Predykcja'}, Location='best')


