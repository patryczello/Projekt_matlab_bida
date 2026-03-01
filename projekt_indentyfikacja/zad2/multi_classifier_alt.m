clc; clear; close all;

rng(2)
warning('off','all');

if isempty(gcp('nocreate'))
    parpool;
end

path_data = './Faults_database/';
errors = 1:63;      % errors included
fs = 20000;         % sampling rate
windowLen = 100;    % window length in samples
hop = windowLen/10; % window shift step
tailStart = 20001;  % time of fault occurance in samples
nCases = length(errors);

S0 = load(sprintf('%s%s%d%s', path_data, 'fault', errors(1), '_faultType0.mat'));
if numel(S0.ia) < tailStart
    error('File contains less than %d samples.', tailStart);
end
nTail = numel(S0.ia) - (tailStart-1);

ia_fault = zeros(nTail,nCases);
ib_fault = zeros(nTail,nCases);
ic_fault = zeros(nTail,nCases);
ialfa_fault = zeros(nTail,nCases);
ibeta_fault = zeros(nTail,nCases);
motor_torque_fault = zeros(nTail,nCases);

for i = 1:nCases
    case_number = errors(i);
    S = load(sprintf('%s%s%d%s', path_data, 'fault', case_number, '_faultType0.mat'));

    ia_fault(:,i) = S.ia(tailStart:end,1);
    ib_fault(:,i) = S.ib(tailStart:end,1);
    ic_fault(:,i) = S.ic(tailStart:end,1);
    ialfa_fault(:,i) = S.ialfa(tailStart:end,1);
    ibeta_fault(:,i) = S.ibeta(tailStart:end,1);
    motor_torque_fault(:,i) = S.motor_torque(tailStart:end,1);
end

Slast = S; % last file as example
ia_ok = Slast.ia(1:tailStart-1,1);
ib_ok = Slast.ib(1:tailStart-1,1);
ic_ok = Slast.ic(1:tailStart-1,1);
ialfa_ok = Slast.ialfa(1:tailStart-1,1);
ibeta_ok = Slast.ibeta(1:tailStart-1,1);
motor_torque_ok = Slast.motor_torque(1:tailStart-1,1);

train_plot_dir = './training_data/';
if ~exist(train_plot_dir,'dir')
    mkdir(train_plot_dir);
end

function plot_example_window(mat, Xf, windowLen, hop, fs, title_str, save_path)

    example_window_idx = 1; % pierwsze okno
    sample_window = mat((example_window_idx-1)*hop+1:(example_window_idx-1)*hop+windowLen, :);
    t_win = (0:windowLen-1)/fs;

    var_names = {'ia','ib','ic','ialfa','ibeta','torque'};
    nVars = numel(var_names);

    Xwin = Xf(example_window_idx, :);
    nFeatTotal = numel(Xwin);

    nFeatPerVar = 7;    %6 * 7 = 42
    nMainFeats  = nVars*nFeatPerVar;

    if nFeatTotal < nMainFeats + 2
        error('Too few features: %d. Expecting at least %d.', ...
              nFeatTotal, nMainFeats + 2);
    end

    zero_seq = Xwin(end-1);
    pos_seq  = Xwin(end);

    fig = figure( ...
        'Name', title_str, ...
        'NumberTitle','off', ...
        'Position',[100 100 1100 900], ...
        'Visible','off' );

    subplot(4,2,[1 2])
    plot(t_win, sample_window, 'LineWidth',1.2)
    xlabel('Czas [s]')
    ylabel('Amplituda')
    legend({'ia','ib','ic','ialfa','ibeta','motor\_torque'}, 'Location', 'best');
    title(['Example signal window - ' title_str])
    grid on

    for v = 1:nVars
        idx_start = (v-1)*nFeatPerVar + 1;
        idx_end   = v*nFeatPerVar;

        feats_v = Xwin(idx_start:idx_end);

        subplot(4,2,2+v)
        bar(feats_v)
        xlabel('Feature index')
        ylabel('Value')
        title(['Features - ' var_names{v}])
        grid on
    end

    annotation(fig, 'textbox', ...
        [0.1 0.02 0.8 0.05], ...
        'String', sprintf('zero\\_seq = %.4g    |    pos\\_seq = %.4g', zero_seq, pos_seq), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'EdgeColor','none', ...
        'FontSize',11, ...
        'FontWeight','bold');

    saveas(fig, save_path);
    close(fig)
end

for i = 1:nCases
    case_number = errors(i);
    mat = [ia_fault(:,i), ib_fault(:,i), ic_fault(:,i), ialfa_fault(:,i), ibeta_fault(:,i), motor_torque_fault(:,i)];
    Xf = build_features_multi(mat, fs, windowLen, hop);

    save_path = fullfile(train_plot_dir, sprintf('case%d_example_window.png', case_number));
    plot_example_window(mat, Xf, windowLen, hop, fs, sprintf('Case %d', case_number), save_path);
end

mat_ok = [ia_ok, ib_ok, ic_ok, ialfa_ok, ibeta_ok, motor_torque_ok];
X_ok = build_features_multi(mat_ok, fs, windowLen, hop);

save_path = fullfile(train_plot_dir, 'ok_example_window.png');
plot_example_window(mat_ok, X_ok, windowLen, hop, fs, 'OK signal', save_path);
