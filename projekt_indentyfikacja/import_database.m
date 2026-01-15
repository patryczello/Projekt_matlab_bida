% @author Dominik £uczak
% @date    2021-12-18
% @details Load database
clc;clear; close;
%% Load database into memory
database_data = load(sprintf('all_faults_faultType0.mat'));

% Select case
case_number = 0;
system_data = database_data.data_collection(:, :, case_number+1);

% Access to data
t = system_data(:,1);
n_ref = system_data(:,2);
n = system_data(:,3);
iq_ref = system_data(:,4);
iq = system_data(:,5);
id_ref = system_data(:,6);
id = system_data(:,7);
ialfa = system_data(:,8);
ibeta = system_data(:,9);
ia = system_data(:,10);
ib = system_data(:,11);
ic = system_data(:,12);
theta_electrical = system_data(:,13);
motor_torque = system_data(:,14);

% Plot data
close all;
figure(1);
plot(t, n);
hold on;
plot(t, n_ref);
hold off;
xlabel('Time (s)');
ylabel('Velocity');
legend('n', 'n_{ref}');

figure(2);
plot(t, iq);
hold on;
plot(t, iq_ref);
hold off;
xlabel('Time (s)');
legend('iq', 'iq_{ref}');

figure(3);
plot(t, id);
hold on;
plot(t, id_ref);
hold off;
xlabel('Time (s)');
legend('id', 'id_{ref}');

figure(4);
plot(t, ialfa);
hold on;
plot(t, ibeta);
hold off;
xlabel('Time (s)');
legend('ialfa', 'ibeta');

figure(5);
plot(t, ia);
hold on;
plot(t, ib);
plot(t, ic);
hold off;
xlabel('Time (s)');
legend('ia', 'ib', 'ic');

figure(6);
plot(t, theta_electrical);
hold on;
plot(t, unwrap(theta_electrical));
hold off;
xlabel('Time (s)');
legend('theta_{electrical}', 'unwrap(theta_{electrical})');

figure(7);
plot(t, motor_torque);
xlabel('Time (s)');
legend('Motor torque');

