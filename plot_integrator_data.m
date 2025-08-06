clear;
close all;

% Load the space-separated data file
filename = 'out.txt';
data = readmatrix(filename);  % Assumes file has no header

% Extract columns by name for clarity
type   = data(:,1);
t      = data(:,2);
h      = data(:,3);
z      = data(:,4);
zdot   = data(:,5);
gamma  = data(:,6);
phi    = data(:,7);



% Filter for rows with type == 0
type0_idx = (type == 0);
t0 = t(type0_idx);
h0 = h(type0_idx);
z0 = z(type0_idx) - 0.2;
zdot0 = zdot(type0_idx);
f0 = gamma(type0_idx) ./ h0;
phi0 = phi(type0_idx);
over_idx = (phi>=0);
z_over = z(over_idx);

% Plot t vs h for type 0
figure;
semilogy(t0, h0, 'o-', t(over_idx), h(over_idx),'x-');
xlabel('t');
ylabel('h');
title('t vs h for type = 0');
grid on;

figure;
plot(t0, z0, 'o-', t(over_idx), z_over,'x--');
xlabel('t');
ylabel('z');
title('t vs z for type = 0');
grid on;

figure;
plot(t0, zdot0, '-');
xlabel('t');
ylabel('zdot');
title('t vs zdot for type = 0');
grid on;

figure;
plot(t0, f0, '-');
xlabel('t');
ylabel('fz');
title('t vs fz for type = 0');
grid on;