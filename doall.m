R = 0.1;
E = 1e9;
m = 0.1;
h = 1;
g = 9.81;
delta = 1e-4;


p.R = R;
p.E = E;
p.delta = delta;
p.m = m;
p.g =g;
p.h = h;

target = @(x) log_energy_target(x, p);

x = fzero(target, delta/2)/delta

