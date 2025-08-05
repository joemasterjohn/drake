function y = log_energy_target(x, p)

R = p.R;
E = p.E;
delta = p.delta;
m = p.m;
g = p.g;
h = p.h;

Ad = 4*pi*delta*(R+delta);
e = x/delta;

V = e.*e + 2*e +2*log(1-e);
V = -V * E*Ad*delta/2;

y = V - m*g*h;
