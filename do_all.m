dt = [1e-2; 4e-3; 1e-3; 4e-4; 1e-4];
[data, data_spec] = load_data();

[error_q, error_v] = calc_error(data);
[error_q_s, error_v_s] = calc_error(data_spec);

figure(1);
regular = plot(data(1).t, data(1).q(:,1));
set(regular,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data(i).t, data(i).q(:,1));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'southeast');
axis([0 1 0 1]);
title('X (No Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('x position [m]', 'FontName', 'Time', 'FontSize', 16);

figure(2);
spec = plot(data_spec(1).t, data_spec(1).q(:,1));
set(spec,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data_spec(i).t, data_spec(i).q(:,1));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'southeast');
axis([0 1 0 1]);
title('X (Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('x position [m]', 'FontName', 'Time', 'FontSize', 16);


figure(3);
regular = plot(data(1).t, data(1).q(:,2));
set(regular,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data(i).t, data(i).q(:,2));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'northeast');
axis([0 1 0.049 0.051]);
title('Y (No Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('y position [m]', 'FontName', 'Time', 'FontSize', 16);

figure(4);
spec = plot(data_spec(1).t, data_spec(1).q(:,2));
set(spec,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data_spec(i).t, data_spec(i).q(:,2));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'northeast');
axis([0 1 0.049 0.051]);
title('Y (Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('y position [m]', 'FontName', 'Time', 'FontSize', 16);

figure(5);
err = loglog(dt, error_q, '-o', dt, error_q_s, '-o');
%axis([0 1 0.049 0.051]);
title('Error');
set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time step [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('Error [m]', 'FontName', 'Time', 'FontSize', 16);
legend("regular", "speculative");

figure(6);
regular = plot(data(1).t, data(1).f(:,1));
set(regular,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data(i).t, data(i).f(:,1));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'northeast');
%axis([0 1 0.049 0.051]);
title('Contact force Fx (No Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('Fx [N]', 'FontName', 'Time', 'FontSize', 16);

figure(7);
regular = plot(data_spec(1).t, data_spec(1).f(:,1));
set(regular,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data_spec(i).t, data_spec(i).f(:,1));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'northeast');
%axis([0 1 0.049 0.051]);
title('Contact force Fx (Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('Fx [N]', 'FontName', 'Time', 'FontSize', 16);

figure(8);
regular = plot(data(1).t, data(1).f(:,1));
set(regular,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data(i).t, data(i).f(:,1));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'northeast');
%axis([0 1 0.049 0.051]);
title('Contact force Fy (No Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('Fy [N]', 'FontName', 'Time', 'FontSize', 16);

figure(9);
regular = plot(data_spec(1).t, data_spec(1).f(:,2));
set(regular,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data_spec(i).t, data_spec(i).f(:,2));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'northeast');
%axis([0 1 0.049 0.051]);
title('Contact force Fy (Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('Fy [N]', 'FontName', 'Time', 'FontSize', 16);



figure(10);
regular = plot(data(1).t, data(1).f(:,3));
set(regular,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data(i).t, data(i).f(:,3));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'northeast');
%axis([0 1 0.049 0.051]);
title('Contact Torque (No Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('tau [Nm]', 'FontName', 'Time', 'FontSize', 16);

figure(11);
regular = plot(data_spec(1).t, data_spec(1).f(:,3));
set(regular,'LineWidth',1.2,'MarkerSize',6);
hold on;
for i=2:5
    plot(data_spec(i).t, data_spec(i).f(:,3));
end
hold off;
legend('1e-2', '4e-3', '1e-3', '4e-4', '1e-4', 'Location', 'northeast');
%axis([0 1 0.049 0.051]);
title('Contact Torque (Speculative)');

set(gca, 'FontName', 'Time', 'FontSize', 16);
set(gca,'LineWidth',1.2,'TickLength',[0.02 0.02]);
xlabel('Time [s]', 'FontName', 'Time', 'FontSize', 16);
ylabel('tau [Nm]', 'FontName', 'Time', 'FontSize', 16);