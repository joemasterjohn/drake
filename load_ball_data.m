function data = load_ball_data(filename)
    % Load speculative contact surface data from a file where each line is:
    % Read the entire file into a matrix
    raw = readmatrix(filename);

    s = size(raw, 2);

    if s ~= 20 & s ~= 10
        error('Each line must contain exactly 20 (6dofs) or 10 (3dofs) values');
    end

     % Extract components
    if s == 20
        data.t = raw(:, 1);     % double t
        data.q = raw(:, 2:8);   % Vector7 q
        data.v = raw(:, 9:14);  % Vector6 v
        data.f = raw(:, 15:20); % Vector6 f
    elseif s == 10
        data.t = raw(:, 1);     % double t
        data.q = raw(:, 2:4);   % Vector7 q
        data.v = raw(:, 5:7);  % Vector6 v
        data.f = raw(:, 8:10); % Vector6 f
    end
end