function data = load_speculative_contact_surface(filename)
    % Load speculative contact surface data from a file where each line is:
    % Vector3 p_WC, double time_of_contact, Vector3 zhat_BA_W, double
    % coefficient, Vector3 nhat_BA_W, Vector3 grad_eA_W, Vector3 grad_eB_W,
    % Vector3 p_A, string type_A, int[3] indices_A, Vector3 q_B, string
    % type_B, int[3] indices_B, double squared_dist, int tetA, int tetB

    % Read the entire file into a matrix
    raw = readmatrix(filename);

    if size(raw, 2) ~= 34
        error('Each line must contain exactly 33 values');
    end

    % Extract components
    data.p_WC = raw(:, 1:3);           % Vector3 p_WC
    data.time_of_contact = raw(:, 4);  % double time_of_contact
    data.zhat_BA_W = raw(:, 5:7);      % Vector3 zhat_BA_W
    data.coefficient = raw(:, 8);      % double coefficient
    data.nhat_BA_W = raw(:, 9:11);     % Vector3 nhat_BA_W
    data.grad_eA_W = raw(:, 12:14);    % Vector3 grad_eA_W
    data.grad_eB_W = raw(:, 15:17);     % Vector3 grad_eB_W
    data.p_A = raw(:, 18:20);          % Vector3 p_A
    data.type_A = raw(:, 21);          % String type_A
    data.indices_A = raw(:, 22:24);    % int[3] indices_A
    data.q_B = raw(:, 25:27);          % Vector3 q_B
    data.type_B = raw(:, 28);          % String type_B
    data.indices_B = raw(:, 29:31);    % int[3] indices_B
    data.squared_dist = raw(:, 32);       % double squared_dist
    data.tetA = raw(:, 33);               % int tetA
    data.tetB = raw(:, 34);               % int tetB
end