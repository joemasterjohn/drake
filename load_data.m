function [data, data_spec] = load_data()
    data = [load_ball_data("data_1e-2.txt")];
    data(end+1) = load_ball_data("data_4e-3.txt");
    data(end+1) = load_ball_data("data_1e-3.txt");
    data(end+1) = load_ball_data("data_4e-4.txt");
     data(end+1) = load_ball_data("data_1e-4.txt");
    data(end+1) = load_ball_data("data_1e-5.txt");
    data_spec = [load_ball_data("data_spec_1e-2.txt")];
    data_spec(end+1) = load_ball_data("data_spec_4e-3.txt");
    data_spec(end+1) = load_ball_data("data_spec_1e-3.txt");
    data_spec(end+1) = load_ball_data("data_spec_4e-4.txt");
    data_spec(end+1) = load_ball_data("data_spec_1e-4.txt");
    data_spec(end+1) = load_ball_data("data_spec_1e-5.txt");
end