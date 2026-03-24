function [total_cell_objs, make_objs_time] = make_objs_file(txt_path, save_m_path, save_name)
%MAKE_OBJS_FILE Build per-frame object arrays from detection txt files.
%
% Input
%   txt_path     : directory containing detection result txt files
%                  (e.g., IMG_0.txt, IMG_1.txt, ...)
%   save_m_path  : directory to save .mat file
%   save_name    : base name for saved file
%
% Output
%   total_cell_objs : cell array, one cell per frame
%   make_objs_time  : elapsed processing time (sec)
%
% Each frame object array has the following rows:
%   1 : x coordinate
%   2 : y coordinate
%   3 : status flag (initialized as 1)
%   4 : local object index
%   5 : frame index
%   6 : reserved value (initialized as 0)
%   7 : estimated radius = (w + h) / 4

    txt_files = dir(fullfile(txt_path, '*.txt'));
    frame_num = length(txt_files);

    final_objs = cell(frame_num, 1);

    tic;

    for i = 1:frame_num
        txt_file = fullfile(txt_path, sprintf('IMG_%d.txt', i - 1));

        if ~isfile(txt_file)
            warning('Detection file not found: %s', txt_file);
            final_objs{i} = [];
            continue;
        end

        data = load(txt_file);

        if isempty(data)
            final_objs{i} = [];
            fprintf('Frame %d: empty detection file\n', i);
            continue;
        end

        % Expected format:
        % column 1 = x
        % column 2 = y
        % column 3 = width
        % column 4 = height
        xmin = data(:, 1);
        ymin = data(:, 2);
        w = data(:, 3);
        h = data(:, 4);

        rad = (w + h) / 4;

        num_objects = size(data, 1);
        status_flag = ones(1, num_objects);
        object_index = 1:num_objects;
        frame_index = i * ones(1, num_objects);
        reserved = zeros(1, num_objects);

        frame_objs = [
            xmin';
            ymin';
            status_flag;
            object_index;
            frame_index;
            reserved;
            rad'
        ];

        final_objs{i} = frame_objs;

        fprintf('Processed frame %d / %d\n', i, frame_num);
    end

    total_cell_objs = final_objs;
    make_objs_time = toc;

    save_file = fullfile(save_m_path, sprintf('%s_total_objs.mat', save_name));
    save(save_file, 'total_cell_objs');
end