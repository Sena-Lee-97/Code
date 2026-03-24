function [finish] = save_cell_xy(save_m_path, dataset, save_name, save_cell_info)
%SAVE_CELL_XY Save sorted cell trajectories as MAT and CSV files.
%
% Input
%   save_m_path     : directory to save output files
%   dataset         : dataset name
%   save_name       : base name for MAT file
%   save_cell_info  : final validated cell tracking results
%
% Output
%   finish          : status string ('finish')
%
% Description
%   This function:
%   1) extracts first-frame cell information,
%   2) sorts cells by their first-frame position,
%   3) reassigns cell labels based on sorted order,
%   4) saves the reordered tracking results as MAT and CSV files.
%
% Output files
%   - <save_name>_sorted_save_cell_info.mat
%   - <dataset>cell_xy.csv

    %% =========================
    % Extract first-frame information
    % ==========================
    first_frame_info = [];

    for i = 1:length(save_cell_info)
        x = save_cell_info{1,i}{1,1}(1);   % first-frame x
        y = save_cell_info{1,i}{1,1}(2);   % first-frame y
        label = save_cell_info{1,i}{1,1}(6); % original cell ID

        first_frame_info(i,1) = label;
        first_frame_info(i,2) = x;
        first_frame_info(i,3) = y;
    end

    % Add original cell index
    cell_idx = (1:length(first_frame_info))';
    first_frame_info = [first_frame_info, cell_idx];

    %% =========================
    % Sort cells by first-frame position
    % ==========================
    % Columns:
    % 1 = original label
    % 2 = x
    % 3 = y
    % 4 = original index in save_cell_info
    coords = first_frame_info;

    % Sort by y first
    sorted_coords_y = sortrows(coords, 3);

    % Then sort within each y group by label/x ordering
    sorted_coords = [];
    unique_y_coords = unique(sorted_coords_y(:,3));

    for i = 1:length(unique_y_coords)
        rows_current_y = sorted_coords_y(sorted_coords_y(:,3) == unique_y_coords(i), :);
        sorted_coords = [sorted_coords; sortrows(rows_current_y, 1)]; %#ok<AGROW>
    end

    %% =========================
    % Rebuild sorted tracking results with new labels
    % ==========================
    sorted_save_cell_info = {};

    for c = 1:length(save_cell_info)
        cell_array = save_cell_info{1, sorted_coords(c,4)};
        new_cell_array = {};

        for i = 1:numel(cell_array)
            x = cell_array{i}(1);
            y = cell_array{i}(2);
            new_label = c;
            cell_rad = cell_array{i}(7);

            save_new_label = [x; y; new_label; cell_rad];
            new_cell_array{i,1} = save_new_label;
        end

        sorted_save_cell_info{1,c} = new_cell_array; %#ok<AGROW>
    end

    %% =========================
    % Save sorted MAT file
    % ==========================
    mat_filename = fullfile(save_m_path, sprintf('%s_sorted_save_cell_info.mat', save_name));
    save(mat_filename, 'sorted_save_cell_info');

    %% =========================
    % Build x,y coordinate matrix
    % ==========================
    xys = [];

    for i = 1:length(sorted_save_cell_info)
        cell_track = sorted_save_cell_info{1,i};
        xy = zeros(length(cell_track), 2);

        for j = 1:length(cell_track)
            xy(j,1) = cell_track{j,1}(1);
            xy(j,2) = cell_track{j,1}(2);
        end

        xys = [xys, xy]; %#ok<AGROW>
    end

    %% =========================
    % Save CSV
    % ==========================
    csv_filename = fullfile(save_m_path, sprintf('%scell_xy.csv', dataset));
    writematrix(xys, csv_filename);

    finish = 'finish';
end