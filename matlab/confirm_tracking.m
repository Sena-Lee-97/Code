function [save_cell_info, num_out_cell, no_out_cell] = confirm_tracking( ...
    img_path, save_m_path, total_cell_objs, s_c, save_name, last_frame_idx)
%CONFIRM_TRACKING Validate tracking results and recover overlapping tracks.
%
% Input
%   img_path         : image directory
%   save_m_path      : directory for saving intermediate/final results
%   total_cell_objs  : per-frame detection objects
%   s_c              : raw tracking results from main_tracking
%   save_name        : dataset / experiment name
%   last_frame_idx   : last frame index

%
% Output
%   save_cell_info   : final validated tracking results
%   num_out_cell     : number of incomplete tracks
%   no_out_cell      : complete tracks before recovery step
%
% Description
%   This function:
%   1) separates incomplete and complete trajectories,
%   2) detects possible duplicated / merged tracks,
%   3) tries to recover valid trajectories,
%   4) saves intermediate outputs.

    global loss_s_c

    %% =========================
    % Initialization
    % ==========================

    total_frame = length(total_cell_objs);

    %% =========================
    % Separate incomplete tracks
    % ==========================
    out_cell = [];
    out_cell_first = [];
    z = 0;

    for i = 1:length(s_c)
        if length(s_c{1,i}) ~= total_frame
            z = z + 1;
            out_cell(z,1) = s_c{1,i}{1,1}(6,1);
            out_cell_first(z,1) = s_c{1,i}{1,1}(1,1);
            out_cell_first(z,2) = s_c{1,i}{1,1}(2,1);
        end
    end

    num_out_cell = length(out_cell);

    %% =========================
    % Store incomplete trajectories for reference
    % ==========================
    track_cell = {};
    s = 0;

    for x = 1:length(out_cell)
        i = out_cell(x,1);
        s = s + 1;

        for j = 1:length(s_c{1,i})
            track_cell{s,1}(j,1:2) = s_c{1,i}{j,1}(1:2,:)';
        end
    end

    %% =========================
    % Build image mask (reserved for visualization / compatibility)
    % ==========================
    img = imread(fullfile(img_path, sprintf('IMG_%d.jpg', 0)));
    bw = imbinarize(img);
    bw = imfill(bw, 'holes');

    se = strel('disk', 50);
    opened_bw = imopen(bw, se);

    valid_mask = ones(size(img));
    valid_mask(1:10,:) = 0;
    valid_mask(1014:1024,:) = 0;
    valid_mask = valid_mask .* opened_bw; %#ok<NASGU>

    %% =========================
    % Collect complete tracks
    % ==========================
    no_out_cell = {};
    z = 0;

    for i = 1:length(s_c)
        if length(s_c{1,i}) == total_frame
            z = z + 1;
            no_out_cell{1,z} = s_c{1,i};
        end
    end

    %% =========================
    % Check duplicate positions on random frames
    % ==========================
    confirm_frame = generate_random_frames(last_frame_idx, 5);
    [moving, duplicate_points] = check_duplicate_tracks(no_out_cell, confirm_frame);

    %% =========================
    % Estimate possible track loss
    % ==========================
    [duplicate_score, duplicate_indices] = summarize_duplicates(moving, duplicate_points, confirm_frame); %#ok<ASGLU>

    % Default output: complete tracks only
    save_cell_info = no_out_cell;

    if sum(cell2mat(duplicate_score)) ~= 0
        %% -------------------------
        % Select tracks with last-frame duplication
        % --------------------------
        last_loss = duplicate_indices{1, length(confirm_frame)};
        loss_s_c = {};

        for i = 1:length(last_loss)
            loss_s_c{i,1} = no_out_cell{1, last_loss(i,1)};
        end

        %% -------------------------
        % Group duplicated trajectories by final coordinate match
        % --------------------------
        reference_frame = min(total_frame, size(loss_s_c{1,1}, 1));
        matched = build_match_matrix(loss_s_c, reference_frame);

        unique_groups = unique(matched, 'rows');

        %% -------------------------
        % Expand grouped trajectories into xy sequences
        % --------------------------
        xyxy = build_grouped_xy_sequences(loss_s_c, unique_groups);

        %% -------------------------
        % Select recoverable trajectories
        % --------------------------
        save_xys = select_recoverable_tracks(xyxy);

        add_save_s_c = {};
        s = 0;

        for i = 1:length(save_xys)
            if save_xys(i,1) ~= inf
                s = s + 1;
                save_idx = unique_groups(i, save_xys(i,1));
                add_save_s_c{s,1} = loss_s_c{save_idx,1};
            end
        end

        add_save_s_c = add_save_s_c';

        %% -------------------------
        % Remove duplicated tracks from complete set
        % --------------------------
        save_s_c = no_out_cell;

        for i = 1:length(last_loss)
            save_s_c{1, last_loss(i,1)} = {};
        end

        save_s_c = save_s_c(~cellfun(@isempty, save_s_c));
        save_cell_info = horzcat(save_s_c, add_save_s_c);

        %% -------------------------
        % Save intermediate recovery results
        % --------------------------
        loss = last_loss';
        loss_s_c = {};

        for i = 1:length(loss)
            loss_s_c{i,1} = no_out_cell{1, loss(1,i)};
        end

        % save(fullfile(save_m_path, sprintf('%s_last_loss.mat', save_name)), 'loss_s_c');
        % save(fullfile(save_m_path, sprintf('%s_add_save_s_c.mat', save_name)), 'add_save_s_c');
        % save(fullfile(save_m_path, sprintf('%s_save_cell_info.mat', save_name)), 'save_cell_info');
    end
end


%% =========================================================
% Local functions
%% =========================================================
function random_frame = generate_random_frames(last_frame_idx, n)
    random_frame = zeros(n,1);
    for i = 1:n
        random_frame(i,1) = round(last_frame_idx * rand);
    end
    random_frame = sort(random_frame);
    random_frame(random_frame < 1) = 1;
end


function [moving, duplicate_points] = check_duplicate_tracks(no_out_cell, confirm_frame)
    moving = [];
    duplicate_points = {};

    for s = 1:length(confirm_frame)
        frame_idx = confirm_frame(s);
        all_points = [];

        for i = 1:length(no_out_cell)
            if frame_idx > length(no_out_cell{1,i})
                continue;
            end
            track_point = no_out_cell{1,i}{frame_idx,1}(1:2,:)';
            all_points = [all_points; track_point]; %#ok<AGROW>
        end

        for i = 1:length(no_out_cell)
            if frame_idx > length(no_out_cell{1,i})
                continue;
            end

            track_point = no_out_cell{1,i}{frame_idx,1}(1:2,:)';
            is_same = track_point == all_points;

            if sum(is_same(:,1) & is_same(:,2)) == 1
                moving(i,s) = 1; %#ok<AGROW>
                idx = find(is_same(:,1) & is_same(:,2) == 1);
                duplicate_points{i,s} = all_points(idx,1:2); %#ok<AGROW>

            elseif sum(is_same(:,1) & is_same(:,2)) > 1
                moving(i,s) = 2; %#ok<AGROW>
                idx = find(is_same(:,1) & is_same(:,2) == 1);
                duplicate_points{i,s} = all_points(idx,1:2); %#ok<AGROW>
            end
        end
    end
end


function [duplicate_score, duplicate_indices] = summarize_duplicates(moving, duplicate_points, confirm_frame)
    duplicate_score = {};
    duplicate_indices = {};

    for i = 1:length(confirm_frame)
        duplicate_tracks = find(moving(:,i) > 1);

        if isempty(duplicate_tracks)
            duplicate_score{i} = 0; %#ok<AGROW>
            duplicate_indices{i} = 0; %#ok<AGROW>
            continue;
        end

        duplicate_indices{i} = duplicate_tracks; %#ok<AGROW>

        nums = [];
        for j = 1:length(duplicate_tracks)
            pts = duplicate_points{duplicate_tracks(j,1), i};
            nums = [nums, length(pts)]; %#ok<AGROW>
        end

        find_2 = sum(nums == 2) / 2;
        find_3 = sum(nums == 3) / 3 * 2;
        find_4 = sum(nums == 4) / 4 * 3;
        find_5 = sum(nums == 5) / 5 * 4;

        duplicate_score{i} = find_2 + find_3 + find_4 + find_5; %#ok<AGROW>
    end

    duplicate_score = duplicate_score';
end


function matched = build_match_matrix(loss_s_c, reference_frame)
    matched = [];

    for i = 1:length(loss_s_c)
        ref_xy = loss_s_c{i,1}{reference_frame,1}(1:2,:);
        s = 0;

        for j = 1:length(loss_s_c)
            dif_xy = loss_s_c{j,1}{reference_frame,1}(1:2,:);
            if sum(ref_xy - dif_xy) == 0
                s = s + 1;
                matched(i,s) = j; %#ok<AGROW>
            end
        end
    end
end


function xyxy = build_grouped_xy_sequences(loss_s_c, unique_groups)
    xyxy = {};

    for i = 1:size(unique_groups, 1)
        for j = 1:size(unique_groups, 2)
            idx = unique_groups(i,j);

            if idx == 0
                xyxy{i,j} = 0; %#ok<AGROW>
                continue;
            end

            xy = [];
            for z = 1:length(loss_s_c{1,1})
                xy = [xy; loss_s_c{idx,1}{z,1}(1:2,:)']; %#ok<AGROW>
            end
            xyxy{i,j} = xy; %#ok<AGROW>
        end
    end
end


function save_xys = select_recoverable_tracks(xyxy)
    c_f = 5;
    save_xys = inf(size(xyxy,1), 1);

    for i = 1:size(xyxy,1)
        num_cols = size(xyxy,2);

        if isequal(xyxy{i,num_cols}, 0)
            threshold = 3;
        else
            threshold = 5;
        end

        best_idx = inf;
        best_value = inf;

        for a = 1:num_cols
            if isequal(xyxy{i,a}, 0)
                continue;
            end

            for b = a+1:num_cols
                if isequal(xyxy{i,b}, 0)
                    continue;
                end

                same_idx = find(sum(xyxy{i,a}(:,1:2) == xyxy{i,b}(:,1:2), 2) == 2, 1, 'first');

                if isempty(same_idx) || same_idx <= c_f
                    continue;
                end

                da = sum(xyxy{i,a}(same_idx,1:2) - xyxy{i,a}(same_idx-c_f,1:2));
                db = sum(xyxy{i,b}(same_idx,1:2) - xyxy{i,b}(same_idx-c_f,1:2));

                candidates = abs([da, db]);
                [local_best, local_idx] = min(candidates);

                if local_best < best_value
                    best_value = local_best;
                    pair_indices = [a, b];
                    best_idx = pair_indices(local_idx);
                end
            end
        end

        if best_value <= threshold
            save_xys(i,1) = best_idx;
        end
    end
end