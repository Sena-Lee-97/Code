function [s_c] = main_tracking(total_cell_objs, save_name, img_path, save_m_path, fast_moving_frame, first_frame_idx)
%MAIN_TRACKING Track cells across frames using detection results.
%
% Input
%   total_cell_objs    : cell array containing per-frame detected objects
%   save_name          : dataset / experiment name
%   img_path           : image directory
%   save_m_path        : save directory (reserved for future use)
%   fast_moving_frame  : indices of fast-moving frames
%   first_frame_idx    : start object index in first frame
%
% Output
%   s_c                : cell array of tracked objects for each starting cell
%
% Description
%   This function performs heuristic cell tracking using:
%   - nearest-neighbor matching
%   - ROI-based forward search
%   - overlap (IoU-like) validation
%   - different rules for fast-motion and normal-motion intervals

    %% =========================
    % Build valid image mask
    % ==========================
    first_img_path = fullfile(img_path, sprintf('IMG_%d.jpg', 0));
    img = imread(first_img_path);

    bw = imbinarize(img);
    bw = imfill(bw, 'holes');

    se = strel('disk', 50);
    opened_bw = imopen(bw, se);

    valid_mask = ones(size(img));
    valid_mask(1:10, :) = 0;
    valid_mask(1014:1024, :) = 0;
    valid_mask = valid_mask .* opened_bw;

    %% =========================
    % Initialization
    % ==========================
    frame_indices = 1:length(total_cell_objs);
    num_frames = length(frame_indices);

    ROI_th = 20;
    find_next = 10;
    interval = 50;

    s_c = {};
    each_cell_times = {};

    %% =========================
    % Expand fast-moving frame window
    % ==========================
    expanded_fast_frames = [];
    for i = 1:length(fast_moving_frame)
        current_range = fast_moving_frame(i)-interval : fast_moving_frame(i)+interval;
        expanded_fast_frames = [expanded_fast_frames, current_range]; %#ok<AGROW>
    end
    expanded_fast_frames = unique(expanded_fast_frames);

    %% =========================
    % Main tracking loop
    % ==========================
    num_first_frame_cells = size(total_cell_objs{1,1}, 2);

    for q = first_frame_idx:num_first_frame_cells
        t_cell = tic;

        matched_objs = {};
        matched_objs{1} = total_cell_objs{1,1}(:, q);
        matched_objs{1}(6,:) = matched_objs{1}(4,:);

        for k = 1:num_frames-1
            cur_obj = matched_objs{k,1};
            next_frame_objs = total_cell_objs{k+1,1};

            if isempty(cur_obj) || isempty(next_frame_objs)
                break;
            end

            cur_xy = cur_obj(1:2,:)';
            x_cur = round(cur_xy(1,1));
            y_cur = round(cur_xy(1,2));

            if y_cur < 1 || y_cur > size(valid_mask,1) || x_cur < 1 || x_cur > size(valid_mask,2)
                break;
            end

            if valid_mask(y_cur, x_cur) == 0
                break;
            end

            %% -------------------------
            % Set parameters by motion regime
            % --------------------------
            if isempty(next_frame_objs) || size(next_frame_objs,1) < 7 || isempty(next_frame_objs(7,:))
                rad_mean = 10;   % fallback
            else
                rad_mean = mean(next_frame_objs(7,:), 'omitnan');
                if ~isfinite(rad_mean) || rad_mean <= 0
                    rad_mean = 10;   % fallback
                end
            end

            if ismember(k, expanded_fast_frames)
                distance_threshold = floor(rad_mean / 2);
                future_search_threshold = rad_mean;
                iou_threshold = 0.40;
                radius_divider = 1.0;
                mode_flag = 1;   % fast-moving interval
            else
                distance_threshold = 0.5;
                future_search_threshold = floor(rad_mean / 2.5);
                iou_threshold = 0.45;
                radius_divider = 1.5;
                mode_flag = 2;   % normal interval
            end

            %% -------------------------
            % Find nearest object in next frame
            % --------------------------
            [nearest_obj, nearest_dist] = find_nearest_object(cur_obj, next_frame_objs);

            if nearest_dist < distance_threshold
                candidate_obj = nearest_obj;
                candidate_dist = nearest_dist;
            else
                [candidate_obj, candidate_dist] = search_future_roi( ...
                    cur_obj, total_cell_objs, k, num_frames, ROI_th, find_next);
            end

            if isempty(candidate_obj)
                candidate_obj = cur_obj;
                candidate_dist = inf;
            end

            %% -------------------------
            % IoU-based validation
            % --------------------------
            [iou_value, is_valid_overlap] = compute_circle_box_iou(cur_obj, candidate_obj, radius_divider);

            if is_valid_overlap && iou_value >= iou_threshold
                candidate_obj(5,1) = k + 1;
                candidate_obj(6,1) = cur_obj(6,1);
                matched_objs{k+1,1} = candidate_obj;

            else
                matched_objs{k+1,1} = resolve_ambiguous_match( ...
                    cur_obj, total_cell_objs, k, num_frames, ...
                    future_search_threshold, find_next, mode_flag);
            end
        end

        s_c{1,q} = matched_objs; %#ok<AGROW>
        each_cell_times{q,1} = toc(t_cell); %#ok<AGROW>

        fprintf('Tracked cell %d / %d\n', q, num_first_frame_cells);
    end
end


%% =========================================================
% Local functions
%% =========================================================
function [nearest_obj, nearest_dist] = find_nearest_object(cur_obj, next_objs)
    nearest_obj = [];
    nearest_dist = inf;

    if isempty(cur_obj) || isempty(next_objs)
        return;
    end

    pcur = cur_obj(1:2,:);
    pnex = next_objs(1:2,:);

    aa = sum(pnex .* pnex, 1);
    bb = sum(pcur .* pcur, 1);
    d = sqrt(abs(aa(ones(size(bb,2),1), :)' + bb(ones(size(aa,2),1), :) - 2 * pnex' * pcur));

    nearest_dist = min(d, [], 1);
    nearest_obj = next_objs(:, d == min(d, [], 1));

    if size(nearest_obj, 2) > 1
        nearest_obj = nearest_obj(:,1);
    end
end


function [candidate_obj, candidate_dist] = search_future_roi(cur_obj, total_cell_objs, k, num_frames, ROI_th, find_next)
    candidate_obj = [];
    candidate_dist = inf;

    x_range = (cur_obj(1,:) - ROI_th):(cur_obj(1,:) + ROI_th);
    y_range = (cur_obj(2,:) - ROI_th):(cur_obj(2,:) + ROI_th);

    future_pool = [];

    search_end = min(k + find_next, num_frames);

    for z = k+1:search_end
        future_objs = total_cell_objs{z,1};

        if isempty(future_objs)
            continue;
        end

        roi_idx = ...
            (min(x_range) < future_objs(1,:)) & (future_objs(1,:) < max(x_range)) & ...
            (min(y_range) < future_objs(2,:)) & (future_objs(2,:) < max(y_range));

        if any(roi_idx)
            future_pool = [future_pool, future_objs(:, roi_idx)]; %#ok<AGROW>
        end
    end

    if isempty(future_pool)
        return;
    end

    pcur = cur_obj(1:2,:);
    pnex = future_pool(1:2,:);

    aa = sum(pnex .* pnex, 1);
    bb = sum(pcur .* pcur, 1);
    d = sqrt(abs(aa(ones(size(bb,2),1), :)' + bb(ones(size(aa,2),1), :) - 2 * pnex' * pcur));

    candidate_dist = min(d, [], 1);
    candidate_obj = future_pool(:, d == min(d, [], 1));

    if size(candidate_obj, 2) > 1
        candidate_obj = candidate_obj(:,1);
    end
end


function matched_obj = resolve_ambiguous_match(cur_obj, total_cell_objs, k, num_frames, th, find_next, mode_flag)
    matched_obj = cur_obj;
    matched_obj(5,1) = k + 1;

    if isempty(cur_obj) || size(cur_obj,1) < 2
        return;
    end

    if ~isscalar(th) || ~isfinite(th) || th <= 0
        th = 10;   % fallback
    end

    cur_xy = cur_obj(1:2,:)';

    if size(cur_xy,1) < 1 || size(cur_xy,2) < 2
        return;
    end
    
    if ~isscalar(th) || ~isfinite(th) || th <= 0
        th = 10;
    end
    
    cx = cur_xy(1,1);
    cy = cur_xy(1,2);
    
    x_range = (cx - th):(cx + th);
    y_range = (cy - th):(cy + th);


    roi_candidates = {};
    existence = [];
    z = 0;

    search_end = min(k + find_next, num_frames);

    for j = k+1:search_end
        z = z + 1;
        next_objs = total_cell_objs{j,1};

        if isempty(next_objs)
            roi_candidates{z} = []; %#ok<AGROW>
            existence = [existence, 0]; %#ok<AGROW>
            continue;
        end

        next_xy = next_objs(1:2,:);

        roi_idx = ...
            (min(x_range) < next_xy(1,:)) & (next_xy(1,:) < max(x_range)) & ...
            (min(y_range) < next_xy(2,:)) & (next_xy(2,:) < max(y_range));

        num_roi = sum(roi_idx);

        if num_roi > 1
            pcur = [cx; cy];
            pnex = next_objs(1:2, roi_idx == 1);
            pt = next_objs(:, roi_idx == 1);

            aa = sum(pnex .* pnex, 1);
            bb = sum(pcur .* pcur, 1);
            d = sqrt(abs(aa(ones(size(bb,2),1), :)' + bb(ones(size(aa,2),1), :) - 2 * pnex' * pcur))';

            nearest_roi = pt(:, d == min(d));
            if size(nearest_roi, 2) > 1
                nearest_roi = nearest_roi(:,1);
            end
            roi_candidates{z} = nearest_roi;

        elseif num_roi == 1
            roi_candidates{z} = next_objs(:, roi_idx);

        else
            roi_candidates{z} = [];
        end

        existence = [existence, num_roi]; %#ok<AGROW>
    end

    if sum(existence) == 0
        return;
    end

    valid_idx = find(existence ~= 0);
    closest_future_idx = min(valid_idx);

    if closest_future_idx >= 5
        return;
    end

    add_obj = roi_candidates{closest_future_idx};
    if isempty(add_obj)
        return;
    end

    pcur = cur_obj(1:2,:);
    pnex = add_obj(1:2,:);

    aa = sum(pnex .* pnex, 1);
    bb = sum(pcur .* pcur, 1);
    d = sqrt(abs(aa(ones(size(bb,2),1), :)' + bb(ones(size(aa,2),1), :) - 2 * pnex' * pcur));
    d = min(d, [], 1);

    if d < th
        if mode_flag == 1
            % fast-moving region: directly use detected future object
            add_obj(5,1) = k + 1;
            matched_obj = add_obj;
        else
            % normal region: interpolate one step toward future object
            diff_xy = (pcur - pnex) / (closest_future_idx + 1);
            interp_xy = pcur - diff_xy;

            add_obj(1:2,1) = interp_xy;
            add_obj(5,1) = k + 1;
            matched_obj = add_obj;
        end
    end
end


function [iou_value, is_valid] = compute_circle_box_iou(cur_obj, next_obj, radius_divider)
    iou_value = 0;
    is_valid = false;

    if isempty(cur_obj) || isempty(next_obj)
        return;
    end

    cur_xy = cur_obj(1:2,:)';
    next_xy = next_obj(1:2,:)';

    cur_r = cur_obj(7,:) / radius_divider;
    next_r = next_obj(7,:) / radius_divider;

    x1 = cur_xy(1,1) - cur_r;
    x2 = cur_xy(1,1) + cur_r;
    y1 = cur_xy(1,2) - cur_r;
    y2 = cur_xy(1,2) + cur_r;

    x1n = next_xy(1,1) - next_r;
    x2n = next_xy(1,1) + next_r;
    y1n = next_xy(1,2) - next_r;
    y2n = next_xy(1,2) + next_r;

    inter_w = min(x2, x2n) - max(x1, x1n);
    inter_h = min(y2, y2n) - max(y1, y1n);

    if inter_w < 0 || inter_h < 0
        return;
    end

    inter_area = inter_w * inter_h;
    area1 = (x2 - x1) * (y2 - y1);
    area2 = (x2n - x1n) * (y2n - y1n);

    union_area = area1 + area2 - inter_area;

    if area1 == inter_area
        union_area = inter_area;
    end

    if union_area <= 0
        return;
    end

    iou_value = inter_area / union_area;
    is_valid = true;
end