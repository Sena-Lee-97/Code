function [fast_moving_frame, threshold_value] = first_frame( ...
    total_cell_objs, img_path, first_frame_idx, save_m_path, save_name)
%FIRST_FRAME Estimate fast-moving frames from the first tracked cell set.
%
% Input
%   total_cell_objs   : cell array containing per-frame object information
%   img_path          : directory of image sequence
%   first_frame_idx   : first frame index used in tracking
%   save_m_path       : path to save output plot
%   save_name         : base name for saved figure
%
% Output
%   fast_moving_frame : indices of fast-moving frames
%   threshold_value   : adaptive threshold used for peak selection
%
% Description
%   This function:
%   1) reads the first frame image,
%   2) builds a valid image mask,
%   3) estimates motion between adjacent frames,
%   4) detects peaks from the motion curve,
%   5) returns fast-moving frame indices.

    base_threshold_ratio = 0.4;

    %% =========================
    % Load first frame and generate valid mask
    % ==========================
    first_image_path = fullfile(img_path, sprintf('IMG_%d.jpg', first_frame_idx - 1));
    img = imread(first_image_path);

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

    matched_tracks = {};
    motion_values = [];

    find_next = 10;
    roi_threshold = 20; %#ok<NASGU>  % kept for compatibility / future tuning

    tic;

    %% =========================
    % Track each cell from the first frame
    % ==========================
    num_cells_first_frame = size(total_cell_objs{1,1}, 2);

    for q = 1:num_cells_first_frame
        matched_objs = {};
        matched_objs{1} = total_cell_objs{1,1}(:, q);
        matched_objs{1}(6, :) = matched_objs{1}(4, :);

        for k = 1:num_frames - 1
            cur_obj = matched_objs{k, 1};
            next_candidates = total_cell_objs{k + 1, 1};

            if isempty(cur_obj) || isempty(next_candidates)
                break;
            end

            cur_xy = cur_obj(1:2, :)';

            % Stop if current point is outside valid mask
            x_cur = round(cur_xy(1,1));
            y_cur = round(cur_xy(1,2));

            if y_cur < 1 || y_cur > size(valid_mask,1) || x_cur < 1 || x_cur > size(valid_mask,2)
                break;
            end

            if valid_mask(y_cur, x_cur) == 0
                break;
            end

            %% -------------------------
            % Find nearest object in next frame
            % --------------------------
            pcur = cur_obj(1:2, :);
            pnex = next_candidates(1:2, :);

            aa = sum(pnex .* pnex, 1);
            bb = sum(pcur .* pcur, 1);
            d = sqrt(abs(aa(ones(size(bb,2),1), :)' + bb(ones(size(aa,2),1), :) - 2 * pnex' * pcur));

            min_dist = min(d, [], 1);
            nearest_obj = next_candidates(:, d == min(d, [], 1));

            if size(nearest_obj, 2) > 1
                nearest_obj = nearest_obj(:, 1);
            end

            candidate_obj = nearest_obj;
            connection_dist = min_dist;

            %% -------------------------
            % Estimate average motion between current and next frame
            % --------------------------
            current_frame_objs = total_cell_objs{k,1}(1:2,:);
            next_frame_objs = total_cell_objs{k+1,1}(1:2,:);

            aa1 = sum(next_frame_objs .* next_frame_objs, 1);
            bb1 = sum(current_frame_objs .* current_frame_objs, 1);
            d1 = sqrt(abs(aa1(ones(size(bb1,2),1), :)' + bb1(ones(size(aa1,2),1), :) - 2 * next_frame_objs' * current_frame_objs));

            moving_pixel = min(d1, [], 1);
            moving_pixel = mean(moving_pixel(1, :));

            rad_mean = 9.5;

            if moving_pixel > rad_mean
                moving_pixel = 0;
            end

            motion_values = [motion_values; moving_pixel]; %#ok<AGROW>

            if moving_pixel > 0.25
                cc = 0.8;
            else
                cc = 0.6;
            end

            if connection_dist > (rad_mean / 1.2) * cc
                candidate_obj = cur_obj(:,1);
                connection_dist = 0;
            end

            %% -------------------------
            % Compute IoU-like overlap using radius
            % --------------------------
            [iou_value, overlap_ok] = compute_circle_box_iou(cur_obj, candidate_obj);

            if overlap_ok && iou_value >= 0.45
                candidate_obj(5,1) = k + 1;
                candidate_obj(6,1) = cur_obj(6,1);
                matched_objs{k + 1,1} = candidate_obj;

            else
                %% -------------------------
                % Search forward within ROI
                % --------------------------
                if connection_dist >= moving_pixel
                    search_radius = rad_mean;
                else
                    search_radius = round(cur_obj(7,:) / 2);
                end

                cur_xy_col = cur_xy';
                x_range = (cur_xy_col(1,:) - search_radius):(cur_xy_col(1,:) + search_radius);
                y_range = (cur_xy_col(2,:) - search_radius):(cur_xy_col(2,:) + search_radius);

                roi_candidates = {};
                existence = [];
                z = 0;

                search_end = min(k + find_next, num_frames);

                for j = k+1:search_end
                    z = z + 1;
                    future_objs = total_cell_objs{j,1};

                    if isempty(future_objs)
                        roi_candidates{z} = []; %#ok<AGROW>
                        existence = [existence, 0]; %#ok<AGROW>
                        continue;
                    end

                    future_xy = future_objs(1:2,:);

                    roi_idx = ...
                        (min(x_range) < future_xy(1,:)) & (future_xy(1,:) < max(x_range)) & ...
                        (min(y_range) < future_xy(2,:)) & (future_xy(2,:) < max(y_range));

                    num_roi = sum(roi_idx);

                    if num_roi > 1
                        pcur_roi = cur_xy_col;
                        pnex_roi = future_objs(1:2, roi_idx == 1);
                        pt_roi = future_objs(:, roi_idx == 1);

                        aa_roi = sum(pnex_roi .* pnex_roi, 1);
                        bb_roi = sum(pcur_roi .* pcur_roi, 1);
                        d_roi = sqrt(abs(aa_roi(ones(size(bb_roi,2),1), :)' + bb_roi(ones(size(aa_roi,2),1), :) - 2 * pnex_roi' * pcur_roi))';

                        nearest_roi = pt_roi(:, d_roi == min(d_roi));
                        if size(nearest_roi, 2) > 1
                            nearest_roi = nearest_roi(:,1);
                        end
                        roi_candidates{z} = nearest_roi;

                    elseif num_roi == 1
                        roi_candidates{z} = future_objs(:, roi_idx);

                    else
                        roi_candidates{z} = [];
                    end

                    existence = [existence, num_roi]; %#ok<AGROW>
                end

                %% -------------------------
                % Decide matching object
                % --------------------------
                if sum(existence) > 0
                    valid_idx = find(existence ~= 0);
                    closest_future_idx = min(valid_idx);

                    if closest_future_idx >= 5
                        matched_objs{k + 1,1} = cur_obj;

                    else
                        add_obj = roi_candidates{closest_future_idx};

                        if isempty(add_obj)
                            matched_objs{k + 1,1} = cur_obj;
                        else
                            pcur2 = cur_obj(1:2,:);
                            pnex2 = add_obj(1:2,:);

                            aa2 = sum(pnex2 .* pnex2, 1);
                            bb2 = sum(pcur2 .* pcur2, 1);
                            d2 = sqrt(abs(aa2(ones(size(bb2,2),1), :)' + bb2(ones(size(aa2,2),1), :) - 2 * pnex2' * pcur2));
                            d2 = min(d2, [], 1);

                            if d2 < search_radius
                                add_obj(5,1) = k + 1;
                                add_obj(6,1) = cur_obj(6,1);
                                matched_objs{k + 1,1} = add_obj;
                            else
                                matched_objs{k + 1,1} = cur_obj;
                            end
                        end
                    end
                else
                    carry_obj = cur_obj;
                    carry_obj(5,1) = k + 1;
                    matched_objs{k + 1,1} = carry_obj;
                end
            end
        end

        matched_tracks{1,q} = matched_objs; %#ok<AGROW>

        if length(matched_tracks{1,q}) == num_frames
            break;
        end

        fprintf('Processed first-frame object %d / %d\n', q, num_cells_first_frame);
    end

    %% =========================
    % Find peaks in motion values
    % ==========================
    motion_values = motion_values(:);

    if isempty(motion_values)
        fast_moving_frame = [];
        threshold_value = 0;

        fig = figure('Visible', 'off');
        plot([]);
        title('Moving pixel');
        saveas(fig, fullfile(save_m_path, sprintf('moving_pixel_%s.jpg', save_name)));
        close(fig);
        return;
    end

    [~, prominence_values] = islocalmax(motion_values);
    threshold_value = max(prominence_values) * base_threshold_ratio;
    peak_idx = find(prominence_values > threshold_value);

    if length(peak_idx) > 20
        adjust_flag = true;
        while adjust_flag
            base_threshold_ratio = base_threshold_ratio + 0.05;
            threshold_value = max(prominence_values) * base_threshold_ratio;
            peak_idx = find(prominence_values > threshold_value);

            if length(peak_idx) < 20
                adjust_flag = false;
            end
        end
    end

    fast_moving_frame = peak_idx;

    %% =========================
    % Save motion plot
    % ==========================
    fig = figure('Visible', 'off');
    plot(1:length(motion_values), motion_values, 'b-'); hold on;
    plot(peak_idx, motion_values(peak_idx), 'r*');
    xlabel('Frame index');
    ylabel('Moving pixel');
    title('Moving pixel peaks');
    axis tight;

    saveas(fig, fullfile(save_m_path, sprintf('moving_pixel_%s.jpg', save_name)));
    close(fig);
end


%% =========================================================
% Local function
%% =========================================================
function [iou_value, is_valid] = compute_circle_box_iou(cur_obj, next_obj)
% Approximates overlap using square boxes derived from radius.

    is_valid = false;
    iou_value = 0;

    if isempty(cur_obj) || isempty(next_obj)
        return;
    end

    cur_xy = cur_obj(1:2,:)';
    cur_r = cur_obj(7,:) / 2;

    next_xy = next_obj(1:2,:)';
    next_r = next_obj(7,:) / 2;

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