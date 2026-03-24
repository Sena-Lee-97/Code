clear; close all; clc;

datasets = {'test_1','test_2','test_3'};

for d = 1:length(datasets)

    dataset = datasets{d};
    save_name = dataset;
    repo_root = fileparts(mfilename('fullpath'));   % 현재 파일 위치
    repo_root = fullfile(repo_root, '..');          % matlab 폴더 → root 이동
    
    img_path = fullfile(repo_root, 'data', dataset);
    txt_path = fullfile(repo_root, 'results', 'detection_result', dataset);
    save_root = fullfile(repo_root, 'results', 'tracking_result', dataset);
    
    save_files_dir          = fullfile(save_root, 'save_files');
    save_track_dir          = fullfile(save_root, 'save_track');
    
    mkdir(save_files_dir);
    mkdir(save_track_dir);
    
    save_m_path   = [save_files_dir filesep];
    save_fig_path = [save_track_dir filesep];
    
    %% =========================
    % Find frame range
    % ==========================
    image_list = dir(fullfile(img_path, '*.jpg'));
    img_nums = zeros(length(image_list), 1);
    
    for k = 1:length(image_list)
        file_name = image_list(k).name;
        numbers = regexp(file_name, '\d+', 'match');
        img_nums(k) = str2double(numbers{1});
    end
    
    first_frame_idx = min(img_nums) + 1;
    last_frame_idx  = max(img_nums) + 1;
    
    %% =========================
    % Performance initialization
    % ==========================
    tracking_times = [];
    tracking_fps = [];
    cpu_usages = [];
    memory_deltas = [];
    
    total_start_time = tic;
    
    initial_memory = memory;
    initial_memory_usage = initial_memory.MemUsedMATLAB;
    initial_cpu = get_cpu_usage();
    
    %% =========================
    % Step 1. Build object file
    % ==========================
    tracking_start_time = tic;
    [total_cell_objs, make_objs_time] = make_objs_file(txt_path, save_m_path, save_name);
    
    %% =========================
    % Step 2. Select first frame / moving pixels
    % ==========================
    [fast_moving_frame, threshold_value] = first_frame( ...
        total_cell_objs, img_path, first_frame_idx, save_m_path, save_name);
    
    %% =========================
    % Step 3. Main tracking
    % ==========================
    s_c = main_tracking( ...
        total_cell_objs, save_name, img_path, save_m_path, ...
        fast_moving_frame, first_frame_idx);
    
    %% =========================
    % Step 4. Confirm tracking results
    % ==========================
    [save_cell_info, num_out_cell, no_out_cell] = confirm_tracking( ...
        img_path, save_m_path, total_cell_objs, s_c, save_name, ...
        last_frame_idx);
    
    current_tracking_time = toc(tracking_start_time);
    tracking_times(end + 1) = current_tracking_time;
    
    fprintf('Tracking time: %.4f sec\n', current_tracking_time);
    
    %% =========================
    % Step 5. Final performance measurement
    % ==========================
    total_tracking_time = toc(total_start_time);
    
    final_memory = memory;
    final_memory_usage = final_memory.MemUsedMATLAB;
    final_cpu = get_cpu_usage();
    
    memory_delta = (final_memory_usage - initial_memory_usage) / 1024 / 1024; % MB
    avg_cpu_usage = mean([initial_cpu, final_cpu]);
    
    tracking_fps(end + 1) = 1 / total_tracking_time;
    cpu_usages(end + 1) = avg_cpu_usage;
    memory_deltas(end + 1) = memory_delta;
    
    fprintf('\n=== Tracking Performance ===\n');
    fprintf('Total Tracking Time: %.4f sec\n', total_tracking_time);
    fprintf('Tracker FPS: %.2f\n', 1 / total_tracking_time);
    fprintf('CPU Utilization: %.1f%%\n', avg_cpu_usage);
    fprintf('Memory Delta: %.2f MB\n', memory_delta);
    
    %% =========================
    % Step 6. Save performance metrics
    % ==========================
    performance_metrics = struct( ...
        'tracking_times', tracking_times, ...
        'tracking_fps', tracking_fps, ...
        'cpu_usages', cpu_usages, ...
        'memory_deltas', memory_deltas, ...
        'make_objs_time', make_objs_time, ...
        'first_frame_idx', first_frame_idx, ...
        'last_frame_idx', last_frame_idx, ...
        'threshold_value', threshold_value, ...
        'num_out_cell', num_out_cell, ...
        'no_out_cell', no_out_cell ...
    );
    
    % save(fullfile(save_m_path, 'tracking_performance.mat'), 'performance_metrics');
    % save(fullfile(save_m_path, 'tracking_times.mat'), 'tracking_times');
    % save(fullfile(save_m_path, 'tracking_fps.mat'), 'tracking_fps');
    % save(fullfile(save_m_path, 'cpu_usages.mat'), 'cpu_usages');
    % save(fullfile(save_m_path, 'memory_deltas.mat'), 'memory_deltas');
    
    %% =========================
    % Step 7. Save performance summary
    % ==========================
    summary_file = fullfile(save_m_path, 'tracking_performance_summary.txt');
    fid = fopen(summary_file, 'w');
    
    fprintf(fid, '=== Tracking Performance ===\n');
    fprintf(fid, 'Dataset: %s\n', dataset);
    fprintf(fid, 'Total Tracking Time: %.4f sec\n', total_tracking_time);
    fprintf(fid, 'Mean Tracking Time: %.4f sec\n', mean(tracking_times));
    fprintf(fid, 'Mean Tracker FPS: %.2f\n', mean(tracking_fps));
    fprintf(fid, 'Mean CPU Utilization: %.1f%%\n', mean(cpu_usages));
    fprintf(fid, 'Mean Memory Delta: %.2f MB\n', mean(memory_deltas));
    fprintf(fid, 'First Frame Index: %d\n', first_frame_idx);
    fprintf(fid, 'Last Frame Index: %d\n', last_frame_idx);
    fprintf(fid, 'Threshold Value: %.4f\n', threshold_value);
    fprintf(fid, 'Out-of-frame Cells: %d\n', num_out_cell);
    fprintf(fid, 'Remaining Cells: %d\n', numel(no_out_cell));
    fprintf(fid, 'Final Saved Cells: %d\n', numel(save_cell_info));
    fclose(fid);
    
    %% =========================
    % Step 8. Save tracking coordinates
    % ==========================
    finish = save_cell_xy(save_m_path, dataset, save_name, save_cell_info);
    
    fprintf('\nFinal elapsed time: %.4f sec\n', toc(tracking_start_time));
    


end


%% =========================
% Local function: CPU usage
% ==========================
function cpu_usage = get_cpu_usage()
    if ispc
        [~, result] = system('wmic cpu get loadpercentage');
        values = regexp(result, '\d+', 'match');
        if isempty(values)
            cpu_usage = 0;
        else
            cpu_usage = str2double(values{1});
        end
    elseif ismac
        [~, result] = system('ps -A -o %cpu | awk ''{sum += $1} END {print sum}''');
        cpu_usage = str2double(strtrim(result));
    elseif isunix
        [~, result] = system('top -bn1 | grep "Cpu(s)" | awk ''{print $2 + $4}''');
        cpu_usage = str2double(strtrim(result));
    else
        cpu_usage = 0;
    end
end