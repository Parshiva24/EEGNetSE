% Converting the structured dataset into double for reading in Python
clc; clear; close all;
[parent_dir, ~, ~] = fileparts(pwd);
A = dir(fullfile(parent_dir, "data", "*_Calibration.mat"));
for sub = 1: length(A)
    filepath = fullfile(A(sub).folder, A(sub).name);
    load(filepath)
    X = permute(double(EEG_MI.data), [3, 2, 1]);
    Y = zeros([size(X, 1),1]);

    for i=1:length(EEG_MI.epoch)
        if EEG_MI.epoch(i).eventcode==2
            Y(i) = 0;
        else
            Y(i)=1;
        end
    end
    fs = EEG_MI.srate;
    outfile = fullfile(parent_dir, "data", sprintf("%s_midata.mat", A(sub).name(1:end-16)));
    disp(outfile)

    save(outfile, 'X', 'Y', 'fs')
end
