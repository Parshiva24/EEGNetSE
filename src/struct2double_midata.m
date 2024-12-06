
clc; clear; close all;

subNames = ["Asish", "Danish", "Gayathri", "Navaneeth", "Nithyasree", "Rakesh", "Shanmu", "Ajul", "Ananthu",... 
    "Aswathy", "Athira", "Bharath2","Durga", "Greeshma", "Jijomon", "Kumudini", "Mithul", "Pramod1", "Sagila", "Venkatesh"];
[parent_dir, ~, ~] = fileparts(pwd);
for s = 1: length(subNames)
    basepath = "E:\Ph.D. Backup\2021PhD\Improvement in direction decoding using ErrP\Dataset_MAT\";
    calib_file = fullfile(basepath, sprintf("%s_Calibration.mat", subNames{s}));
    online_file = fullfile(basepath, sprintf("%s_Online.mat", subNames{s}));

    [Xtrain, Ytrain, Ytrain_fb] = load_eeg_data(calib_file);
    [Xtest, Ytest, Ytest_fb] = load_eeg_data(online_file);
    if strcmp(subNames(s), "Gayathri")
        Xtest = NaN;
        Ytest = NaN;
        Ytest_fb = NaN;
    end

    outfile = fullfile(parent_dir, "data", sprintf("S%.2d_mitrials.mat", s));
    fprintf('File Saved as: %s\n', outfile);
  
    % outfile = fullfile(parent_dir, "data", sprintf("%s_midata.mat", subNames{s}));
    save(outfile, 'Xtrain', 'Ytrain', 'Ytrain_fb', 'Xtest', 'Ytest', 'Ytest_fb')
    
    % if isnan(Xtrain)
    %     continue
    % end
    % 
    % if isnan(Xtest)
    %     continue
    % end
    % 
end




function [X, Y, Yfb] = load_eeg_data(filepath)

try
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
    
    Yfb = zeros([size(X, 1), 1]);
    for i=1:length(EEG_FB.epoch)
        if EEG_FB.epoch(i).eventcode==4
            Yfb(i) = 0;
        else
            Yfb(i)=1;
        end
    end

catch
    warning('Problem using function.  Assigning a value of 0.');
    X = NaN;
    Y = NaN;
    Yfb = NaN;
end

end




% -----------------------------------%
