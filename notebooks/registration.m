path_alignment = cell2mat(importdata('./temp/folder_aligned.txt'));
number_labels = importdata('./temp/labels_number.txt');
number_labels_lab = importdata('./temp/labels_number_lab.txt');
number_labels_lab_GM_WM = importdata('./temp/labels_number_lab_GM_WM.txt');

path_ortho = strcat(path_alignment, "\polarimetry.png");
path_ortho_650 = strcat(path_alignment, "\polarimetry_650.png");
path_depol = strcat(path_alignment, "\depolarization.png");

ortho_650 = imread(path_ortho_650);
depol = imread(path_depol);

path_histology = strcat(path_alignment, "\histology_rgb_upscaled.png");
histology = imread(path_histology);
path_labels = strcat(path_alignment, "\histology_labels_upscaled.png");
labels = imread(path_labels);
path_labels_GM_WM = strcat(path_alignment, "\histology_labels_GM_WM_upscaled.png");
labels_GM_WM = imread(path_labels_GM_WM);

alpha = 0.8;
% img_polarimetry = alpha * ortho_650 + (1 - alpha) * depol;
img_polarimetry = ortho_650;
% img_polarimetry = imfuse(ortho_650,depol);

% if number_labels_lab_GM_WM >= 2
%    img_display = imfuse(histology,labels_GM_WM);
% elseif number_labels_lab >= 2
%     img_display = imfuse(histology,labels);
% else
%     img_display = histology;
% end
img_display = histology;

if isfile("./temp/mp_fp.mat")
    S = load("./temp/mp_fp.mat");
    mp = S.mp;
    fp = S.fp;
elseif number_labels == 1
     mp = [[0,0],[0,100],[300,200],[200,300],[500,400],[400,500]];
     fp = [[0,0],[0,100],[300,200],[200,300],[500,400],[400,500]];
else
    [mp,fp] = cpselect(img_display,img_polarimetry,Wait=true);
end

save(strcat(path_alignment, "\mp_fp.mat"),'mp','fp');