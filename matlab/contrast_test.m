clc, clear, clf

num_sources = 100;
% 10 random indices between 0 and 23707 (there are 23708 images in the
% database).
indices = randi([0, 23707], 1, num_sources);

num_targets = 100;

%target_indices = [17319, 23665, 20385, 2634, 16114, 22753, 4111, 8924, 15583];
random_indices = randi([0, 23707], 1, num_targets);
target_indices = zeros(1, num_targets);

source_files = strings(1, num_sources);
target_files = strings(1, num_targets);
random_files = strings(1, num_targets);

sources = zeros(200,200,3,num_sources);
targets = zeros(200,200,3,num_sources);
randoms = zeros(200,200,3,num_sources);

for i=1:num_sources
    source_files(i) = ['./images/img_', int2str(indices(i)), '.jpg'];
    sources(:,:,:,i) = imread(source_files(i));
end


for i=1:num_targets
    random_files(i) = ['./images/img_', int2str(random_indices(i)), '.jpg'];
    randoms(:,:,:,i) = imread(random_files(i));
end


i = 1;
while i <= num_targets
    idx = randi([0, 23707], 1, 1);
    file = ['./images/img_', int2str(idx), '.jpg'];
    image = imread(file);
    if all(image > 15, 'all') && all(image < 240, 'all')
        target_indices(i) = idx;
        target_files(i) = file;
        targets(:,:,:,i) = image;
        i = i+1;
    end
end



metadata = zeros(num_targets*num_sources, 2);
idx = 1;
for i=1:num_targets
    for j=1:num_sources
        metadata(idx, 1) = perturb(targets(:,:,:,i), sources(:,:,:,j), 10);
        metadata(idx, 2) = perturb(randoms(:,:,:,i), sources(:,:,:,j), 10);
        idx = idx+1;
    end
end



[~, idx] = sort(sum(metadata, 2));
metadata = metadata(idx, :);
target_files = target_files(:, idx);

ximages = imtile(source_files, 'GridSize', [1, num_sources]);
yimages = imtile(target_files, 'GridSize', [num_targets, 1]);

a = NaN(num_targets+1, num_sources+1);
a(1:end-1,2:end) = metadata;
metadata = a;

h = heatmap(metadata);
h.XDisplayLabels = nan(size(h.XDisplayData));
h.YDisplayLabels = nan(size(h.YDisplayData));
h.XLabel = 'Source Image';
h.YLabel = 'Target Image';
h.Title = 'Number of Metadata Pixels'; 


axes('Position',[0.2 0.64 0.67 0.5]);
imshow(ximages);
axes('Position', [0 0.11 0.33 0.74])
imshow(yimages);

imwrite(ximages, 'ximages.jpg');
imwrite(yimages, 'yimages.jpg');
;
