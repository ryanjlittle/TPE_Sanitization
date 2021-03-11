clc, clear, clf


target = imread('./images/img_2634.jpg');


tests = 100;
% 10 random indices between 0 and 23707 (there are 23708 images in the
% database).
indices = randi([0, 23707], 1, tests);

files = strings(1, tests);
images = zeros(200,200,3,tests);
for i=1:tests
    files(i) = ['./images/img_', int2str(indices(i)), '.jpg'];
    images(:,:,:,i) = imread(files(i));
end

results = zeros(20);
for i=1:20
    for j=1:20
        target_block = target((i-1)*10+1:i*10, (j-1)*10+1:j*10, :);
        for k=1:tests
            source_block = images((i-1)*10+1:i*10, (j-1)*10+1:j*10, :, k);
            results(i,j) = results(i,j) + perturb(target_block, source_block, 10);
        end
    end
end

ximages = imtile(files, 'GridSize', [1, tests]);
yimages = imtile(files, 'GridSize', [tests, 1]);


h = heatmap(results);
h.XDisplayLabels = nan(size(h.XDisplayData));
h.YDisplayLabels = nan(size(h.YDisplayData));
h.XLabel = 'Source Image';
h.YLabel = 'Target Image';
h.Title = 'Number of Metadata Pixels';


%axes('Position',[0.2 0.64 0.67 0.5]);
%imshow(ximages);
%axes('Position', [0 0.11 0.33 0.74])
%imshow(yimages);

imwrite(ximages, 'ximages.jpg');
imwrite(ximages, 'ximages.jpg');

