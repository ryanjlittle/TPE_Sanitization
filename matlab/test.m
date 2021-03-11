clc, clear, clf

tests = 12;
% 10 random indices between 0 and 23707 (there are 23708 images in the
% database).
indices = randi([0, 23707], 1, tests);

files = strings(1, tests);
images = zeros(200,200,3,tests);
for i=1:tests
    files(i) = ['./images/img_', int2str(indices(i)), '.jpg'];
    images(:,:,:,i) = imread(files(i));
end

results = zeros(tests);
for i=1:tests
    for j=1:tests
        results(i,j) = perturb(images(:,:,:,i), images(:,:,:,j), 10);
        %imshow(results(i,j))
    end
end

ximages = imtile(files, 'GridSize', [1, tests]);
yimages = imtile(files, 'GridSize', [tests, 1]);

a = NaN(tests+1);
a(2:end,2:end) = results;
results = a;

h = heatmap(results);
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
imwrite(ximages, 'ximages.jpg');

