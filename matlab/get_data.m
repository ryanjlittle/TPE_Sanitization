clc, clf, clear

%we want to convert source to target
target = imread('./target.jpg');
source = imread('./source.jpg');

metadata_size = [];


block_widths = [2, 4, 8, 16, 32, 64, 128, 256, 512];
for block_width = block_widths
    [metadata, ~] = perturb(target, source, block_width);    
    metadata_size = [metadata_size, metadata];
end

metadata_size = 100* metadata_size / (512^2);
bar(categorical(block_widths), metadata_size);
xlabel('Block Width')
ylabel('Metadata Size')
ytickformat('percentage')
ax = gca;
ax.YAxis.Exponent=0;