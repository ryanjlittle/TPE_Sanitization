clc, clf, clear

%we want to convert source to target
im_1 = imread('./target.jpg');
im_2 = imread('./source.jpg');

sanitize = [];
desanitize = [];

block_widths = [2, 4, 8, 16, 32, 64, 128, 256, 512];
for block_width = block_widths
    
    tic
        im_3 = perturb(im_1, im_2, block_width);
    toc;
    sanitize = [sanitize; toc];
    
    % print to confirm thumbnails match
    has_same_thumbnail(im_3(1:512,:,:), im_1, block_width)
    
    
    tic
        restored_img = reverse_perturbation(im_3);
    toc;
    desanitize = [desanitize; toc];
    all(restored_img ==im_2, 'all')
end
bar(categorical(block_widths), [sanitize, desanitize]);
xlabel('Block Width')
ylabel('Seconds')
legend({'Sanitization', 'Restoration'}, 'location', 'northeast')