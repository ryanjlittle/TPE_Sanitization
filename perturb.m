function perturbed = perturb(target, source, block_width)

img_height = size(target, 1);
img_width = size(target, 2);

% The number of bytes we need to store the remainder is determined by the block_width.
if block_width <= 16
    remain_len = 1;
elseif block_width <= 256
    remain_len = 2;
else
    remain_len = 3;
end

% Ensure our offset is coprime to the block size
offset = (block_width+1) * floor(block_width / 3);
while gcd(offset, block_width^2) ~= 1
    offset = offset + 1;
end

new_img = zeros(size(target), 'uint8');
metadata = uint8.empty(0, 3);

for i = 1:block_width:img_height
    bottom = min(img_height, i+block_width-1);
    for j = 1:block_width:img_width
        right = min(img_width, j+block_width-1);
        target_block = target(i:bottom, j:right, :);
        source_block = source(i:bottom, j:right, :);
        
        block_offset = offset; 
        height = size(source_block, 1);
        width = size(source_block, 2);
        
        % If the block is on the right or bottom edge of the image and
        % isn't the full size, we need to compute a new offset.
        if height*width ~= block_width^2
            block_offset = (height+1) * floor(width / 3);
            while gcd(block_offset, height*width) ~= 1
                block_offset = block_offset + 1;
            end
        end
        [new_block, block_metadata] = perturb_block(target_block, source_block, remain_len, block_offset);
        
        new_img(i:bottom, j:right, :) = new_block;
        
        metadata = [metadata; block_metadata];
        
    end
end

padding_len = mod(-size(metadata, 1)-7, img_width);
metadata(end+1:end+padding_len+7, :) = zeros(padding_len+7, 3);


% Serialize block block_width and padding lenght into two bytes
block_width_16 = uint16(block_width);
padding_16 = uint16(padding_len);
img_height_16 = uint16(img_height);
padding_bytes = typecast(padding_16, 'uint8')';
block_width_bytes = typecast(block_width_16, 'uint8')';
img_height_bytes = typecast(img_height_16, 'uint8')';
remain_len_byte = uint8(remain_len);

% Write "header" info (actually stored at the end) to metadata
metadata(end-6:end-5, :) = [padding_bytes, padding_bytes, padding_bytes];
metadata(end-4, :) = [remain_len_byte, remain_len_byte, remain_len_byte];
metadata(end-3:end-2, :) = [block_width_bytes, block_width_bytes, block_width_bytes];
metadata(end-1:end, :) = [img_height_bytes, img_height_bytes, img_height_bytes];


metadata = reshape(metadata, img_width, [], 3);
metadata = permute(metadata, [2 1 3]);

perturbed = [new_img; metadata];

