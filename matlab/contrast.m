function result = contrast(image)
    img_max = max(max(image));
    img_min = min(min(image));
    result = reshape((img_max - img_min) ./ (img_min + img_max), [1 3]);
end