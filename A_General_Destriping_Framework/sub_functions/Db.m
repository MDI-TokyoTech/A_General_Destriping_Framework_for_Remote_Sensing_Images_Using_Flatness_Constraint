function result = Db(z)
    n3 = size(z, 3);
    result = z(:, :, [2:n3, n3]) - z;
end