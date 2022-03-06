img = imread("../../inputs/0002.jpg");
imshow(img);
gray = rgb2gray(img);
imshow(gray)

surf = zeros(500,1);

for i=2:500
    tic
    points = detectSURFFeatures(gray);
    surf(i) = toc + surf(i-1);
end

sift=ones(500,1);
for i=2:500
    tic
    points = detectSIFTFeatures(gray);
    sift(i) = toc + sift(i-1);
end


fprintf(surf)
fprintf(sift)