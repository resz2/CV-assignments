img = imread("");
imshow(img);
gray = rgb2gray(img);

surf = zeros(500,1);

for i=2:500
    tic
    pnt = detectSURFFeatures(gray);
    surf(i) = toc + surf(i-1);
end

sift=ones(500,1);
for i=2:500
    tic
    pnt = detectSIFTFeatures(gray);
    sift(i) = toc + sift(i-1);
end
tic 

figure;
plot(surf);
hold on;
plot(sift);
hold off;