
/* 
 * Real-time traffic snapshot estimation using single pass satellite imagery and disparity maps.
 * 
 * Author: essam
 *
 * Created on July 17, 2015, 4:31 PM
 * g++ -O2 -std=c++11 preprocessing_markers.cpp -o markers.o `pkg-config --libs --cflags opencv`
 */
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        cout << " Usage: [Road image] [Results path]." << endl;
        return -1;
    }

    Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    string path = argv[2];
    string ext = ".png";
    int gray_th = 64;
//    Size size = src.size.operator()();

    // Create a kernel that we will use for accuting/sharpening our image
    Mat kernel = (Mat_<float>(3, 3) <<
            1, 1, 1,
            1, -8, 1,
            1, 1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    Mat sharp = src; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    string out = path+"/New Sharped Image"+ext;
    imwrite(out, imgResult);
    
//    imshow("New Sharped Image", imgResult);
    src = imgResult; // copy back
    // Create binary image from source image
    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, gray_th, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    out = path+"/Binary Image"+ext;
    imwrite(out, bw);
//    imshow("Binary Image", bw);
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    out = path+"/Distance Transform Image"+ext;
    imwrite(out, dist);
    
//    imshow("Distance Transform Image", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);
    out = path+"/Peaks"+ext;
    imwrite(out, dist);
    
//    imshow("Peaks", dist);
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int> (i), Scalar::all(static_cast<int> (i) + 1), -1);
    // Draw the background marker
    circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);
    
    out = path+"/Markers"+ext;
    imwrite(out, markers * 10000);
    
//    imshow("Markers", markers * 10000);
    // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int> (contours.size()))
                dst.at<Vec3b>(i, j) = colors[index - 1];
            else
                dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }
    // Visualize the final image
    out = path+"/Final Result"+ext;
    imwrite(out, dst);
    
//    imshow("Final Result", dst);
//    waitKey(0);
    return 0;
}

