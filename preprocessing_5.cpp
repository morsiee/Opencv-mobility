
/* 
 * Real-time traffic snapshot estimation using single pass satellite imagery and disparity maps.
 * 
 * Author: essam
 *
 * Created on July 17, 2015, 4:31 PM
 * g++ -O2 -std=c++11 -fopenmp Utils.cpp preprocessing_5.cpp -o preprocessing_5.o `pkg-config --libs --cflags opencv` -lopencvblobslib -lboost_filesystem -lboost_system
 */
#include "Utils.h"
#include <cstdlib>
#include<iostream>
#include<random>
#include <stdio.h>
#include <fstream>
#include<cmath>
#include<random>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/photo/photo.hpp>
#include <opencvblobslib/blob.h>
#include <opencvblobslib/BlobResult.h>
#include <sstream>
#include <vector>
#include <ctime>
#include<string>
#include <omp.h>
#include <vector>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>

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
    int NUMCORES = 4;
    int gray_th = 64;
    Size size = src.size.operator()();
    Utils utils;
    //    detailEnhance(Mat src, Mat dst, float sigma_s=10, float sigma_r=0.15f);

    Mat imask(size.width, size.height, src.type());

#if 1
    utils.imenhance(src, src);
    string enh_output = path + "/w_z_6" + ext;
    imwrite(enh_output, src);
#endif

#if 1
    //    blur(src, imask, Size(5, 5));
    utils.imblur(src, imask);
#else
    imask = src;
#endif
    // reduce the intensity of the of the figure to reduce the effects of the shadow.
    imask.convertTo(imask, -1, 1, -gray_th);
    //scanImageAndInvertIterator(imask);

    cvtColor(imask, imask, CV_RGB2GRAY);
    Mat edges;
    utils.CannyThreshold(imask, edges, 50, 3);

    string gray_output = path + "/gray" + ext;
    imwrite(gray_output, imask);

    string edges_output = path + "/edges" + ext;
    imwrite(edges_output, edges);

    threshold(imask, imask, gray_th, 224, THRESH_BINARY);

    //    iterative_imopen(imask, 0, 5);
    utils.iterative_imclose(imask, 1, 3);

    //    iterative_imclose(imask, 0, 4);
    string binary_output = path + "/binary" + ext;
    imwrite(binary_output, imask);
    cout << "finish writing binary image .." << endl;
    //    adaptiveThreshold(imask, imask,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);

#if 1
    bool blob = false;
    utils.fillBlobs(edges, edges, blob, NUMCORES);
    //    iterative_imopen(edges, 0, 5);
    utils.iterative_imclose(edges, 1, 3);

    string blob_output = path + "/blob" + ext;
    imwrite(blob_output, edges);

    Mat res;
    bitwise_and(edges, imask, res);

    //changed from 11 to 21
    utils.iterative_imclose(res, 1, 3);
    utils.iterative_imopen(res, 1, 15);

    string and_output = path + "/bitwise_and" + ext;
    imwrite(and_output, res);

    blob = true;
    utils.fillBlobs(res, res, blob, NUMCORES);
    utils.iterative_imopen(res, 1, 15);

    string blob_and_output = path + "/marker" + ext;
    imwrite(blob_and_output, res);



    Mat dist_8u;
    res.convertTo(dist_8u, CV_8U);

    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(res.size(), CV_32SC1);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int> (i), Scalar::all(static_cast<int> (i) + 1), -1);
    // Draw the background marker
    circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);

//    imshow("Markers", markers * 10000);

    string output_path = path + "/Markers contours" + ext;
    imwrite(output_path, markers * 10000);

 // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                  // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i,j) = colors[index-1];
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    // Visualize the final image
//    imshow("Final Result", dst);
    
    output_path = path + "/Final Result" + ext;
    imwrite(output_path, dst);
    
#endif
    return 0;
}

