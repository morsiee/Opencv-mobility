/* 
 * File:   preprocessing_moment.cpp
 * Author: essam
 *
 * Created on August 28, 2015, 9:34 AM
 */

#include <cstdlib>
#include"Utils.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

void imclose(Mat &src, int dilation_elem, int dilation_size) {
    int dilation_type;
    if (dilation_elem == 0) {
        dilation_type = MORPH_RECT;
    } else if (dilation_elem == 1) {
        dilation_type = MORPH_CROSS;
    } else if (dilation_elem == 2) {
        dilation_type = MORPH_ELLIPSE;
    }

    Mat element = getStructuringElement(dilation_type,
            Size(2 * dilation_size + 1, 2 * dilation_size + 1),
            Point2d(dilation_size, dilation_size));


    morphologyEx(src, src, MORPH_CLOSE, element);
    //    imshow("Dilation Demo", dilation_dst);
}

void iterative_imclose(Mat& im, int structure, int size) {
    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;
    do {
        //        cout << "imclose .." << endl;
        imclose(im, structure, size);
        imclose(im, structure, size);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    } while (countNonZero(diff) > 0);
}

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
    //    int NUMCORES = 4;
    //    int gray_th = 64;
    Mat src_gray;
    int thresh = 20;
    int max_thresh = 255;
    RNG rng(12345);

    cvtColor(src, src_gray, CV_RGB2GRAY);

    //    Utils utils;
    //    utils.imblur(src_gray, src_gray);
    GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
    for (int i = 0; i < 9; i++)
        blur(src_gray, src_gray, Size(3, 3));
    //    blur(src_gray, src_gray, Size(3, 3));

    Mat edges;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using canny
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    int kernel_size = 3;
#if 1
    /// Apply Laplace function
    Mat dst;
    Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(dst, edges);
    //    Canny(src_gray, edges, thresh, thresh * 2, 3);
#else

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    /// Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);

    //    Canny(edges, edges, thresh, thresh * 2, 3);

    
#endif
    string edges_out = path + "/edges" + ext;
    imwrite(edges_out, edges);
    //    iterative_imclose(canny_output, 0, 15);

    /// Find contours
    //    findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
#if 0
    /// Get the moments
    vector<Moments> mu(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        mu[i] = moments(contours[i], false);
    }

    ///  Get the mass centers:
    vector<Point2f> mc(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        mc[i] = Point2f(static_cast<float> (mu[i].m10 / mu[i].m00), static_cast<float> (mu[i].m01 / mu[i].m00));
    }
#endif
    /// Draw contours
    Mat drawing = Mat::zeros(edges.size(), CV_8UC3);
    //    for (size_t i = 0; i < contours.size(); i++) {
    //        Scalar color = Scalar::all(255);
    //        drawContours(drawing, contours, (int) i, color, 2, 8, hierarchy, 0, Point());
    //        circle(drawing, mc[i], 4, color, -1, 8, 0);
    //    }

    /// Show in a window
    //    namedWindow("Contours", WINDOW_AUTOSIZE);
    //    imshow("Contours", drawing);

    /// Calculate the area with the moments 00 and compare with the result of the OpenCV function
    printf("\t Info: Area and Contour Length \n");
    vector<Point> approx;
    for (size_t i = 0; i < contours.size(); i++) {

        //        if (mu[i].m00 < 20 && arcLength(contours[i], true)<100) continue;
        //        printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", (int) i, mu[i].m00, contourArea(contours[i]), arcLength(contours[i], true));
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
        vector<vector<Point> > contours0;
        contours0.push_back(approx);
        drawContours(drawing, contours0, -1, color, 2, 8, hierarchy, 0, Point());
        //        circle(drawing, mc[i], 4, color, -1, 8, 0);
    }

    string out = path + "/moment" + ext;
    imwrite(out, drawing);
    return 0;
}

