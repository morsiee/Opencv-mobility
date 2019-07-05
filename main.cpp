/* 
 * File:   main.cpp
 * Author: essam
 *
 * Created on July 12, 2015, 12:05 PM
 * g++ main.cpp -o main.out `pkg-config --cflags --libs opencv`
 */

#include <cstdlib>
#include<iostream>
//#include <cv.h>
#include <opencv/highgui.h>
#include<opencv/cv.h>
#include <opencv/cv.hpp>

//#include<opencv/cv.hpp>

using namespace cv;
using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }
    Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file
    //    Mat dst = imread("dest.jpg", 1);
    Size size = src.size.operator()();
    //    int new_w = size.width;
    //    int new_h = size.height;
    //    if (src.cols > dst.cols)
    //        new_w = dst.cols;
    //    else
    //        new_w = src.cols;
    //
    //    if (src.rows > dst.rows)
    //        new_h = dst.rows;
    //    else
    //        new_h = src.rows;

    //    Rect rectROI(0, 0, new_w, new_h);
    Mat mask = Mat::zeros(size, src.type());

    Point P1(389456.4275170417, 3951009.2903417293);
    Point P2(389456.6274921351, 3951009.2798435967);
    Point P3(389452.63250786485, 3950929.3801564015);
    Point P4(389452.4324829583, 3950929.3896582713);
    vector< vector<Point> > co_ordinates;
    vector<Point> vertices;
    //    Point2d p(388632.7862481127,3950313.751284572);
    vertices.push_back(P1);
    vertices.push_back(P2);
    vertices.push_back(P3);
    vertices.push_back(P4);
    co_ordinates.push_back(vertices);
    drawContours(mask, co_ordinates, 0, Scalar(255), CV_FILLED, 8);
//    namedWindow("Mask", WINDOW_AUTOSIZE);
//    imshow("Mask", mask);
    //    imwrite()
    //    Mat srcROI = src(rectROI);
    //    Mat dstROI = dst(rectROI);
    Mat masked_image;
    bitwise_and(src, mask, masked_image);
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", masked_image);
    //    Mat dst1;
    //    Mat dst2;
    //
    //    srcROI.copyTo(dst1, mask);
    //    imwrite("dst1.jpg", dst1);
    //
    //    bitwise_not(mask, mask);
    //    dstROI.copyTo(dst2, mask);
    //
    //    dstROI.setTo(0);
    //    dstROI = dst1 + dst2;
    //    imshow("final result", dst);
    return 0;
}

