/* 
 * File:   main_houghline.cpp
 * Author: essam
 *
 * Created on July 20, 2015, 11:13 PM
 */
#include "opencv2/opencv.hpp"
#include <cstdlib>
#include <math.h>

using namespace std;
using namespace cv;

/*
 * 
 */
int main(int argc, char** argv) {
    Mat src, dst, color_dst;
    if (argc != 2 || !(src = imread(argv[1], 0)).data)
        return -1;

    Canny(src, dst, 50, 200, 3);
    cvtColor(dst, color_dst, CV_GRAY2BGR);

#if 0
    vector<Vec2f> lines;
    HoughLines(dst, lines, 1, CV_PI / 180, 100);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        Point pt1(cvRound(x0 + 1000 * (-b)),
                cvRound(y0 + 1000 * (a)));
        Point pt2(cvRound(x0 - 1000 * (-b)),
                cvRound(y0 - 1000 * (a)));
        line(color_dst, pt1, pt2, Scalar(0, 0, 255), 3, 8);
    }
#else
    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10);
    for (size_t i = 0; i < lines.size(); i++) {
        line(color_dst, Point(lines[i][0], lines[i][1]),
                Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
    }
#endif
//    namedWindow("Source", 1);
//    imshow("Source", src);
//
//    namedWindow("Detected Lines", 1);
//    imshow("Detected Lines", color_dst);
    string path = "/Users/essam/w_z_6/all/";
    int thr = 50;
    string road = "93713584";
    string ext = ".bmp";
    string output = path + "/" + road +"-houghline" + ext;
    imwrite(output, color_dst);
//    waitKey(0);
    return 0;
}

