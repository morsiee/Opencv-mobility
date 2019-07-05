/* 
 * File:   main_scale.cpp
 * Author: essam
 *
 * Created on August 9, 2015, 10:28 PM
 */

#include <cstdlib>
#include<opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include<iostream>
using namespace std;
using namespace cv;

/**
 * 
 * @param width
 * @param x
 * @param xmin
 * @param xmax
 * @return 
 */
double scaleX(double width, double x, double xmin, double xmax) {
    return width * (x - xmin) / (xmax - xmin);
}

/**
 * 
 * @param height
 * @param y
 * @param ymin
 * @param ymax
 * @return 
 */
double scaleY(double height, double y, double ymin, double ymax) {
    return height * (ymax - y) / (ymax - ymin);
}

/*
 * 
 */
int main(int argc, char** argv) {
    //Map parameters ..
//    x:387968.529665	y:3949750.693277
//    x:389547.318664	y:3951528.063522
    const double XMIN = 387968.529665;
    const double YMIN = 3949750.693277;
    const double XMAX = 389547.318664;
    const double YMAX = 3951528.063522;

    double mw = XMAX - XMIN;
    double mh = YMAX - YMIN;
    int w= 4;
    
//    Mat src;
//    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);

        Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    Size size = src.size.operator()();
    cout << "Map parameters:\t" << mw << " : " << mh << endl;
    cout << "Image parameters:\t" << size.width << " : " << size.height << endl;

    cout << "Aspect ratio:\t" << (size.width* size.height)/ (mw*mh) << endl;
    
    return 0;
}

