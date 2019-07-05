/* 
 * File:   main_plot.cpp
 * Author: essam
 *
 * Created on August 13, 2015, 9:22 AM
 */

#include <cstdlib>
#include<iostream>
#include<random>
#include <stdio.h>
#include <fstream>
#include<cmath>
#include<random>
//#include <cv.h>
//#include <opencv/highgui.h>
//#include<opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgproc/types_c.h>
#include <opencv/cv.h>
//#include<opencv.hpp>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <ctime>
#include<string>
#include <omp.h>
#include <opencvblobslib/blob.h>
#include <opencvblobslib/BlobResult.h>
//#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <vector>

using namespace std;
using namespace cv;

/*
 * 
 */
int main(int argc, char** argv) {
    int w = 13056, h = 15104;
    Mat img(w, h, CV_8UC3, Scalar::all(0));

    Point2d PL(7827.12, 12227.7);
    Point2d PL1(7776.02, 12247.5);
    line(img, PL, PL1, Scalar::all(255), 3, 8, 0);
    //print free shape ..
    Point2d P0(8839, 14839);
    Point2d P1(8788, 14858);
    Point2d P2(6764, 9637);
    Point2d P3(6816, 9617);

    line(img, P0, P1, Scalar(0, 0, 255), 4, 8, 0);
    line(img, P2, P1, Scalar(0, 0, 255), 4, 8, 0);
    line(img, P2, P3, Scalar(0, 0, 255), 4, 8, 0);
    line(img, P0, P3, Scalar(0, 0, 255), 4, 8, 0);
    
    imwrite("res.png", img);

    return 0;
}

