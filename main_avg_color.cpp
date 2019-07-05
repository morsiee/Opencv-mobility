/* 
 * File:   main_avg_color.cpp
 * Author: essam
 *
 * Created on August 17, 2015, 11:23 AM
 */

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void readme();

float * rgb2lab(float rgb[3]) {
    // bring input in range [0,1]
    rgb[0] = rgb[0] / 255;
    rgb[1] = rgb[1] / 255;
    rgb[2] = rgb[2] / 255;

    // copy rgb in Mat data structure and check values
    cv::Mat rgb_m(1, 1, CV_32FC3, cv::Scalar(rgb[0], rgb[1], rgb[2]));
    std::cout << "rgb_m = " << std::endl << " " << rgb_m << std::endl;
    cv::Vec3f elem = rgb_m.at<cv::Vec3f>(0,0);
    float R = elem[0];
    float G = elem[1];
    float B = elem[2];
    printf("RGB =\n [%f, %f, %f]\n", R, G, B);

    // create lab data structure and check values
    cv::Mat lab_m(1, 1, CV_32FC3, cv::Scalar(0, 0, 0));
    std::cout << "lab_m = " << std::endl << " " << lab_m << std::endl;

    // convert
    cv::cvtColor(rgb_m, lab_m, CV_RGB2Lab);

    // check lab value after conversion
    std::cout << "lab_m2 = " << std::endl << " " << lab_m << std::endl;
    cv::Vec3f elem2 = lab_m.at<cv::Vec3f>(0,0);
    float l = elem2[0];
    float a = elem2[1];
    float b = elem2[2];
    printf("lab =\n [%f, %f, %f]\n", l, a, b);

    // generate the output and return
    static float lab[] = { l, a, b };
    return lab;
}
/*
 * 
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        readme();
        return -1;
    }

    //    cv::Vec4d d = a-b;
    //    double distance = cv::norm(d);

    Mat rd = imread(argv[1], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    Mat tpl = imread(argv[2], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);

    int rd_cols = rd.cols;
    int rd_rows = rd.rows;

    int tpl_cols = tpl.cols;
    int tpl_rows = tpl.rows;
    // Mean color of the template image ..
    Scalar tpl_scalar = mean(tpl);

    int step = ((tpl_cols / 5)+(tpl_rows / 5)) / 2;
    vector<Rect> matchs;
    //    Size ts = tpl.size.operator()();
    //    cout << ts.width << " : " << ts.height << " --> " << tpl_cols << " : " << tpl_rows << endl;

    for (int x = 0; x < rd_cols - tpl_cols; x = x + step) {
        for (int y = 0; y < rd_rows - tpl_rows; y = y + step) {
            Rect rect = Rect(x, y, tpl_cols, tpl_rows);
            Scalar block_scalar = mean(Mat(rd, rect));
            
            Vec4d d = tpl_scalar - block_scalar;
            double distance = cv::norm(d);
            if (distance < 10) {
                //                cout << distance << endl;
                matchs.push_back(rect);
            }
        }
    }

    for (vector<Rect>::size_type i = 0; i != matchs.size(); i++) {
        Rect rect = matchs[i];
        rectangle(rd, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), Scalar::all(255), CV_FILLED);
    }

    imwrite("/home/essam/work/rd.png", rd);
    return 0;
}

/** @function readme */
void readme() {
    std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl;
}
