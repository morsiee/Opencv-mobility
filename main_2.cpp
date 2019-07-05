/* 
 * File:   main_2.cpp
 * Author: essam
 *
 * Created on July 13, 2015, 4:31 PM
 */

#include <cstdlib>
#include<iostream>
//#include <cv.h>
#include <opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <ctime>

using namespace cv;
using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    //    using boost::lexical_cast;
    //    using boost::bad_lexical_cast;

    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    string path = "/home/essam";
    string road = "93713584";
    string ext = ".bmp";
    int p_count = 36;
    string shape = "11196.289468889738,11067.480376135021 11287.88157571899,11393.738751539711 11283.807388683863,11356.24961209326 11293.174138566512,11417.787944505748 11284.295983938522,11362.852825388467 11355.416095731222,11576.824061508403 11381.163282836113,11655.088279786136 11480.45245939151,11960.634930329907 11497.005870278308,12009.49241955899 11669.996669042688,12501.117230491287 11687.851253447521,12552.306956286335 11920.76480306214,13226.132604973638 11946.794830656438,13305.927375884954 12161.155038029014,14002.308376691779 12176.65334025356,14050.307088488484 12354.865160614834,14575.351287545544 12371.985806215094,14627.645261603304 12666.384652675626,15600.733733750947 12351.565467663973,15100.063798448924 12057.166697350076,14731.975301066437 12041.374217743669,14683.82352922137 11864.322756966903,14162.26044696751 11847.238719166906,14109.473494947159 11631.412802595767,13408.451919113935 11607.163483142467,13334.26387143999 11376.150820661509,12666.093564745433 11358.611391027938,12615.817790867262 11185.931678817398,12125.079807722599 11168.083431193865,12072.471797290336 11067.41296767223,11762.767009321531 11042.053844783364,11685.688737321467 10971.267115232144,11472.723063623027 10962.388960603672,11417.787944505748 10961.170584792242,11431.227890990123 10957.096397756151,11393.738751539711 10880.649634181667,11169.160134614986";

    Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    Size size = src.size.operator()();
    Mat mask = Mat::zeros(size, src.type());

    cout << "Size width: " << size.width << " height: " << size.height << endl;
    //    namedWindow("Shape", WINDOW_AUTOSIZE);
    //    imshow("Shape", mask);

    //    389317.56231068086, 3951312.9142964506
    //    389317.8343902457, 3951327.2815348804
    //    389319.22430312255, 3951360.1815385614
    //    389239.2956968773, 3951363.5584614393
    //    389237.90560975415, 3951330.65846512
    //    389237.57768931903,3951314.48570355

    //    Point P1(2965.378721460947, 10937.690222613504);
    //    Point P2(3007.6274922072007, 10645.107556201034);
    //    Point P3(3110.749862538762, 9984.392110664518);
    //    Point P4(2457.527971350934, 9876.732003391333);
    //    Point P5(2354.405783221202, 10537.447478958777);
    //    Point P6(2310.981192874715, 10837.85162718213);

    cv::Rect irect(cv::Point(), size);


    //    Point P1(50.5, 150);
    //    Point P2(150, 150.6);
    //    Point P3(250, 50);
    //    Point P4(350.2, 150);
    //    Point P5(250, 250);
    //    Point P6(50, 300);

    vector< vector<Point> > co_ordinates;
    vector<Point> vertices;
    //    Point2d p(388632.7862481127,3950313.751284572);
    clock_t begin = clock();
    int i = 0;
    stringstream ssin(shape);
    while (ssin.good() && i < p_count) {
        string p;
        ssin >> p;
        boost::replace_all(p, ",", " ");
        stringstream psin(p);
        int j = 0;
        string xy[2];
        while (psin.good() && j < 2) {

            psin >> xy[j];
            //            cout << xy[j] << endl;
            ++j;
        }
        double x = (boost::lexical_cast<double>(xy[0]) > size.width) ? size.width - 1 : boost::lexical_cast<double>(xy[0]);
        double y = (boost::lexical_cast<double>(xy[1]) > size.height) ? size.height - 1 : boost::lexical_cast<double>(xy[1]);
        vertices.push_back(Point(x, y));
        ++i;
    }

    //    vertices.push_back(P1);
    //    vertices.push_back(P2);
    //    vertices.push_back(P3);
    //    vertices.push_back(P4);
    //    vertices.push_back(P5);
    //    vertices.push_back(P6);
    co_ordinates.push_back(vertices);
    drawContours(mask, co_ordinates, 0, Scalar(255, 255, 255), CV_FILLED, 8);

    //    for (vector<Point>::iterator it = vertices.begin(); it != vertices.end(); ++it) {
    //        if (!(irect.contains(*it) && src.at<uchar>(it->x, it->y) == 0)) {
    //            cout << it->x << "," << it->y << endl;
    //        }
    //    }
    Rect rect = boundingRect(co_ordinates[0]);

    Mat res;
    bitwise_and(src, mask, res);

    //    rectangle(res, rect, Scalar(0, 0, 255), 5, LINE_8, 0);

    Mat cropped(rect.width, rect.height, src.type());
    res(rect).copyTo(cropped);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Elapsed time \t" << elapsed_secs << endl;
    cout << "New size width: " << rect.width << " height: " << rect.height << endl;
   
//    Canny(cropped,)
    //    fillPoly()
    namedWindow("Shape", WINDOW_AUTOSIZE);
    imshow("Shape", cropped);

//    vector<int> qualityType;
//    qualityType.push_back(CV_IMWRITE_JPEG_QUALITY);
//    qualityType.push_back(90);
    string output = path + "/" + road + ext;
    imwrite(output, cropped);
    waitKey(0);
    return 0;
}

