/* 
 * File:   main_blob_1.cpp
 * Author: essam
 *
 * Created on August 14, 2015, 5:44 PM
 * 
 * g++ -O2 -std=c++11 -fopenmp main_blobs_1.cpp -o blobs_1.out `pkg-config --libs --cflags opencv` -I/usr/local/include/opencvblobslib -lopencvblobslib -lboost_filesystem -lboost_system
 */
#include <cstdlib>
#include<iostream>
#include<random>
#include <stdio.h>
#include <fstream>
#include<cmath>
#include<random>
#include <opencv/cv.h>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <ctime>
#include<string>
#include <omp.h>
#include <opencvblobslib/blob.h>
#include <opencvblobslib/BlobResult.h>
#include <vector>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include "Utils.h"


using namespace cv;
using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    //Mat& src, vector<Point2d>& lvs, Point& bounding_box_point, int thr, int NUMCORES

    if (argc != 8) {
        cout << " Usage: [Number of cores] [Scaling factor] [Tolerance] [no lanes] [lane width] [Complete image] [Results path]." << endl;
        return -1;
    }
    const double lane_w_tolerance = 1;
    const int NUMCORES = atoi(argv[1]);
    const double scaling = atof(argv[2]);
    const int thr = 100;
    int tolerance = atoi(argv[3]) * scaling;
    int lanes = atoi(argv[4]);
    double lane_width = atof(argv[5]) * lane_w_tolerance*scaling;
    string path = argv[7];
    Utils utils;
    string road_id = "254325088";
    //make directory of all results of a single road ..
    path += "/" + road_id;
    boost::filesystem::path directory(path);

    if (!(boost::filesystem::exists(directory))) {
        if (boost::filesystem::create_directory(directory))
            cout << "Successfully created !" << endl;
    }

    string ext = ".png";
    int p_count = 7;
    //    string shape = "11522.376776458877,15103.0 11504.024386622477,14814.41164474354 11453.009514084446,13908.672590695121 11428.16626305738,13632.04085876345 11401.277063364134,13423.481394582857 11362.426527386293,13273.55277796619 11325.002850915282,13112.498711141932 11245.715290737206,12812.511276097875 10983.078741533796,12015.46634859921 10727.286491824929,11237.855957630849 10394.624883254446,10214.95998875001 10319.585734242113,9980.173276851165 10256.157685045275,9773.534099229992 10194.15241390104,9577.744603558393 10152.019861420627,9432.608267846776 10117.81991483328,9267.767804205068 10102.79329545622,9126.15907161077 10106.375273310825,8965.502869270635 10119.32822946215,8787.84555341935 10124.661994113723,8720.88171051332 9547.71695404924,8672.353311594512 9542.382896712166,8739.317129886975 9529.121127508628,8921.03803688513 9524.102690737334,9111.177129870164 9541.49301880146,9323.507933370985 9584.69680385682,9550.847772041963 9638.326560398227,9743.929768555941 9705.575597538678,9957.231491801977 9767.515130792914,10159.088676732705 9844.637163644162,10400.52697332764 10178.259599784902,11426.40129878578 10434.6881981491,12205.954907715233 10697.421207845402,13003.294992541772 10766.077318513728,13267.316369304315 10799.27627753809,13411.253657829828 10841.798895340764,13576.174016317009 10853.280776287473,13701.753869292312 10876.566768653574,13963.127228131065 10926.110590216515,14848.670642634233 10944.475355134973,15103.0";
    string lshape = "3638.722424212053,1468.0929444110545 3571.4076404965,1442.0892233867496 3029.870956024203,1259.1284019643526 2818.871360631042,1175.9759803358133 2491.352686300741,1017.4467595448327 1775.078731569055,706.9742276935485 1656.7816823001526,655.7316009796579";
    //    RNG random;
    //    int64 time;
    Mat src = imread(argv[6], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    Size size = src.size.operator()();

    vector<Point2d> lvs;
    utils.convert_double_coord(lvs, lshape, p_count, size);

    Point bounding_box_point;
    Mat road = utils.crop(src, lvs, bounding_box_point, tolerance);
    utils.shift_coordinates(lvs, bounding_box_point.x, bounding_box_point.y);
    double sv = utils.road_offset(road, lvs, thr, NUMCORES);

    int dir = utils.sgn(sv);
    sv = abs(sv);
    vector<Point2d> nlvs(lvs.size());
    utils.shift_road_shape(lvs, nlvs, sv, dir);

    utils.plot_line(road, lvs, Scalar::all(255), 4);
    utils.plot_line(road, lvs, Scalar(0, 0, 255), 4);

    string road_path = path + "/roads" + ext;
    imwrite(road_path, road);

    //    Mat opt_road = utils.crop(road, nlvs, bounding_box_point, tolerance);



    return 0;
}

