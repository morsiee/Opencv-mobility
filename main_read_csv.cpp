/* 
 * File:   main_read_csv.cpp
 * Author: essam
 *
 * Created on August 12, 2015, 12:39 PM
 * 
 * g++ -std=c++11 Utils.cpp MAP_IO.cpp main_read_csv.cpp -o read.o `pkg-config --libs --cflags opencv` -lopencvblobslib -lboost_filesystem -lboost_system -lpthread
 */
#include "MAP_IO.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;

/*
 * 
 */
int main(int argc, char** argv) {


    MAP_IO io;
#if 1
    string csv_file = argv[1];
    Mat src = imread(argv[2], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    string out = argv[3];
    Scalar color(0, 0, 255);
    int thickness = 8;

    vector<vector<string> > data;

    io.read_csv(csv_file, data);
//    vector<string> tmp_count = data[1];
//    cout << "the number of elements:\t" << data[1][0] << endl;

    unordered_map<string, vector<Point2d> > p_records;
    io.parse_points(data, p_records);

    io.plot_map(p_records, src, color, thickness, true);
    imwrite(out, src);
#if 0
    io.crop_roads(p_records, src, 8.3 * 60, out, "png");
    vector<string> record = data[10];
    for (vector<string>::size_type i = 0; i != record.size(); i++) {
        cout << record[i] << " ";
    }
    cout << endl;
#endif
    
#else
    string path_shape = argv[1];
    string path_types = argv[2];

    vector<vector<string> > data;
    unordered_map<string, vector<string> > types;
    unordered_map<string, vector<Point2d> > p_records;
    unordered_map<string, vector<Point2d> > subset;

    io.read_csv(path_types, data);
    io.parse_types(data, types);

    data.clear();
    io.read_csv(path_shape, data);
    io.parse_points(data, p_records);

    io.roads_subset(p_records, subset, types, "highway.trunk_link");

    for (auto it = subset.begin(); it != subset.end(); it++) {
        cout << it->first << endl;
    }
    //    unordered_map<string, vector<string> >::const_iterator residential = types.find("highway.residential");
    //    if (residential != types.end()) {
    //        vector<string> rds = residential->second;
    //        cout << "Forth element:\t" << rds[3] << endl;
    //    }

#endif
    return 0;
}



