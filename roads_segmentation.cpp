/* 
 * File:   roads_segmentation.cpp
 * Author: essam
 *
 * Created on August 18, 2015, 12:33 PM
 * 
 * g++ -O2 -std=c++11 -fopenmp Utils.cpp MAP_IO.cpp roads_segmentation.cpp -o segement.o `pkg-config --libs --cflags opencv` -lopencvblobslib -lboost_filesystem -lboost_system
 * ./segement.o 4 8.3 25 ~/work/w_z_6/all/edges.csv ~/work/w_z_6/all/types.csv ~/work/w_z_6/not_res.png ~/work/w_z_6/w_z_6.png ~/work/w_z_6/Aug_25/
 */

#include "MAP_IO.h"
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
#include <opencv2/highgui/highgui_c.h>

using namespace cv;
using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    if (argc != 9) {
        cout << " Usage: [Number of cores] [Scaling factor] [Tolerance] [CSV shape file][CSV types file] [Binary image] [Colored image] [Results path]." << endl;
        return -1;
    }

    const int NUMCORES = atoi(argv[1]);
    const double scaling = atof(argv[2]);
    int tolerance = atoi(argv[3]) * scaling;

    string csv_file = argv[4];
    string csv_types = argv[5];

    Mat binary = imread(argv[6], CV_LOAD_IMAGE_GRAYSCALE);
    Mat src = imread(argv[7], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    string output = argv[8];

    MAP_IO io;
    vector<vector<string> > data;
    io.read_csv(csv_file, data);
    unordered_map<string, vector<Point2d> > p_records;
    io.parse_points(data, p_records);

    data.clear();
    io.read_csv(csv_types, data);
    unordered_map<string, vector<string> > types;
    io.parse_types(data, types);

    boost::filesystem::path out_dir(output);

    if (!(boost::filesystem::exists(out_dir))) {
        if (boost::filesystem::create_directory(out_dir))
            cout << "Successfully created !" << endl;
    }

    cout << "start road types .." << endl;
    for (auto it = types.begin(); it != types.end(); it++) {
        string rd_type = it->first;
        // ignore highway link roads
        //        if (rd_type.find("link") != string::npos) continue;

        string t_output = output + "/" + rd_type;

        //create a directory for a road type ...
        boost::filesystem::path directory(t_output);

        if (!(boost::filesystem::exists(directory))) {
            if (boost::filesystem::create_directory(directory))
                cout << "Successfully created !" << endl;
        }

        unordered_map<string, vector<Point2d> > subset;
        io.roads_subset(p_records, subset, types, rd_type);

        // Increase the tolerance with the residential roads make a conflict with the neighbor road segments ..
        int type_tolerance = tolerance;
        if (rd_type.compare("highway.residential") != 0) {
            type_tolerance = 1.5 * tolerance;
        }
        double shift_factor = 1;
#if 0
        if (rd_type.compare("highway.primary") == 0 || rd_type.compare("highway.secondary") == 0|| rd_type.compare("highway.trunk") == 0) {
            shift_factor = 0.55;
        }
#endif
        unordered_map<string, vector<Point2d> > nlvs;
        io.crop_roads(subset, binary, src, nlvs, type_tolerance, shift_factor, t_output, "png");

        subset.clear();
    }



    return 0;
}

