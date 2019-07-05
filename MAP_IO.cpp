/* 
 * File:   MAP_IO.cpp
 * Author: essam
 * 
 * Created on August 13, 2015, 2:26 AM
 */

#include "MAP_IO.h"
#include "Utils.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include<opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

MAP_IO::MAP_IO() {
}

MAP_IO::MAP_IO(const MAP_IO& orig) {
}

MAP_IO::~MAP_IO() {
}

/**
 * Read CSV file ..
 * 
 * @param path
 * @param data
 */
void MAP_IO::read_csv(string path, vector<vector<string> >& data) {
    ifstream infile;
    infile.open(path, ifstream::in);
    while (infile) {
        string s;
        if (!getline(infile, s)) break;

        istringstream ss(s);
        vector <string> record;

        while (ss) {
            string s;
            if (!getline(ss, s, ' ')) break;
            record.push_back(s);
        }

        data.push_back(record);
    }
    if (!infile.eof()) {
        cerr << "error!\n";
    }
}

/**
 * 
 * @param data
 * @param p_records
 */
void MAP_IO::parse_points(vector<vector<string> >& data, unordered_map<string, vector<Point2d> >& p_records) {
    int count = boost::lexical_cast<int>(data[1][0]);
    for (int i = 3; i != count + 3; i++) {
        string id = data[i][0];
        int no_points = boost::lexical_cast<int>(data[i][1]);
        vector<Point2d> vertices;
        for (int j = 2; j != no_points + 2; j++) {
            string p = data[i][j];
            boost::replace_all(p, ",", " ");
            stringstream psin(p);
            int k = 0;
            string xy[2];
            while (psin.good() && k < 2) {
                psin >> xy[k];
                ++k;
            }
            Point2d dP(boost::lexical_cast<double>(xy[0]), boost::lexical_cast<double>(xy[1]));
            vertices.push_back(dP);

        }
        p_records.insert(make_pair(id, vertices));
    }
}

/**
 * 
 * @param data
 * @param p_records
 */
void MAP_IO::parse_types(vector<vector<string> >& data, unordered_map<string, vector<string> >& types) {
    int count = boost::lexical_cast<int>(data[1][0]);
    for (int i = 3; i != count + 3; i++) {
        string id = data[i][0];
        int no_points = boost::lexical_cast<int>(data[i][1]);
        vector<string> rds;
        string p = data[i][2];

        boost::replace_all(p, ",", " ");
        stringstream psin(p);
        int k = 0;
        string seg;
        while (psin.good() && k < no_points) {
            psin >> seg;
            rds.push_back(seg);
            ++k;
        }
        types.insert(make_pair(id, rds));
    }
}

/**
 * Plot map roads using GPS data ...
 * 
 * @param data
 * @param src
 * @param color
 * @param thickness
 */
void MAP_IO::plot_map(
        unordered_map<string, vector<Point2d> >& data,
        Mat& src,
        Scalar color,
        int thickness,
        bool id_flag) {

    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 0.8;
    int font_thickness = 3;

    Utils utils;
    for (unordered_map<string, vector<Point2d> >::iterator it = data.begin(); it != data.end(); ++it) {
        string id = it->first;
        vector<Point2d> vertices = it->second;
        utils.plot_line(src, vertices, color, thickness);
        if (id_flag) {
            // then put the text itself
            putText(src, id, vertices[0], fontFace, fontScale, Scalar::all(255), font_thickness, 8);
        }

    }
}

/**
 * 
 * @param data
 * @param binary_src
 * @param colored_src
 * @param nlvs
 * @param tolerance
 * @param path
 * @param ext
 */
void MAP_IO::crop_roads(
        unordered_map<string, vector<Point2d> >& data,
        Mat& binary_src,
        Mat& colored_src,
        unordered_map<string, vector<Point2d> >& nlvs,
        double tolerance,
        double shift_factor,
        string path,
        string ext) {

    Utils utils;
    //    boost::filesystem::path directory(path);
    //
    //    if (!(boost::filesystem::exists(directory))) {
    //        if (boost::filesystem::create_directory(directory))
    //            cout << "Successfully created !" << endl;
    //    }

    //    int i = 0;
    for (unordered_map<string, vector<Point2d> >::iterator it = data.begin(); it != data.end(); ++it) {

        string id = it->first;
        vector<Point2d> lvs = it->second;

        // create director for each road segment images ..

        string rd_dir = path + "/" + id;
        boost::filesystem::path rd_directory(rd_dir);

        if (!(boost::filesystem::exists(rd_directory))) {
            if (boost::filesystem::create_directory(rd_directory))
                cout << "Successfully created !" << endl;
        }

        //check the length of the road, if it is less than 5m ignore it.
        //        Vec4d aprox_line(lvs[0].x, lvs[0].y, lvs[lvs.size() - 1].x, lvs[lvs.size() - 1].y);
        //        double length = utils.euclidean(aprox_line);
        //        cout << "Approximate road: " << id << "\tlength: " << length << endl;
        //        if (length < 5)continue;
        // For debug
        //        if(vertices.size()<3)continue;
        try {
            Point coord;
            Mat imroad = utils.crop(binary_src, lvs, coord, tolerance);

            Mat imcroad = utils.crop(colored_src, lvs, coord, tolerance);

            imwrite(rd_dir + "/binary." + ext, imroad);
            // shift coordinates of the GPS road to match the cropped image ...
            utils.shift_coordinates(lvs, coord.x, coord.y);

            int shift = utils.road_offset(imroad, lvs, rd_dir, true) * shift_factor;
            cout << "Road: " << id << "\tshift: " << shift << endl;


            vector<Point2d> rnlvs(lvs.size());
            utils.shift_road_shape(lvs, rnlvs, shift);

            nlvs.insert(make_pair(id, rnlvs));


            // plot lines the original lines and the shifted lines...
            utils.plot_line(imcroad, lvs, Scalar::all(255), 8);
            utils.plot_line(imcroad, rnlvs, Scalar(0, 0, 255), 8);

            // write detected lines image...
            imwrite(rd_dir + "/colored road segment." + ext, imcroad);
        } catch (cv::Exception) {

            continue;
        }
        //for debug ..
        //        if (i == 30)break;
        //        i++;
    }
}

/**
 * Split data based on the road type  ...
 * 
 * @param data
 * @param data_subset
 * @param types
 * @param rd_type
 */
void MAP_IO::roads_subset(
        unordered_map<string, vector<Point2d> >& data,
        unordered_map<string, vector<Point2d> >& data_subset,
        unordered_map<string, vector<string> >& types,
        string rd_type) {

    unordered_map<string, vector<string> >::const_iterator residential = types.find(rd_type);
    if (residential != types.end()) {
        vector<string> rds = residential->second;
        for (vector<string>::const_iterator it = rds.begin(); it != rds.end(); it++) {
            unordered_map<string, vector<Point2d> >::const_iterator rd_it = data.find(*it);
            if (rd_it != data.end()) {
                data_subset.insert(*rd_it);
            }
        }

        //        cout << "The size of the subset:\t" << data_subset.size() << endl;

    } else {
        cout << "Error road type is not available ...." << endl;
    }
}