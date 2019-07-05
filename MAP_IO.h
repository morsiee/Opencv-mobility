/* 
 * File:   MAP_IO.h
 * Author: essam
 *
 * Created on August 13, 2015, 2:26 AM
 */

#ifndef MAP_IO_H
#define	MAP_IO_H


#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include<opencv/cv.h>

using namespace std;
using namespace cv;

class MAP_IO {
public:
    MAP_IO();
    MAP_IO(const MAP_IO& orig);
    virtual ~MAP_IO();

    void read_csv(string path, vector<vector<string> >& data);
    void parse_points(vector<vector<string> >& data, unordered_map<string, vector<Point2d> >& p_records);
    void parse_types(vector<vector<string> >& data, unordered_map<string, vector<string> >& types);
    void plot_map(unordered_map<string, vector<Point2d> >& data, Mat& src, Scalar color, int thickness, bool id_flag);
    void crop_roads(unordered_map<string, vector<Point2d> >& data, Mat& binary_src, Mat& colored_src, unordered_map<string, vector<Point2d> >& nlvs, double tolerance, double shift_factor, string path, string ext);
    void roads_subset(unordered_map<string, vector<Point2d> >& data, unordered_map<string, vector<Point2d> >& data_subset, unordered_map<string, vector<string> >& types, string rd_type);
private:

};

#endif	/* MAP_IO_H */

