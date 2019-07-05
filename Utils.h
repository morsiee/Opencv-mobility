/* 
 * File:   Utils.h
 * Author: essam
 *
 * Created on August 13, 2015, 2:03 AM
 */

#ifndef UTILS_H
#define	UTILS_H

#include <cstdlib>
#include <opencv/cv.h>
#include <vector>

using namespace std;
using namespace cv;

class Utils {
public:
    Utils();
    Utils(const Utils& orig);
    virtual ~Utils();

    template <typename T> int sgn(T val);
    int position(Vec4d& l1, Vec4d& l2);
    int position(Vec4d& l1, Point2d& M);
    double scaleX(double width, double x, double xmin, double xmax);
    double scaleY(double height, double y, double ymin, double ymax);
    double avg_k_smallest(vector<double>& elements);
    void kmean(vector<double>& shift,int clusters, vector<unsigned long>& assignments);
    double getShortestDistance(Vec4d& line1, Vec4d& line2);
    Vec4d solve_circle_Perpendicular_line(double x, double y, double m, double r);
    void shift_line(Vec4d& ol, Vec4d& nl, double sv, int dir);
    void shift_road_shape(vector<Point2d>& lvs, vector<Point2d>& nlvs, double sv);
    void bounding_box(Vec4d& ol, Point2d *vertices, double sv);
    double cross(Point v1, Point v2);
    bool intersectionPoint(Point a1, Point a2, Point b1, Point b2, Point & intPnt);
    bool isPointInBoundingBox(double x1, double y1, double x2, double y2, double px, double py);
    bool segmentsIntersection(Point a1, Point a2, Point b1, Point b2, Rect& roi, Point& pnt);
    void ch_graham_anderson(vector<Point>& in, vector<Point>& out);
    void bounding_free_shape(vector<Vec4d>& lines, vector<Point>& vertices, Rect& roi, double sv);
    void convert_arr_vec(Point2d* tmp_verts, int p_count, vector<Point>& vertices);
    Mat threshSegments(Mat &src, double threshSize);
    void removeSmallBlobs(Mat& im, double size);
    void erosion(Mat &src, int erosion_elem, int erosion_size);
    void dilation(Mat &src, int dilation_elem, int dilation_size);
    void imclose(Mat &src, int dilation_elem, int dilation_size);
    void imopen(Mat &src, int dilation_elem, int dilation_size);
    void iterative_imclose(Mat& im, int structure, int size);
    void iterative_imopen(Mat& im, int structure, int size);
    void imenhance(Mat& src, Mat& dest);
    void thinningIteration(cv::Mat& img, int iter);
    void thinning(const cv::Mat& src, cv::Mat& dst);
    void skeletonizing(const cv::Mat& src, cv::Mat& dst);
    Mat& scanImageAndInvertIterator(Mat& im);
    void get_int_coords(vector<Point>& vertices, string shape, int p_count, Size size);
    void convert_double_coord(vector<Point2d>& vertices, string shape, int p_count, Size size);
    void shift_coordinates(vector<Point2d>& vertices, int tx, int ty);
    void imblur(Mat& src, Mat& dest);
    void road_shape_lines(vector<Point2d>& vertices, vector<Vec4d>& lines);
    double euclidean(Vec4d& l);
    double gps_line_shift(vector<Vec4d>& gps_line, vector<Vec4d>& detected_lines, vector<double>& shift);
    void plot_line(Mat& im, vector<Point2d>& vertices, Scalar color, int thickness);
    static bool isEqual(const Vec4d& _l1, const Vec4d& _l2);
    bool isCorrelated(const vector<Vec4d>& gps_lines, const Vec4d& l);
    void partition(vector<Vec4d>& gps_lines, vector<Vec4d>& detected_lines, vector<Vec4d>& grouped_lines);
    Mat crop(Mat& src, vector<Point2d>& lvs, Point& bounding_box_point, double tolerance);
    int road_offset(Mat& src, vector<Point2d>& lvs, string path, bool write);
    void CannyThreshold(Mat& src, Mat& dst, int threshold, int kernel_size);
    void fillBlobs(Mat& src, Mat& dst, bool blob, int NUMCORES);
private:

};

#endif	/* UTILS_H */

