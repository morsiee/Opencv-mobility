/* 
 * File:   Utils.cpp
 * Author: essam
 * 
 * Created on August 13, 2015, 2:03 AM
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
#include <opencv/cv.h>
//#include<opencv2/imgproc.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/core/utility.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv/cv.h>
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
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <vector>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include <dlib/clustering.h>
#include <dlib/rand.h>
#include <opencv2/imgproc/imgproc.hpp>
//#include <CGAL/Kernel/interface_macros.h>

#include "Utils.h"

using namespace cv;
using namespace std;

Utils::Utils() {
}

Utils::Utils(const Utils& orig) {
}

Utils::~Utils() {
}

/**
 * Use the sign of the determinant of vectors (AB,AM), where M(X,Y) is the query point
 * Position = sign( (Bx-Ax)*(Y-Ay) - (By-Ay)*(X-Ax) )
 * It is 0 on the line, and +1 on one side, -1 on the other side.
 * @param l
 * @param p
 * @return 
 */
template <typename T> int Utils::sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

/**
 * Use the sign of the determinant of vectors (AB,AM), where M(X,Y) is the query point
 * Position = sign( (Bx-Ax)*(Y-Ay) - (By-Ay)*(X-Ax) )
 * It is 0 on the line, and +1 on one side, -1 on the other side.
 * @param l
 * @param p
 * @return 
 */
int Utils::position(Vec4d& l1, Vec4d& l2) {


    Point2d A, B, M;
    // first line representation points ..
    A.x = l1[0];
    A.y = l1[1];
    B.x = l1[2];
    B.y = l1[3];
    // Query point
    M.x = l2[0];
    M.y = l2[1];
    return sgn((B.x - A.x)*(M.y - A.y) - (B.y - A.y)*(M.x - A.x));
}

/**
 * Use the sign of the determinant of vectors (AB,AM), where M(X,Y) is the query point
 * Position = sign( (Bx-Ax)*(Y-Ay) - (By-Ay)*(X-Ax) )
 * It is 0 on the line, and +1 on one side, -1 on the other side.
 * @param l1
 * @param M
 * @return 
 */
int Utils::position(Vec4d& l1, Point2d& M) {


    Point2d A, B;
    // first line representation points ..
    A.x = l1[0];
    A.y = l1[1];
    B.x = l1[2];
    B.y = l1[3];

    return sgn((B.x - A.x)*(M.y - A.y) - (B.y - A.y)*(M.x - A.x));
}

/**
 * 
 * @param width
 * @param x
 * @param xmin
 * @param xmax
 * @return 
 */
double Utils::scaleX(double width, double x, double xmin, double xmax) {
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
double Utils::scaleY(double height, double y, double ymin, double ymax) {
    return height * (ymax - y) / (ymax - ymin);
}
/**
 * get the shortest distance between lines ..
 * 
 * @param line1
 * @param line2
 * @return double distance value .
 */
const double EPS = 0.00000001;

double Utils::getShortestDistance(Vec4d& line1, Vec4d& line2) {


    //    cout << "line1[3]" << line1[3] << endl;
    Point2d delta21;
    delta21.x = line1[2] - line1[0];
    delta21.y = line1[3] - line1[1];
    //    delta21.z = line1->z2 - line1->z1;

    Point2d delta41;
    delta41.x = line2[2] - line2[0];
    delta41.y = line2[3] - line2[1];
    //    delta41.z = line2->z2 - line2->z1;

    Point2d delta13;
    delta13.x = line1[0] - line2[0];
    delta13.y = line1[1] - line2[1];
    //    delta13.z = line1->z1 - line2->z1;


    double a = delta21.ddot(delta21);
    double b = delta21.ddot(delta41);
    double c = delta41.ddot(delta41);
    double d = delta21.ddot(delta13);
    double e = delta41.ddot(delta13);
    double D = a * c - b * b;

    double sc, sN, sD = D;
    double tc, tN, tD = D;

    if (D < EPS) {
        sN = 0.0;
        sD = 1.0;
        tN = e;
        tD = c;
    } else {
        sN = (b * e - c * d);
        tN = (a * e - b * d);
        if (sN < 0.0) {
            sN = 0.0;
            tN = e;
            tD = c;
        } else if (sN > sD) {
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {
        tN = 0.0;

        if (-d < 0.0)
            sN = 0.0;
        else if (-d > a)
            sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    } else if (tN > tD) {
        tN = tD;
        if ((-d + b) < 0.0)
            sN = 0;
        else if ((-d + b) > a)
            sN = sD;
        else {
            sN = (-d + b);
            sD = a;
        }
    }

    if (abs(sN) < EPS) sc = 0.0;
    else sc = sN / sD;
    if (abs(tN) < EPS) tc = 0.0;
    else tc = tN / tD;

    Point2d dP;
    dP.x = delta13.x + (sc * delta21.x) - (tc * delta41.x);
    dP.y = delta13.y + (sc * delta21.y) - (tc * delta41.y);
    //    dP.z = delta13.z + (sc * delta21.z) - (tc * delta41.z);

    return position(line1, line2) * sqrt(dP.ddot(dP));
}

/**
 * The same functionality of solve circle line, you obtain the same values
 * by passing (-1/m) instead of (m) to find points on the same line.
 *
 * @param x
 * @param y
 * @param m slop
 * @param r distance/width of the polygon
 * @return
 */
Vec4d Utils::solve_circle_Perpendicular_line(double x, double y, double m, double r) {
    if (m == 0) {
        /**
         * The perpendicular line to the give segment is horizontal
         */
        return Vec4d(x, y + r, x, y - r);

    } else if (m != m) {

        /**
         * The perpendicular line to the give segment is horizontal
         */
        return Vec4d(x + r, y, x - r, y);
    } else {

        double A = x;
        double B = y;
        double C = (y + x / m);
        double a = 1 + (1 / pow(m, 2));
        double b = (2 * B / m) - 2 * A - (2 * C / m);
        double c = pow(A, 2) + pow(B, 2) + pow(C, 2) - 2 * B * C - pow(r, 2);

        double q_rs = sqrt(pow(b, 2) - 4 * a * c);
        double q_x = (-b + q_rs) / (2 * a);
        double q_x1 = (-b - q_rs) / (2 * a);

        double q_y = -q_x / m + C;
        double q_y1 = -q_x1 / m + C;
        return Vec4d(q_x, q_y, q_x1, q_y1);
    }
}

/**
 * Shift a line into specific direction with sift value ..
 * 
 * @param Oringinal line
 * @param new shifted line
 * @param shift value
 * @param direction
 */
void Utils::shift_line(Vec4d& ol, Vec4d& nl, double sv, int dir) {
    double x, y, x1, y1;
    x = ol[0];
    y = ol[1];
    x1 = ol[2];
    y1 = ol[3];


    // Shapes points list
    double m = (y - y1) / (x - x1);
    Vec4d pl1 = solve_circle_Perpendicular_line(x, y, m, sv);
    cout << "PL1 coordinates:\t" << pl1[0] << "," << pl1[1] << "," << pl1[2] << "," << pl1[3] << endl;
    Vec4d pl2 = solve_circle_Perpendicular_line(x1, y1, m, sv);
    cout << "PL2 coordinates:\t" << pl2[0] << "," << pl2[1] << "," << pl2[2] << "," << pl2[3] << endl;

    //check positions ..
    Point2d pl1_p1(pl1[0], pl1[1]);
    int pl1_d1 = position(ol, pl1_p1);
    Point2d pl2_p1(pl2[0], pl2[1]);
    int pl2_d1 = position(ol, pl2_p1);
    //    double nl[4];
    //first point
    if (sgn(dir) == pl1_d1) {
        nl[0] = pl1[0];
        nl[1] = pl1[1];
    } else {
        nl[0] = pl1[2];
        nl[1] = pl1[3];
    }
    //second point
    if (sgn(dir) == pl2_d1) {
        nl[2] = pl2[0];
        nl[3] = pl2[1];
    } else {
        nl[2] = pl2[2];
        nl[3] = pl2[3];
    }
}

/**
 * Shift the GPS road shape coordinates into the new positions ..
 * @param lvs
 * @param nlvs
 * @param sv
 * @param dir
 */
void Utils::shift_road_shape(vector<Point2d>& lvs, vector<Point2d>& nlvs, double sv) {

    int dir = sgn(sv);
    sv = abs(sv);

    for (vector<Point2d>::size_type i = 0; i != lvs.size() - 1; i++) {
        Point2d p1 = lvs[i];
        Point2d p2 = lvs[i + 1];
        Vec4d ol(p1.x, p1.y, p2.x, p2.y);
        Vec4d nl;
        shift_line(ol, nl, sv, dir);
        Point2d np(nl[0], nl[1]);
        nlvs[i] = np;
        if (i == lvs.size() - 2) {
            //add the last point in the new shape ...
            Point2d np1(nl[2], nl[3]);
            nlvs[i + 1] = np1;
        }
    }
}

/**
 * find the vertices of the bounding box started from the upper left corner in counter clockwise order ..
 * 
 * @param ol
 * @param vertices
 * @param sv
 */
void Utils::bounding_box(Vec4d& ol, Point2d *vertices, double sv) {
    const int dir = -1;
    double x, y, x1, y1;
    x = ol[0];
    y = ol[1];
    x1 = ol[2];
    y1 = ol[3];


    // Shapes points list
    double m = (y - y1) / (x - x1);
    Vec4d pl1 = solve_circle_Perpendicular_line(x, y, m, sv);
    Vec4d pl2 = solve_circle_Perpendicular_line(x1, y1, m, sv);

    //check positions ..
    Point2d pl1_p1(pl1[0], pl1[1]);
    int pl1_d1 = position(ol, pl1_p1);
    Point2d pl2_p1(pl2[0], pl2[1]);
    int pl2_d1 = position(ol, pl2_p1);

    //first point
    if (sgn(dir) == pl1_d1) {
        vertices[0] = Point2d(pl1[0], pl1[1]);
        vertices[1] = Point2d(pl1[2], pl1[3]);
    } else {
        vertices[1] = Point2d(pl1[0], pl1[1]);
        vertices[0] = Point2d(pl1[2], pl1[3]);
    }
    //second point
    if (sgn(dir) == pl2_d1) {
        vertices[3] = Point2d(pl2[0], pl2[1]);
        vertices[2] = Point2d(pl2[2], pl2[3]);
    } else {
        vertices[2] = Point2d(pl2[0], pl2[1]);
        vertices[3] = Point2d(pl2[2], pl2[3]);
    }


}

/**
 * Find bounding free shape poly-lines ..
 * 
 * @param lines
 * @param vertices
 * @param sv
 */
//void bounding_free_shape(vector<Vec4d>& lines, Point2d *vertices, double sv) {
//    const int count = 2 * lines.size() + 2;
//    for (vector<Vec4d>::size_type i = 0; i != lines.size(); i++) {
//        Vec4d l = lines[i];
//        Point2d vl[4];
//        bounding_box(l, vl, sv);
//        if (i != (lines.size() - 1)) {
//            // add only the first two vertices to the polyline vertices ..
//            vertices[i] = vl[0];
//            vertices[(count - 1) - i] = vl[1];
//        } else {
//            // if it is the last line, add four vertices to the poly-line shape ..
//            // the first two elements ..
//            vertices[i] = vl[0];
//            vertices[(count - 1) - i] = vl[1];
//            // Add the last two points ..
//            vertices[i + 1] = vl[3];
//            vertices[(count - 1)-(i + 1)] = vl[2];
//
//        }
//
//    }
//}

/**
 * Cross product ..
 * 
 * @param v1
 * @param v2
 * @return 
 */
double Utils::cross(Point v1, Point v2) {
    return v1.x * v2.y - v1.y * v2.x;
}

/**
 * Check the intersection between two lines 
 * the full illustration of the cross product is present in the following link:
 * http://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function
 * http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
 * 
 * @param a1
 * @param a2
 * @param b1
 * @param b2
 * @param intPnt
 * @return 
 */
bool Utils::intersectionPoint(Point a1, Point a2, Point b1, Point b2, Point & intPnt) {
    Point p = a1;
    Point q = b1;
    Point r(a2 - a1);
    Point s(b2 - b1);

    if (cross(r, s) == 0) {
        return false;
    }

    double t = cross(q - p, s) / cross(r, s);

    intPnt = p + t*r;
    return true;
}

bool Utils::isPointInBoundingBox(double x1, double y1, double x2, double y2, double px, double py) {

    // For Bounding Box
    double x, y;
    x = x2;
    y = y2;
    if (x1 < x2) x = x1;
    if (y1 < y2) y = y1;
    Rect roi(x, y, abs(x2 - x1), abs(y2 - y1));
    Point p(px, py);
    return roi.contains(p);
}

/**
 * Depends on solving the two lines ..
 * 
 * @param a1
 * @param a2
 * @param b1
 * @param b2
 * @param pnt
 * @return 
 */
bool Utils::segmentsIntersection(Point a1, Point a2, Point b1, Point b2, Rect& roi, Point& pnt) {

    double l1x1 = a1.x;
    double l1y1 = a1.y;
    double l1x2 = a2.x;
    double l1y2 = a2.y;

    double l2x1 = b1.x;
    double l2y1 = b1.y;
    double l2x2 = b2.x;
    double l2y2 = b2.y;

    double px, py;
    double m1, c1, m2, c2;
    double dx1, dx2, dy1, dy2;

    dx1 = l1x2 - l1x1;
    dx2 = l2x2 - l2x1;
    dy1 = l1y2 - l1y1;
    dy2 = l2y2 - l2y1;

    if (dx1 == 0) {
        cout << "Line 1 vertical" << endl;
        // Line 1 is vertical ....
        m2 = dy2 / dx2;
        // intercept c = y - mx
        c2 = l2y1 - m2 * l2x1; // which is same as y2 - slope * x2
        px = l1x1;
        px = (px == 0) ? 1 : px - 1;
        py = m2 * px + c2;

        //check if the obtained point lie on the line segment not the extension of the line ..

        //                (pnt.y < l2y1 && pnt.y > l2y2) || (pnt.y > l2y1 && pnt.y < l2y2)
        if (isPointInBoundingBox(l2x1, l2y1, l2x2, l2y2, px, py)) {
            pnt.x = px;
            pnt.y = py;
            return true;
        }
    } else if (dx2 == 0) {
        cout << "Line 2 vertical" << endl;
        // Line 2 is vertical ....
        m1 = dy1 / dx1;
        // y = mx + c
        // intercept c = y - mx
        c1 = l1y1 - m1 * l1x1; // which is same as y2 - slope * x2
        px = l2x1;
        px = (px == 0) ? 1 : px - 1;
        py = m1 * px + c1;
        //check if the obtained point lie on the line segment not the extension of the line ..
        //        (pnt.y < l1y1 && pnt.y > l1y2) || (pnt.y > l1y1 && pnt.y < l1y2)
        if (isPointInBoundingBox(l1x1, l1y1, l1x2, l1y2, px, py)) {
            pnt.x = px;
            pnt.y = py;

            return true;
        }

    } else {
        cout << "Both are non-vertical lines .." << endl;
        m1 = dy1 / dx1;
        m2 = dy2 / dx2;

        // y = mx + c
        // intercept c = y - mx
        c1 = l1y1 - m1 * l1x1; // which is same as y2 - slope * x2
        // y = mx + c
        // intercept c = y - mx
        c2 = l2y1 - m2 * l2x1; // which is same as y2 - slope * x2

        if ((m1 - m2) == 0)
            return false;
        else {
            px = (c2 - c1) / (m1 - m2);
            //            px -= 1;
            px = (px == 0) ? 1 : px;
            px = (px == roi.width) ? roi.width - 1 : px;
            py = m1 * px + c1;
            py = (py == 0) ? 1 : py;
            py = (py == roi.height) ? roi.height - 1 : py;
        }
        cout << "px: " << px << " py: " << py << endl;

        if (dy1 == 0) {
            if (isPointInBoundingBox(l2x1, l2y1, l2x2, l2y2, px, py)) {
                pnt.x = px;
                pnt.y = py;
                return true;
            }
        } else if (dy2 == 0) {
            if (isPointInBoundingBox(l1x1, l1y1, l1x2, l1y2, px, py)) {
                pnt.x = px;
                pnt.y = py;
                return true;
            }
        } else {
            if (isPointInBoundingBox(l2x1, l2y1, l2x2, l2y2, px, py) && isPointInBoundingBox(l1x1, l1y1, l1x2, l1y2, px, py)) {
                pnt.x = px;
                pnt.y = py;
                return true;
            }
        }
    }


    return false;
}

void Utils::ch_graham_anderson(vector<Point>& in, vector<Point>& out) {

    //    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    //    typedef K::Point_2 Point_2;
    //
    //    vector<Point_2> v(in.size());
    //    for(vector<Point>::size_type )
}

/**
 * Find bounding free shape poly-lines ..
 * 
 * @param lines
 * @param vertices
 * @param sv
 */
void Utils::bounding_free_shape(vector<Vec4d>& lines, vector<Point>& vertices, Rect& roi, double sv) {
    vertices.clear();
    const int count = 2 * lines.size() + 2;
    //    vector<Point> vertices[count];
    Point tv[count];

    for (vector<Vec4d>::size_type i = 0; i < lines.size(); i++) {

        Vec4d l = lines[i];
        //        cout << "PL" << " ( " << l[0] << " , " << l[1] << " )" << endl;
        //        cout << "PL" << " ( " << l[2] << " , " << l[3] << " )" << endl;

        Point2d vl[4];
        bounding_box(l, vl, sv);
        if (i != (lines.size() - 1)) {
            // add only the first two vertices to the polyline vertices ..
            tv[i] = vl[0];
            tv[(count - 1) - i] = vl[1];
        } else {
            // if it is the last line, add four vertices to the poly-line shape ..
            // the first two elements ..
            tv[i] = vl[0];
            tv[(count - 1) - i] = vl[1];
            // Add the last two points ..
            tv[i + 1] = vl[3];
            tv[(count - 1)-(i + 1)] = vl[2];

        }


    }

    /*
     * If it is only one line, the output will be a convex hull ....
     */
    if (lines.size() == 1) {
        vector<Point> hull;
        //        vector<Point2d> v(std::begin(tv), std::end(tv));
        vector<Point> v;
        for (int i = 0; i < 4; i++)
            v.push_back(tv[i]);
        convexHull(Mat(v), hull, false);
        for (int i = 0; i < 4; i++)
            tv[i] = hull[i];
        //        tv = &hull;
    }

    //define ROI vertices ..
    Point lt(roi.x, roi.y + roi.height);
    Point lb(roi.x, roi.y);
    Point rb(roi.x + roi.width, roi.y);
    Point rt(roi.x + roi.width, roi.y + roi.height);

    //print free shape ..
    cout << "print free shape .." << endl;

    for (int i = 0; i < count; i++) {

        if (!roi.contains(tv[i])) {
            //find the intersection point between  a line from the outside point and inside point and the boundaries of the rectangle. 
            Point p1 = tv[i];
            Point p2, tp;
            if (i != 0) {
                p2 = tv[i - 1];
            } else {
                p2 = tv[i + 1];
            }
            //
            //            cout << "bounding box" << " ( " << roi.width << " , " << roi.height << " )" << endl;
            //            cout << "before" << " ( " << tv[i].x << " , " << tv[i].y << " )" << endl;
            //            
            if (segmentsIntersection(p1, p2, lt, lb, roi, tp)) {
                cout << "intersect left .." << endl;
                tv[i] = tp;
            } else if (segmentsIntersection(p1, p2, lb, rb, roi, tp)) {
                cout << "intersect down .." << endl;
                tv[i] = tp;
            } else if (segmentsIntersection(p1, p2, rb, rt, roi, tp)) {
                cout << "intersect right .." << endl;
                tv[i] = tp;
            } else if (segmentsIntersection(p1, p2, rt, lt, roi, tp)) {
                cout << "intersect top .." << endl;
                tv[i] = tp;
            }

            //            else {
            //
            //                //                cout << "error: can't find the intersection point" << endl;
            //                if (tv[i].x < roi.x) {
            //                    tv[i].x = roi.x + 1;
            //                } else if (tv[i].x > roi.x + roi.width) {
            //                    tv[i].x = roi.x + roi.width - 1;
            //                }
            //
            //                if (tv[i].y < roi.y) {
            //                    tv[i].y = roi.y + 1;
            //                } else if (tv[i].y > roi.y + roi.height) {
            //                    tv[i].y = roi.y + roi.height - 1;
            //                }
            //            }


            //                        if (tv[i].x < roi.x) {
            //                            tv[i].x = roi.x + 1;
            //                        } else if (tv[i].x > roi.x + roi.width) {
            //                            tv[i].x = roi.x + roi.width - 1;
            //                        }
            //            
            //                        if (tv[i].y < roi.y) {
            //                            tv[i].y = roi.y + 1;
            //                        } else if (tv[i].y > roi.y + roi.height) {
            //                            tv[i].y = roi.y + roi.height - 1;
            //                        }
            //            cout << "After" << " ( " << tv[i].x << " , " << tv[i].y << " )" << endl;

        }
        //            vertices[i] = tv[i];
        //          
        vertices.push_back(tv[i]);
        cout << i << " ( " << vertices[i].x << " , " << vertices[i].y << " )" << endl;
    }
    //    return vertices;
}

void Utils::convert_arr_vec(Point2d* tmp_verts, int p_count, vector<Point>& vertices) {
    for (int i = 0; i != p_count; i++) {
        vertices[i] = Point(tmp_verts[i].x, tmp_verts[i].y);
    }
}

/**
 * Get the biggest blobs .. 
 * unfortunately it is not working well ...
 * 
 * @param src
 * @param threshSize
 * @return 
 */
Mat Utils::threshSegments(Mat &src, double threshSize) {
    // FindContours:
    vector<vector<Point2d> > contours;
    vector<Vec4d> hierarchy;
    Mat srcBuffer, output;
    src.copyTo(srcBuffer);
    findContours(srcBuffer, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);

    vector<vector<Point2d> > allSegments;

    // For each segment:
#pragma omp for
    for (size_t i = 0; i < contours.size(); ++i) {
        drawContours(srcBuffer, contours, i, Scalar(200, 0, 0), 1, 8, hierarchy, 0, Point2d());
        Rect brect = boundingRect(contours[i]);
        rectangle(srcBuffer, brect, Scalar(255, 0, 0));

        int result;
        vector<Point2d> segment;

        for (unsigned int row = brect.y; row < brect.y + brect.height; ++row) {
            for (unsigned int col = brect.x; col < brect.x + brect.width; ++col) {
                result = pointPolygonTest(contours[i], Point2d(col, row), false);
                if (result == 1 || result == 0) {
                    segment.push_back(Point2d(col, row));
                }
            }
        }
        allSegments.push_back(segment);
    }

    output = Mat::zeros(src.size(), CV_8U);
    int totalSize = output.rows * output.cols;
#pragma omp for
    for (int segmentCount = 0; segmentCount < allSegments.size(); ++segmentCount) {
        vector<Point2d> segment = allSegments[segmentCount];
        if (segment.size() > totalSize * threshSize) {
            for (int idx = 0; idx < segment.size(); ++idx) {
                output.at<uchar>(segment[idx].y, segment[idx].x) = 255;
            }
        }
    }

    return output;
}

/**
 * Replacement for Matlab's bwareaopen()
 * Input image must be 8 bits, 1 channel, black and white (objects)
 * with values 0 and 255 respectively
 */
void Utils::removeSmallBlobs(Mat& im, double size) {
    // Only accept CV_8UC1
    if (im.channels() != 1 || im.type() != CV_8U)
        return;

    // Find all contours
    std::vector<std::vector<Point2d> > contours;
    findContours(im.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cout << "Contours size from remove small blobs:\t" << contours.size() << endl;
#pragma omp for
    for (int i = 0; i < contours.size(); i++) {
        // Calculate contour area
        double area = contourArea(contours[i]);

        // Remove small objects by drawing the contour with black color
        if (area > 0 && area <= size) {
            cout << "eliminate small blob ..." << endl;
            drawContours(im, contours, i, CV_RGB(0, 0, 0), -1);

        }
    }
    std::vector<std::vector<Point2d> > contours1;
    findContours(im.clone(), contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cout << "Contours size after remove small blobs:\t" << contours1.size() << endl;
}

/**
 * Perform morphological erosion with three structures; MORPH_RECT, MORPH_CROSS and MORPH_ELLIPSE.
 * @param src
 * @param erosion_elem
 * @param erosion_size
 */
void Utils::erosion(Mat &src, int erosion_elem, int erosion_size) {
    int erosion_type;
    if (erosion_elem == 0) {
        erosion_type = MORPH_RECT;
    } else if (erosion_elem == 1) {
        erosion_type = MORPH_CROSS;
    } else if (erosion_elem == 2) {
        erosion_type = MORPH_ELLIPSE;
    }

    Mat element = getStructuringElement(erosion_type,
            Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point2d(erosion_size, erosion_size));
    /// Apply the erosion operation
    erode(src, src, element);
    //    imshow("Erosion Demo", erosion_dst);
}

/** @function Dilation */
void Utils::dilation(Mat &src, int dilation_elem, int dilation_size) {
    int dilation_type;
    if (dilation_elem == 0) {
        dilation_type = MORPH_RECT;
    } else if (dilation_elem == 1) {
        dilation_type = MORPH_CROSS;
    } else if (dilation_elem == 2) {
        dilation_type = MORPH_ELLIPSE;
    }

    Mat element = getStructuringElement(dilation_type,
            Size(2 * dilation_size + 1, 2 * dilation_size + 1),
            Point2d(dilation_size, dilation_size));
    //    Apply the dilation operation
    dilate(src, src, element);
    //    imshow("Dilation Demo", dilation_dst);
}

/**
 * Morphological imclose ..
 * @param src
 * @param dilation_elem
 * @param dilation_size
 */
void Utils::imclose(Mat &src, int dilation_elem, int dilation_size) {
    int dilation_type;
    if (dilation_elem == 0) {
        dilation_type = MORPH_RECT;
    } else if (dilation_elem == 1) {
        dilation_type = MORPH_CROSS;
    } else if (dilation_elem == 2) {
        dilation_type = MORPH_ELLIPSE;
    }

    Mat element = getStructuringElement(dilation_type,
            Size(2 * dilation_size + 1, 2 * dilation_size + 1),
            Point2d(dilation_size, dilation_size));


    morphologyEx(src, src, MORPH_CLOSE, element);
    //    imshow("Dilation Demo", dilation_dst);
}

/**
 * iterative imclose till the difference becomes zero ....
 * 
 * @param im
 * @param dilation_elem
 * @param dilation_size
 */
void Utils::iterative_imclose(Mat& im, int structure, int size) {
    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;
    do {
        //        cout << "imclose .." << endl;
        imclose(im, structure, size);
        imclose(im, structure, size);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    } while (countNonZero(diff) > 0);
}

/**
 * iterative imopen till the difference becomes zero ....
 * 
 * @param im
 * @param dilation_elem
 * @param dilation_size
 */
void Utils::iterative_imopen(Mat& im, int structure, int size) {
    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;
    do {
        //        cout << "imclose .." << endl;
        imopen(im, structure, size);
        imopen(im, structure, size);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    } while (countNonZero(diff) > 0);
}

/**
 * Morphological imopen ..
 * @param src
 * @param dilation_elem
 * @param dilation_size
 */
void Utils::imopen(Mat &src, int dilation_elem, int dilation_size) {
    int dilation_type;
    if (dilation_elem == 0) {
        dilation_type = MORPH_RECT;
    } else if (dilation_elem == 1) {
        dilation_type = MORPH_CROSS;
    } else if (dilation_elem == 2) {
        dilation_type = MORPH_ELLIPSE;
    }

    Mat element = getStructuringElement(dilation_type,
            Size(2 * dilation_size + 1, 2 * dilation_size + 1),
            Point2d(dilation_size, dilation_size));
    /// Apply the open followed by close to remove small elements and fill connected buildings....
    morphologyEx(src, src, MORPH_OPEN, element);

}

/**
 * ehance image details ..
 * 
 * @param src
 * @param dest
 */
void Utils::imenhance(Mat& src, Mat& dest) {
#if 0
    // important !!
    // may be useful after extracting individual roads ....
    vector<Mat> channels;
    Mat ihe;

    cvtColor(src, ihe, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
    split(ihe, channels); //split the image into channels
    cout << "channel size:\t" << channels.size() << endl;
    //    for(vector<Mat>::size_type i=0; i!=channels.size();i++)
    //    equalizeHist(channels[i], channels[i]); //equalize histogram on the 1st channel (Y)

    equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)

    merge(channels, ihe); //merge 3 channels including the modified 1st channel into one image
    cvtColor(ihe, src, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)

#else
    cv::Mat lab_image;
    cv::cvtColor(src, lab_image, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes); // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    // convert back to RGB
    cv::cvtColor(lab_image, src, CV_Lab2BGR);

#endif

}

void Utils::thinningIteration(cv::Mat& img, int iter) {
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof (uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne; // north (pAbove)
    uchar *we, *me, *ea;
    uchar *sw, *so, *se; // south (pBelow)

    uchar *pDst;

    // initialize row pointers
    pAbove = NULL;
    pCurr = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);
#pragma omp for
    for (y = 1; y < img.rows - 1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr = pBelow;
        pBelow = img.ptr<uchar>(y + 1);

        pDst = marker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (x = 1; x < img.cols - 1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x + 1]);
            we = me;
            me = ea;
            ea = &(pCurr[x + 1]);
            sw = so;
            so = se;
            se = &(pBelow[x + 1]);

            int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                    (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                    (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                    (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
        }
    }

    img &= ~marker;
}

/**
 * Perform thinning operation but with preserving the topology ...
 * 
 * @param src
 * @param dst
 */
void Utils::skeletonizing(const cv::Mat& src, cv::Mat& dst) {
    src.copyTo(dst);

    cv::Mat skel(dst.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;
    do {
        cv::erode(dst, eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(dst, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(dst);

        done = (cv::countNonZero(dst) == 0);
    } while (!done);

}

/**
 * Function for thinning the given binary image --> preserve topology .....
 *
 * Parameters:
 * 		src  The source image, binary with range = [0,255]
 * 		dst  The destination image
 */
void Utils::thinning(const cv::Mat& src, cv::Mat& dst) {
    dst = src.clone();
    dst /= 255; // convert to binary image

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

Mat& Utils::scanImageAndInvertIterator(Mat& im) {
    // accept only char type matrices
    CV_Assert(im.depth() != sizeof (uchar));

    const int channels = im.channels();
    //    int nRows = im.rows;
    //    int nCols = im.cols * channels;
    //    cout << "channels \t" << channels << endl;

    switch (channels) {
        case 1:
        {

            //#pragma omp parallel
            //            {
            //                MatIterator_<uchar>  end = im.end<uchar>();
            //#pragma omp for
#pragma omp single nowait
            {
                for (MatIterator_<uchar> it = im.begin<uchar>(); it != im.end<uchar>(); ++it) {
#pragma omp task
                    if (*it == 0) {
                        *it = 255;
                    }
                }
            }
            break;
        }
        case 3:
        {
            //#pragma omp parallel
            //            {
            //                MatIterator_<Vec3b>  end = im.end<Vec3b>();
            //#pragma omp for
#pragma omp single nowait
            {
                for (MatIterator_<Vec3b> it = im.begin<Vec3b>(); it != im.end<Vec3b>(); ++it) {
                    //                (*it)[0] = table[(*it)[0]];
                    //                (*it)[1] = table[(*it)[1]];
                    //                (*it)[2] = table[(*it)[2]];
#pragma omp task
                    if ((*it)[0] == 0 && (*it)[1] == 0 && (*it)[2] == 0) {
                        (*it)[0] = 255;
                        (*it)[1] = 255;
                        (*it)[2] = 255;
                    }

                }
            }
        }
    }

    return im;
}

/**
 *  Get integer coordinate ...
 * the purpose of this function is to be used with OpenCV drawcontours function as it produce an exception with double coordinates ..
 * @param vertices
 * @param shape
 * @param size only added for debugging purpose ...
 */
void Utils::get_int_coords(vector<Point>& vertices, string shape, int p_count, Size size) {
    int i = 0;
    //    Point2d points[1][p_count];
    //    Point2d pa[p_count];
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
        Point dP(x, y);
        //        cout<<"x "<<dP.x<<" y "<< dP.y<<endl;
        //        
        vertices.push_back(dP);
        //        points[0][i] = Point2d(x, y);
        //        pa[i] = Point2d(x, y);
        ++i;
    }
}

/**
 *  Get double coordinate ...
 * @param vertices
 * @param shape
 * @param size only added for debugging purpose ...
 */
void Utils::convert_double_coord(vector<Point2d>& vertices, string shape, int p_count, Size size) {
    int i = 0;
    //    Point2d points[1][p_count];
    //    Point2d pa[p_count];
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
        Point2d dP(x, y);
        //        cout<<"x "<<dP.x<<" y "<< dP.y<<endl;
        //        
        vertices.push_back(dP);
        //        points[0][i] = Point2d(x, y);
        //        pa[i] = Point2d(x, y);
        ++i;
    }
}

/**
 * Shift points to the cropped bounding box ....
 * 
 * @param vertices
 * @param tx bounding rectangle top-left corner x
 * @param ty bounding rectangle top-left corner y
 */
void Utils::shift_coordinates(vector<Point2d>& vertices, int tx, int ty) {
    for (vector<Point2d>::size_type i = 0; i < vertices.size(); i++) {
        vertices[i] = Point2d(vertices[i].x - tx, vertices[i].y - ty);
    }
}

/**
 * blur image ..
 * @param src
 * @param dest
 */
void Utils::imblur(Mat& src, Mat& dest) {
    //    blur(src, dest, Size(3, 3));
    //    GaussianBlur(cropped,contours)
    //    bilateralFilter(src, dest, 13, 255, 255);
    bilateralFilter(src, dest, 7, 150, 50);

    //    bilateralFilter(cropped, imask, 15, 255, 255);
    //    blur(cropped, contours, Size(9, 9));

    for (int i = 0; i < 5; i++)
        blur(dest, dest, Size(3, 3));
    //        medianBlur(dest, dest, 5);
}

/**
 * convert shape vector of points into a set of OpenCv lines 
 * @param vertices
 * @param lines
 */
void Utils::road_shape_lines(vector<Point2d>& vertices, vector<Vec4d>& lines) {
    lines.clear();
    for (vector<Point2d>::size_type i = 0; i != vertices.size() - 1; i++) {
        Point2d p1 = vertices[i];
        Point2d p2 = vertices[i + 1];
        lines.push_back(Vec4d(p1.x, p1.y, p2.x, p2.y));
    }
}

/**
 * Get the euclidean length of a line segment ..
 * @param l
 * @return 
 */
double Utils::euclidean(Vec4d& l) {
    return sqrt(pow(l[0] - l[2], 2) + pow(l[1] - l[3], 2));
}

/**
 * 
 * @param gps_line
 * @param detected_lines
 * @return shift vector ...
 */
double Utils::gps_line_shift(vector<Vec4d>& gps_line, vector<Vec4d>& detected_lines, vector<double>& shift) {
#if 1
    /*
     * right and left side segments length ..
     */
    double rl = 0, ll = 0;
    /*
     * right and left average shift ..
     */
    double rs = 0, ls = 0;
    /*
     * count segments on the left and right ..
     */
    int rc = 0, lc = 0;

    vector<double> lshift;
    vector<double> rshift;


    for (vector<Vec4d>::size_type j = 0; j != detected_lines.size(); j++) {
        int index;
        double min_dist = 100000000.0;
        Vec4d l2 = detected_lines[j];

        for (vector<Vec4d>::size_type i = 0; i != gps_line.size(); i++) {
            Vec4d l1 = gps_line[i];
            double dist = getShortestDistance(l1, l2);
            if (abs(dist) < abs(min_dist)) {
                min_dist = dist;
                index = i;
            }
        }
        //        cout << "min distance:\t" << j << " : " << min_dist << endl;
        shift.push_back(min_dist);
        double e = euclidean(l2);
        if (shift[j] < 0) {
            rs += shift[j];
            rc++;
            rl += e;

            rshift.push_back(abs(min_dist));
        } else {
            ls += shift[j];
            lc++;
            ll += e;
            lshift.push_back(abs(min_dist));
        }

        //        shift[j] = min_dist;
        //        shift.push_back(min_dist);
    }
    // average ..
    //    if (rl > ll)return rs / rc;
    //    return ls / lc;

    // return average for the smallest k elements ...
    int sgn_v;
    double val;
    if (rl > ll) {
        val = avg_k_smallest(rshift);
        sgn_v = sgn(rs);
    } else {
        val = avg_k_smallest(rshift);
        sgn_v = sgn(ls);
    }

    return sgn_v*val;
#else

    int clusters = 3;
    vector<double > fshift(detected_lines.size());
    vector<double > length(detected_lines.size());
    for (vector<Vec4d>::size_type j = 0; j != detected_lines.size(); j++) {
        int index;
        double min_dist = 100000000.0;
        Vec4d l2 = detected_lines[j];

        for (vector<Vec4d>::size_type i = 0; i != gps_line.size(); i++) {
            Vec4d l1 = gps_line[i];
            double dist = getShortestDistance(l1, l2);
            if (abs(dist) < abs(min_dist)) {
                min_dist = dist;
                index = i;
            }
        }

        fshift[j] = min_dist;
        length[j] = euclidean(l2);
    }

    vector<unsigned long> assignments(detected_lines.size());
    if (fshift.size() < clusters) return 0;
    kmean(fshift, clusters, assignments);

    double max = -1;
    double favg = 0;
    int index = -1;
    for (int i = 0; i < clusters; i++) {
        double sum = 0;
        double avg = 0;
        int count = 0;
        for (int j = 0; j < assignments.size(); j++) {
            if (assignments[j] == i) {
                sum += length[j];
                avg += fshift[i];
                count++;
            }
        }
        if (max < sum) {
            favg = avg / count;
            max = sum;
            index = i;
        }
    }
    return favg;
#endif
}

void Utils::kmean(vector<double>& shift, int clusters, vector<unsigned long>& assignments) {
    typedef dlib::matrix<double, 1, 1> sample_type;
    // Now we are making a typedef for the kind of kernel we want to use.  I picked the
    // radial basis kernel because it only has one parameter and generally gives good
    // results without much fiddling.
    typedef dlib::radial_basis_kernel<sample_type> kernel_type;

    // Here we declare an instance of the kcentroid object.  It is the object used to 
    // represent each of the centers used for clustering.  The kcentroid has 3 parameters 
    // you need to set.  The first argument to the constructor is the kernel we wish to 
    // use.  The second is a parameter that determines the numerical accuracy with which 
    // the object will perform part of the learning algorithm.  Generally, smaller values 
    // give better results but cause the algorithm to attempt to use more dictionary vectors 
    // (and thus run slower and use more memory).  The third argument, however, is the 
    // maximum number of dictionary vectors a kcentroid is allowed to use.  So you can use
    // it to control the runtime complexity.  
    dlib::kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8);

    // Now we make an instance of the kkmeans object and tell it to use kcentroid objects
    // that are configured with the parameters from the kc object we defined above.
    dlib::kkmeans<kernel_type> test(kc);
    int num_clusters = clusters;
    vector<sample_type> samples(shift.size());
    vector<sample_type> initial_centers;

    for (int i = 0; i < shift.size(); i++) {
        sample_type m;
        m(0) = shift[i];
        samples[i] = m;
    }

    // tell the kkmeans object we made that we want to run k-means with k set to 3. 
    // (i.e. we want 3 clusters)
    test.set_number_of_centers(num_clusters);

    // You need to pick some initial centers for the k-means algorithm.  So here
    // we will use the dlib::pick_initial_centers() function which tries to find
    // n points that are far apart (basically).  
    dlib::pick_initial_centers(num_clusters, initial_centers, samples, test.get_kernel());

    // now run the k-means algorithm on our set of samples.  
    test.train(samples, initial_centers);
    // Finally, we can also solve the same kind of non-linear clustering problem with
    // spectral_cluster().  The output is a vector that indicates which cluster each sample
    // belongs to.  Just like with kkmeans, it assigns each point to the correct cluster.

    assignments = dlib::spectral_cluster(kernel_type(0.1), samples, num_clusters);

}

double Utils::avg_k_smallest(vector<double>& elements) {
    double sum = 0;
    int count = 0;
    nth_element(elements.begin(), elements.begin() + elements.size() / 2, elements.end());
    int strt = 0;
    int half = elements.size() / 2;
    int end = elements.size();
    if (elements.size() > 5) {
        strt = half / 3;
        end = elements.size() - 2 * strt;
    }

    for (int i = strt; i < end; i++) {
        sum += elements[i];
        count++;
    }
    return sum / count;
}

/**
 * plot line segments from a list of points....
 * @param im
 * @param vertices
 * @param color
 * @param thickness
 */
void Utils::plot_line(Mat& im, vector<Point2d>& vertices, Scalar color, int thickness) {
    if (vertices.size() == 1) return;
    for (vector<Point2d>::size_type i = 0; i < vertices.size() - 1; i++) {
        Point2d p1 = vertices[i];
        Point2d p2 = vertices[i + 1];
        line(im, p1, p2, color, thickness);
    }
}

/**
 * keep only correlated lines, which mean that line with angle difference less than 30
 * 
 * @param _line1
 * @param _line2
 * @return 
 */
bool Utils::isEqual(const Vec4d& _l1, const Vec4d& _l2) {
    Vec4d l1(_l1), l2(_l2);

    double length1 = sqrt((l1[2] - l1[0])*(l1[2] - l1[0]) + (l1[3] - l1[1])*(l1[3] - l1[1]));
    double length2 = sqrt((l2[2] - l2[0])*(l2[2] - l2[0]) + (l2[3] - l2[1])*(l2[3] - l2[1]));

    double product = (l1[2] - l1[0])*(l2[2] - l2[0]) + (l1[3] - l1[1])*(l2[3] - l2[1]);

    //    float angle = acos(fabs(product / (length1 * length2)))* 180.0 / CV_PI;
    ////    cout << "Angle:\t" << cos(CV_PI / 30) << endl;
    //
    //    if (angle < 45)return false;
    if (fabs(product / (length1 * length2)) < cos(CV_PI / 10))
        return false;

    //    float mx1 = (l1[0] + l1[2]) * 0.5f;
    //    float mx2 = (l2[0] + l2[2]) * 0.5f;
    //
    //    float my1 = (l1[1] + l1[3]) * 0.5f;
    //    float my2 = (l2[1] + l2[3]) * 0.5f;
    //    float dist = sqrtf((mx1 - mx2)*(mx1 - mx2) + (my1 - my2)*(my1 - my2));
    //
    //    if (dist > std::max(length1, length2) * 0.5f)
    //        return false;

    return true;
}

bool Utils::isCorrelated(const vector<Vec4d>& gps_lines, const Vec4d& l) {
    for (vector<Vec4d>::const_iterator it = gps_lines.begin(); it != gps_lines.end(); it++)
        if (isEqual(*it, l))
            return true;
    return false;
}

/**
 * 
 * @param gps_lines
 * @param detected_lines
 * @param grouped_lines
 */
void Utils::partition(vector<Vec4d>& gps_lines, vector<Vec4d>& detected_lines, vector<Vec4d>& grouped_lines) {
    for (vector<Vec4d>::const_iterator it = detected_lines.begin(); it != detected_lines.end(); it++) {
        if (isCorrelated(gps_lines, *it))grouped_lines.push_back(*it);
    }

}

/**
 * 
 * @param src
 * @param lvs
 * @param bounding_box_point
 * @param tolerance
 * @return 
 */
Mat Utils::crop(Mat& src, vector<Point2d>& lvs, Point& bounding_box_point, double tolerance) {
    double start = double(getTickCount());
    Size size = src.size.operator()();
    Mat mask = Mat::zeros(size, src.type());

    //image bounding rectangle  ..
    Rect roi(0, 0, size.width, size.height);
    vector< vector<Point> > co_ordinates;
    vector<Vec4d> gps_lines;
    road_shape_lines(lvs, gps_lines);
    cout << "start free shape .." << endl;
    //    bounding_free_shape(gps_lines,tmp_verts,tolerance);
    vector<Point> vertices;
    bounding_free_shape(gps_lines, vertices, roi, tolerance);
    cout << "finish free shape .." << endl;
    //    convert_arr_vec(tmp_verts,p_count, vertices);

    //    get_int_coords(vertices, shape, p_count, size);
    //    cout << "No of vertices:\t" << vertices.size() << endl;

    co_ordinates.push_back(vertices);

    //    vector<vector<Point2d> >hull(co_ordinates.size());
    //    convexHull(Mat(co_ordinates[0]), hull[0], false);
    //    for (int i = 0; i < co_ordinates.size(); i++) {
    //        convexHull(Mat(co_ordinates[i]), hull[i], false);
    //    }

    drawContours(mask, co_ordinates, 0, Scalar(255, 255, 255), CV_FILLED, 8);
    //    vector<Point2d> t_coords = co_ordinates[0];
    //    Point* coords = &t_coords[0];
    //    fillPoly( mask, ppt, npt, 1, Scalar( 255, 255, 255 ), 8 );

    //    fillPoly(mask, pa, p_count, Scalar(255, 255, 255));
    //    string output = path + "/" + road + "-1-poly" + ext;
    //    imwrite(output, mask);

    //    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    //    imshow("Display window", mask); // Show our image inside it.
    //    waitKey(0); // Wait for a keystroke in the window

    //    Mat wmask(src.size(), src.type(), Scalar(255));


    //    for (vector<Point2d>::iterator it = vertices.begin(); it != vertices.end(); ++it) {
    //        if (!(irect.contains(*it) && src.at<uchar>(it->x, it->y) == 0)) {
    //            cout << it->x << "," << it->y << endl;
    //        }
    //    }
    Rect rect = boundingRect(co_ordinates[0]);
    bounding_box_point.x = rect.x;
    bounding_box_point.y = rect.y;

    Mat res;
    bitwise_and(src, mask, res);

    //
    //    cv::Mat locations; // output, locations of non-zero pixels 
    //    cv::findNonZero(res, locations);

    //    Mat res(res_1.size(), res_1.type(), Scalar(255));
    //    inRange(res_1, Scalar(0), Scalar(0), res);
    //    bitwise_not(res, res);
    //


    //    rectangle(res, rect, Scalar(0, 0, 255), 5, LINE_8, 0);
    //    exit(0);

    Mat cropped(rect.width, rect.height, src.type());

    res(rect).copyTo(cropped);
    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    //    clock_t end = clock();
    //    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //    cout << "Elapsed time \t" << duration_ms << "ms" << endl;
    return cropped;
}

/**
 * Calculate  the road offset by thinning binary image of the road shape and find the shift from the GPS road locations ....
 * 
 * @param src
 * @param lvs
 * 
 * @param path  for debug
 * @param write flag for debug 
 * @return offset value ...
 */
int Utils::road_offset(Mat& src, vector<Point2d>& lvs, string path, bool write) {
#if 0
    scanImageAndInvertIterator(src);
    Size size = src.size.operator()();

    //    int tx = bounding_box_point.x;
    //    int ty = bounding_box_point.y;

    //    invert(cropped);

    //    string output1 = path + "/colored" + ext;
    //    imwrite(output1, src);

    Mat imask(size.width, size.height, src.type());
    imblur(src, imask);
    // reduce the intensity of the of the figure to reduce the effects of the shadow.
    //    imask.convertTo(imask, -1, 1, -70);
    //    Canny(cropped, contours,200,350);
    cvtColor(imask, imask, CV_RGB2GRAY);

    //    convert image into binary .....
    threshold(imask, imask, thr, 255, THRESH_BINARY);

    iterative_imclose(imask, 0, 4);
    //    string binary_output = path + "/binary" + ext;
    //    imwrite(binary_output, imask);
    cout << "finish writing binary image .." << endl;
    //    adaptiveThreshold(imask, imask,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);

    CBlobResult blobs;
    //    cropped.setTo(Vec3b(0, 0, 0));
    //    time = getTickCount();
    //    IplImage temp = (IplImage) imask;
    //    blobs = CBlobResult(&temp, NULL, 1);
    //    cout << "found: " << blobs.GetNumBlobs() << endl;
    //    //    cout << "Tempo ST: " << (getTickCount() - time) / getTickFrequency() << endl;
    //    for (int i = 0; i < blobs.GetNumBlobs(); i++) {
    //        cout<<"In the first blob filling ..."<< endl;
    //        blobs.GetBlob(i)->FillBlob(cropped, CV_RGB(255, 255, 255));
    //    }
    //    string blob_output = path + "/" + road + "-" + boost::lexical_cast<string>(thr) + "-blob-1" + ext;
    //    imwrite(blob_output, cropped);
    //    cout << "finish writing binary image .." << endl;

    //    displayOverlay("Blobs Image", "Single Thread");
    //    imshow("Blobs Image", cropped);
    //    waitKey();

    //    time = getTickCount();
    blobs = CBlobResult(imask, Mat(), NUMCORES);
    //    cout << "Tempo MT: " << (getTickCount() - time) / getTickFrequency() << endl;
    CBlob *curblob;
    cout << "found: " << blobs.GetNumBlobs() << endl;
    stringstream s;
    int numBlobs = blobs.GetNumBlobs();
    //    cropped.setTo(Vec3b(0, 0, 0));
    for (int i = 0; i < numBlobs; i++) {
        //        Scalar mean, stddev;
        //        Vec3b color = Vec3b(255, 255, 255);
        curblob = blobs.GetBlob(i);

        if (curblob->Area(PIXELWISE) > 300) {
            //            cout << "Blob area PIXELWISE:\t" << curblob->Area(PIXELWISE) << "\tGreen:\t" << curblob->Area(GREEN) << "\tJoined blobs:\t" << curblob->getNumJoinedBlobs() << endl;
            curblob->FillBlob(imask, Scalar(255));
        }
        //cout <<"Blob "<<i<<": "<< curblob->GetID()<<endl;
        //curblob->FillBlob(cropped, color);
        // 		CvSeqReader reader;
        // 		Point2d pt;
        // 		if(curblob->GetExternalContour()->GetContourPoints()){
        // 			cvStartReadSeq(curblob->GetExternalContour()->GetContourPoints(),&reader);
        // 			for(int j=0;j<curblob->GetExternalContour()->GetContourPoints()->total;j++){
        // 				CV_READ_SEQ_ELEM(pt,reader);
        // 				color_img.at<Vec3b>(pt) = color;		
        // 			}
        // 		}
        s << i;
        //        putText(cropped, s.str(), curblob->getCenter(), 1, imask.size().width / 400, CV_RGB(200, 200, 200), 3);
        //        s.str("");
        // 		displayOverlay("Blobs Image","Press a key to show the next blob",500);
        // 		imshow("Blobs Image",color_img);
        // 		waitKey();
    }

    //    rectangle(cropped, blobs.getBlobNearestTo(Point2d(cropped.size().width / 2, cropped.size().height / 2))->GetBoundingBox(), CV_RGB(200, 100, 50), 1);
    //    string blob_output = path + "/blob" + ext;
    //    imwrite(blob_output, imask);
    //    cout << "finish writing binary image .." << endl;
    //    displayOverlay("Blobs Image", "Multi Thread");
    //    imshow("Blobs Image", cropped);
    //    waitKey();


    //    cout << "Remove small blobs" << endl;
    //    removeSmallBlobs(imask, 200);
    //    //        threshSegments(imask, 250);
    //    //    cout << "end removing small blobs" << endl;
    //    //    Mat sel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    //    //    erosion(imask, 2, 10);
    //    //    dilation(imask, 0, 15);
    //    //    
    //            erosion(imask,0, 7);
    //    //    imopen(imask, 0, 7);
    //    //    dilation(imask, 0, 3);
    imopen(imask, 0, 7);
    //    imclose(imask, 0, 15);

    //    invert the binary image....
    bitwise_not(imask, imask);
    //    imclose(imask, 0, 15);
    iterative_imclose(imask, 0, 21);
    //    string morph_output = path + "/morph" + ext;
    //    imwrite(morph_output, imask);
    cout << "finish writing morphological results..." << endl;
    //    cout << "Remove small blobs" << endl;
    //    removeSmallBlobs(imask, 350);
    //

#endif
    Mat imask;
    cout << "Thinning ..." << endl;
#if 1
    thinning(src, imask);
#else
    skeletonizing(src, imask);
#endif

    if (write) imwrite(path + "/thinning.png", imask);

#if 0
    cout << "Apply canny" << endl;
    Canny(imask, imask, 50, 200, 3); // Apply canny edge
#endif
    // Create and LSD detector with standard or no refinement.
    cout << "Detect lines.." << endl;
#if 1
    Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
#else
    Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
#endif
    // Detect the lines
    vector<Vec4d> lines;
    vector<Vec4d> detected_lines;
    ls->detect(imask, lines);
#if 0
    cout << "Number of detected lines:\t" << lines.size() << endl;

    //    for (vector<Vec4d>::size_type i = 0; i != lines.size(); i++) {
    //        Vec4d l(lines[i]);
    //        //          x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    //        line(imask, Point2d(l[0], l[1]), Point2d(l[2], l[3]), Scalar(255), 8);
    //    }
    vector<int> labels;
    //    bool Utils::isEqual(const Vec4d& _l1, const Vec4d& _l2)
    //    bool(*equal)(const Vec4d& _l1, const Vec4d& _l2);
    //    equal=&isEqual;
    cv::partition(lines, labels, Utils::isEqual);

    //    cout << "number of lines groups:\t" << numberOfLines << endl;
    //    cout << "Printing labels ..." << endl;
    vector<Vec4d> detected_lines;
    vector<Vec4d> detected_lines_group_1;
    for (vector<int>::size_type i = 0; i != labels.size(); i++) {
        //        cout << i << " : " << labels[i] << endl;
        if (labels[i] == 0) {
            detected_lines.push_back(lines[i]);
        } else {
            detected_lines_group_1.push_back(lines[i]);
        }
    }
#endif
    vector<Vec4d> gps_lines;
    road_shape_lines(lvs, gps_lines);
    partition(gps_lines, lines, detected_lines);

    //    cout << "Number of line in Group 1:\t" << detected_lines_group_0.size() << endl;
    //    imask.setTo(Vec3b(0, 0, 0));
    // Draw lines ...
    //    cout << "plotting lines .." << endl;
    //    ls->drawSegments(imask, glines);
    for (vector<Vec4d>::size_type i = 0; i != detected_lines.size(); i++) {
        Vec4d l(detected_lines[i]);
        //          x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
        line(imask, Point2d(l[0], l[1]), Point2d(l[2], l[3]), Scalar(255), 8);
    }
    if (write) imwrite(path + "/detected_lines.png", imask);

    //    CvMat t_srd = imask;
    //    CvMat t_dest;
    //    MorphologicalThinning(&t_srd, &t_srd);
    //    // display difference
    //    cvNamedWindow("video", 1);
    //    cvShowImage("video", &t_srd);
    //    Mat thin = cvarrToMat(&t_srd);
    //    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    //    imshow("Display window", thin); // Show our image inside it.
    //    cvReleaseImage(*cvimask);
    //    cout << "End thinning " << endl;

    // shift coordinates to fit the cropped image..
    //    shift_coordinates(lvs, tx, ty);
    //    plot_line(imask, lvs, Scalar(255, 255, 255), 8);
    //re-populate GPS lines with the shifted coordinates .. 


    vector<double> shift;
    // shift value and sign represent the direction of the shift from the GPS lines ..
    return gps_line_shift(gps_lines, detected_lines, shift);
}

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void Utils::CannyThreshold(Mat& src, Mat& dst, int threshold, int kernel_size) {
    int ratio = 3;
    Mat detected_edges;

    /// Canny detector
    Canny(src, detected_edges, threshold, threshold*ratio, kernel_size);

    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    src.copyTo(dst, detected_edges);

    //    imshow(window_name, dst);
}

/**
 * Detect and fill convex contain blobs ....
 * 
 * @param src
 * @param dst
 * @param NUMCORES
 */
void Utils::fillBlobs(Mat& src, Mat& dst, bool blob, int NUMCORES) {
    CBlobResult blobs;

    blobs = CBlobResult(src, Mat(), NUMCORES);
    //    cout << "Tempo MT: " << (getTickCount() - time) / getTickFrequency() << endl;
    CBlob *curblob;

    int numBlobs = blobs.GetNumBlobs();
    cout << "found: " << numBlobs << endl;
    for (int i = 0; i < numBlobs; i++) {
        curblob = blobs.GetBlob(i);
        //        if (curblob->Area(PIXELWISE) > 3000) {
        if (blob) {
            curblob->FillBlob(dst, Scalar(255));
//            drawContours(dst, curblob->GetExternalContour()->GetContours(), -1, Scalar(255, 255, 255), CV_FILLED, 4);
        } else {
            t_contours hull;
            curblob->GetConvexHull(hull);
            //        if (curblob->Area(PIXELWISE) > 400)
            drawContours(dst, hull, -1, Scalar(255, 255, 255), CV_FILLED, 4);
        }
        //        }
        //        


    }
}