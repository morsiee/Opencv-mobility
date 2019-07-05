
/* 
 * Real-time traffic snapshot estimation using single pass satellite imagery and disparity maps.
 * 
 * Author: essam
 *
 * Created on July 13, 2015, 4:31 PM
 * g++ -O2 -std=c++11 -fopenmp main_blobs.cpp -o blobs.out `pkg-config --libs --cflags opencv` -lopencvblobslib -lboost_filesystem -lboost_system
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
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;

/**
 * Use the sign of the determinant of vectors (AB,AM), where M(X,Y) is the query point
 * Position = sign( (Bx-Ax)*(Y-Ay) - (By-Ay)*(X-Ax) )
 * It is 0 on the line, and +1 on one side, -1 on the other side.
 * @param l
 * @param p
 * @return 
 */
template <typename T> int sgn(T val) {
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
int position(Vec4d& l1, Vec4d& l2) {


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
int position(Vec4d& l1, Point2d& M) {


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
double scaleX(double width, double x, double xmin, double xmax) {
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
double scaleY(double height, double y, double ymin, double ymax) {
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

double getShortestDistance(Vec4d& line1, Vec4d& line2) {


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
Vec4d solve_circle_Perpendicular_line(double x, double y, double m, double r) {
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
void shift_line(Vec4d& ol, Vec4d& nl, double sv, int dir) {
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
void shift_road_shape(vector<Point2d>& lvs, vector<Point2d>& nlvs, double sv, int dir) {

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
void bounding_box(Vec4d& ol, Point2d *vertices, double sv) {
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
double cross(Point v1, Point v2) {
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
bool intersectionPoint(Point a1, Point a2, Point b1, Point b2, Point & intPnt) {
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

bool isPointInBoundingBox(double x1, double y1, double x2, double y2, double px, double py) {

    // For Bounding Box
    double x, y;
    x = x2;
    y = y2;
    if (x1 < x2) x = x1;
    if (y1 < y2) y = y1;
    Rect2d roi(x, y, abs(x2 - x1), abs(y2 - y1));
    Point2d p(px, py);
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
bool segmentsIntersection(Point a1, Point a2, Point b1, Point b2, Rect& roi, Point& pnt) {

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

/**
 * Find bounding free shape poly-lines ..
 * 
 * @param lines
 * @param vertices
 * @param sv
 */
void bounding_free_shape(vector<Vec4d>& lines, vector<Point>& vertices, Rect& roi, double sv) {
    vertices.clear();
    const int count = 2 * lines.size() + 2;
    //    vector<Point> vertices[count];
    Point tv[count];
#pragma omp for
    for (vector<Vec4d>::size_type i = 0; i < lines.size(); i++) {
        Vec4d l = lines[i];
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

    //define ROI vertices ..
    Point lt(roi.x, roi.y + roi.height);
    Point lb(roi.x, roi.y);
    Point rb(roi.x + roi.width, roi.y);
    Point rt(roi.x + roi.width, roi.y + roi.height);

    //print free shape ..
    cout << "print free shape .." << endl;
#pragma omp for
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



            cout << "bounding box" << " ( " << roi.width << " , " << roi.height << " )" << endl;
            cout << "before" << " ( " << tv[i].x << " , " << tv[i].y << " )" << endl;

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
            } else {

                cout << "error: can't find the intersection point" << endl;
                if (tv[i].x < roi.x) {
                    tv[i].x = roi.x + 1;
                } else if (tv[i].x > roi.x + roi.width) {
                    tv[i].x = roi.x + roi.width - 1;
                }

                if (tv[i].y < roi.y) {
                    tv[i].y = roi.y + 1;
                } else if (tv[i].y > roi.y + roi.height) {
                    tv[i].y = roi.y + roi.height - 1;
                }
            }


        }
        //            vertices[i] = tv[i];
        //          
        vertices.push_back(tv[i]);
        cout << i << " ( " << vertices[i].x << " , " << vertices[i].y << " )" << endl;
    }
    //    return vertices;
}

void convert_arr_vec(Point2d* tmp_verts, int p_count, vector<Point>& vertices) {
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
Mat threshSegments(Mat &src, double threshSize) {
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
void removeSmallBlobs(Mat& im, double size) {
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
void erosion(Mat &src, int erosion_elem, int erosion_size) {
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
void dilation(Mat &src, int dilation_elem, int dilation_size) {
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
void imclose(Mat &src, int dilation_elem, int dilation_size) {
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
void iterative_imclose(Mat& im, int structure, int size) {
    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;
    int i = 0;
    do {
        //        cout << "imclose .." << endl;
        imclose(im, structure, size);
        imclose(im, structure, size);
        absdiff(im, prev, diff);
        im.copyTo(prev);
        i++;
    } while (countNonZero(diff) > 0 || i != 35);
}

/**
 * Morphological imopen ..
 * @param src
 * @param dilation_elem
 * @param dilation_size
 */
void imopen(Mat &src, int dilation_elem, int dilation_size) {
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

void thinningIteration(cv::Mat& img, int iter) {
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
 * Function for thinning the given binary image
 *
 * Parameters:
 * 		src  The source image, binary with range = [0,255]
 * 		dst  The destination image
 */
void thinning(const cv::Mat& src, cv::Mat& dst) {
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

Mat& scanImageAndInvertIterator(Mat& im) {
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
void get_int_coords(vector<Point>& vertices, string shape, int p_count, Size size) {
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
void convert_double_coord(vector<Point2d>& vertices, string shape, int p_count, Size size) {
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
void shift_coordinates(vector<Point2d>& vertices, int tx, int ty) {
    for (vector<Point2d>::size_type i = 0; i < vertices.size(); i++) {
        vertices[i] = Point2d(vertices[i].x - tx, vertices[i].y - ty);
    }
}

/**
 * blur image ..
 * @param src
 * @param dest
 */
void blur(Mat& src, Mat& dest) {
    blur(src, dest, Size(3, 3));
    //    GaussianBlur(cropped,contours)
    //    bilateralFilter(src, dest, 13, 255, 255);
    //    bilateralFilter(cropped, imask, 7, 150, 50);

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
void road_shape_lines(vector<Point2d>& vertices, vector<Vec4d>& lines) {
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
double euclidean(Vec4d& l) {
    return sqrt(pow(l[0] - l[2], 2) + pow(l[1] - l[3], 2));
}

/**
 * 
 * @param gps_line
 * @param detected_lines
 * @return shift vector ...
 */
double gps_line_shift(vector<Vec4d>& gps_line, vector<Vec4d>& detected_lines, vector<double>& shift) {

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
        } else {
            ls += shift[j];
            lc++;
            ll += e;
        }

        //        shift[j] = min_dist;
        //        shift.push_back(min_dist);
    }
    if (rl > ll)return rs / rc;
    return ls / lc;
}

/**
 * plot line segments from a list of points....
 * @param im
 * @param vertices
 * @param color
 * @param thickness
 */
void plot_line(Mat& im, vector<Point2d>& vertices, Scalar color, int thickness) {
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
bool isEqual(const Vec4d& _l1, const Vec4d& _l2) {
    Vec4d l1(_l1), l2(_l2);

    double length1 = sqrt((l1[2] - l1[0])*(l1[2] - l1[0]) + (l1[3] - l1[1])*(l1[3] - l1[1]));
    double length2 = sqrt((l2[2] - l2[0])*(l2[2] - l2[0]) + (l2[3] - l2[1])*(l2[3] - l2[1]));

    double product = (l1[2] - l1[0])*(l2[2] - l2[0]) + (l1[3] - l1[1])*(l2[3] - l2[1]);

    //    float angle = acos(fabs(product / (length1 * length2)))* 180.0 / CV_PI;
    ////    cout << "Angle:\t" << cos(CV_PI / 30) << endl;
    //
    //    if (angle < 45)return false;
    if (fabs(product / (length1 * length2)) < cos(CV_PI / 50))
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

/*
 * 
 */
int main(int argc, char** argv) {
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
    //    //Map parameters ..
    //    const double XMIN = 387968.529665;
    //    const double YMIN = 3949750.693277;
    //    const double XMAX = 389547.318664;
    //    const double YMAX = 3951528.063522;
    //    
    string road = "199311841";
    //make directory of all results of a single road ..
    path += "/" + road;
    boost::filesystem::path directory(path);

    if(!(boost::filesystem::exists(directory))){
        if (boost::filesystem::create_directory(directory))
            cout << "Successfully created !" << endl;
    }

    string ext = ".png";
    int p_count = 46;
    //    string shape = "11522.376776458877,15103.0 11504.024386622477,14814.41164474354 11453.009514084446,13908.672590695121 11428.16626305738,13632.04085876345 11401.277063364134,13423.481394582857 11362.426527386293,13273.55277796619 11325.002850915282,13112.498711141932 11245.715290737206,12812.511276097875 10983.078741533796,12015.46634859921 10727.286491824929,11237.855957630849 10394.624883254446,10214.95998875001 10319.585734242113,9980.173276851165 10256.157685045275,9773.534099229992 10194.15241390104,9577.744603558393 10152.019861420627,9432.608267846776 10117.81991483328,9267.767804205068 10102.79329545622,9126.15907161077 10106.375273310825,8965.502869270635 10119.32822946215,8787.84555341935 10124.661994113723,8720.88171051332 9547.71695404924,8672.353311594512 9542.382896712166,8739.317129886975 9529.121127508628,8921.03803688513 9524.102690737334,9111.177129870164 9541.49301880146,9323.507933370985 9584.69680385682,9550.847772041963 9638.326560398227,9743.929768555941 9705.575597538678,9957.231491801977 9767.515130792914,10159.088676732705 9844.637163644162,10400.52697332764 10178.259599784902,11426.40129878578 10434.6881981491,12205.954907715233 10697.421207845402,13003.294992541772 10766.077318513728,13267.316369304315 10799.27627753809,13411.253657829828 10841.798895340764,13576.174016317009 10853.280776287473,13701.753869292312 10876.566768653574,13963.127228131065 10926.110590216515,14848.670642634233 10944.475355134973,15103.0";
    string lshape = "11012.420218770456,0.0 10469.601741733322,534.8058002893616 10419.859920597526,1313.6852284171782 10382.853328812163,2760.7583113821865 10365.404410674739,3250.2401188158788 10332.077803995715,4403.41166862032 10183.348518353807,5650.315585363245 10101.148401630024,6570.813317677094 9955.10674555916,7712.215209432212 9853.390303335684,8490.499781198254 9836.18947408148,8696.617511053917 9830.855563087158,8763.581341653162 9817.748200410208,8943.270453075904 9813.447993096537,9118.668100740468 9829.65646681737,9295.637868790005 9868.358332638483,9491.728019946348 9916.239487148912,9660.837186059145 9980.86664129342,9865.382795514004 10043.550432517031,10069.630976793913 10119.63102344858,10307.743481038826 10452.773045804675,11332.128628208315 10708.88346984217,12110.710628157221 10971.568249292026,12907.903134321803 11045.540084713542,13189.907540223123 11080.851402461229,13342.403217898009 11121.537979351006,13499.827705447955 11140.723519672425,13666.89736402788 11164.788141369492,13935.899909413092 11215.067488419496,14831.541143690865 11233.426065796684,15103.0 11299.33501250603,15103.0 11337.375307971564,15103.0 11395.717543735183,15103.0 11483.292919600575,15103.0 11590.756754291215,15103.0 11796.753223867841,15103.0 11984.018982741725,15103.0 12090.862595223589,15103.0 12184.805585764918,15103.0 12267.62592469748,15103.0 12327.911523381516,15103.0 12358.1370190181,15103.0 12360.1217300858,15103.0 12326.629730816678,15103.0 12178.85145256182,15103.0 12010.853930303923,15103.0";
    //    RNG random;
    //    int64 time;
    Mat src = imread(argv[6], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    Size size = src.size.operator()();
    Mat mask = Mat::zeros(size, src.type());

    //image bounding rectangle  ..
    Rect roi(0, 0, size.width, size.height);
    //    cout << "Size width: " << size.width << " height: " << size.height << endl;
    //    Rect irect(Point2d(), size);


    //    clock_t begin = clock();
    double start = double(getTickCount());

    vector< vector<Point> > co_ordinates;
    vector<Point2d> lvs;
    convert_double_coord(lvs, lshape, p_count, size);
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
    int tx = rect.x;
    int ty = rect.y;

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
    Mat oc(rect.width, rect.height, src.type());

    res(rect).copyTo(cropped);
    // keep version of the originally cropped image ..
    res(rect).copyTo(oc);
    Size oc_size = oc.size.operator()();

    cout << "Scan and invert" << endl;
    scanImageAndInvertIterator(cropped);
    //    invert(cropped);

    string output1 = path + "/colored" + ext;
    imwrite(output1, cropped);

    Mat imask(rect.width, rect.height, src.type());
    blur(cropped, imask);
    // reduce the intensity of the of the figure to reduce the effects of the shadow.
    //    imask.convertTo(imask, -1, 1, -70);
    //    Canny(cropped, contours,200,350);
    cvtColor(imask, imask, CV_RGB2GRAY);

    //    convert image into binary .....
    threshold(imask, imask, thr, 255, THRESH_BINARY);
    iterative_imclose(imask, 0, 4);
    string binary_output = path + "/binary" + ext;
    imwrite(binary_output, imask);
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
        Vec3b color = Vec3b(255, 255, 255);
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
    string blob_output = path + "/blob" + ext;
    imwrite(blob_output, imask);
    cout << "finish writing binary image .." << endl;
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
    string morph_output = path + "/morph" + ext;
    imwrite(morph_output, imask);
    cout << "finish writing morphological results..." << endl;
    //    cout << "Remove small blobs" << endl;
    //    removeSmallBlobs(imask, 350);
    //
    cout << "Thinning ..." << endl;
    thinning(imask, imask);


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
    ls->detect(imask, lines);
    cout << "Number of detected lines:\t" << lines.size() << endl;

    //    for (vector<Vec4d>::size_type i = 0; i != lines.size(); i++) {
    //        Vec4d l(lines[i]);
    //        //          x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    //        line(imask, Point2d(l[0], l[1]), Point2d(l[2], l[3]), Scalar(255), 8);
    //    }

    std::vector<int> labels;
    int numberOfLines = cv::partition(lines, labels, isEqual);
    cout << "number of lines groups:\t" << numberOfLines << endl;

    cout << "Printing labels ..." << endl;
    vector<Vec4d> detected_lines_group;
    for (vector<int>::size_type i = 0; i != labels.size(); i++) {
        //        cout << i << " : " << labels[i] << endl;
        if (labels[i] == 0) detected_lines_group.push_back(lines[i]);
    }
    cout << "Number of line in Group 1:\t" << detected_lines_group.size() << endl;
    //    imask.setTo(Vec3b(0, 0, 0));
    // Draw lines ...
    cout << "plotting lines .." << endl;
    //    ls->drawSegments(imask, glines);
    for (vector<Vec4d>::size_type i = 0; i != detected_lines_group.size(); i++) {
        Vec4d l(detected_lines_group[i]);
        //          x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
        line(imask, Point2d(l[0], l[1]), Point2d(l[2], l[3]), Scalar(255), 8);
    }


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
    cout << "End thinning " << endl;

    // shift coordinates to fit the cropped image..
    shift_coordinates(lvs, tx, ty);
    plot_line(imask, lvs, Scalar(255, 255, 255), 8);
    //re-populate GPS lines with the shifted coordinates .. 
    //    vector<Vec4d> gps_lines;
    road_shape_lines(lvs, gps_lines);
    vector<double> shift;
    // shift value and sign represent the direction of the shift from the GPS lines ..
    double sv = gps_line_shift(gps_lines, detected_lines_group, shift);

    int dir = sgn(sv);
    sv = abs(sv);
    vector<Point2d> nlvs(lvs.size());
    shift_road_shape(lvs, nlvs, sv, dir);

    // crop shifted roads  ...
    gps_lines.clear();
    vertices.clear();

    Rect nroi(0, 0, oc_size.width, oc_size.height);
    tolerance = lanes * lane_width;
    road_shape_lines(nlvs, gps_lines);
    bounding_free_shape(gps_lines, vertices, nroi, tolerance);

    mask = Mat::zeros(oc_size, oc.type());
    co_ordinates.clear();
    co_ordinates.push_back(vertices);
    drawContours(mask, co_ordinates, 0, Scalar(255, 255, 255), CV_FILLED, 8);
    rect = boundingRect(co_ordinates[0]);
    Mat fres;
    bitwise_and(oc, mask, fres);
    //    cropped.release();
    Mat cfres(rect.width, rect.height, oc.type());
    fres(rect).copyTo(cfres);
    //    plot_line(cropped, lvs, Scalar(0, 0, 255), 8);
    //    plot_line(cropped, nlvs, Scalar(255, 0, 255), 8);

    cout << "Shift dir:\t" << sv << endl;
    // print shift values ...
    //    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    //    double fontScale = 4;
    //    int thickness = 5;
    //    for (vector<double>::size_type i = 0; i != shift.size(); i++) {
    //        cout << i << " : " << boost::lexical_cast<string>(shift[i]) << endl;
    //        //        Vec4d l = detected_lines_group[i];
    //        //        // center the text
    //        //        Point textOrg((l[0] + l[2]) / 2, (l[1], l[3]) / 2);
    //        //        putText(imask, boost::lexical_cast<std::string>(shift[i]), textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
    //    }

    //    vector<vector < Point> > contours;
    //    find the contours of the binary image...
    //    findContours(imask, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    //    drawContours(imask, contours, -1, Scalar(255), CV_FILLED, 4);

    //    for (vector< vector<Point2d> >::iterator it = contours.begin(); it != contours.end(); ++it) {
    //        Rect t_rect = boundingRect(*it);
    //        rectangle(imask, t_rect, Scalar(255), 2, 4, 0);
    //    }
    //    Mat res_ones;
    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    //    clock_t end = clock();
    //    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Elapsed time \t" << duration_ms << "ms" << endl;
    cout << "New size width: " << rect.width << " height: " << rect.height << endl;
    string output = path + "/cf_res" + ext;
    imwrite(output, cfres);
    waitKey(0);
    //    cvDestroyWindow("video");
    return 0;
}

