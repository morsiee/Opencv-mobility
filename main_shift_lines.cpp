/* 
 * File:   main_shift_lines.cpp
 * Author: essam
 *
 * Created on August 8, 2015, 11:02 PM
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
#include<opencvblobslib/blob.h>
#include<opencvblobslib/BlobResult.h>
//#include <opencv2/core/core.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <vector>

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
 * find the vertices of the bounding box started from the upper left corner in counter clockwise order ..
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

void bounding_free_shape(vector<Vec4d>& lines, Point2d *vertices, double sv) {
    const int count = 2 * lines.size() + 2;
    for (vector<Vec4d>::size_type i = 0; i != lines.size(); i++) {
        Vec4d l = lines[i];
        Point2d vl[4];
        bounding_box(l, vl, sv);
        if (i != (lines.size() - 1)) {
            // add only the first two vertices to the polyline vertices ..
            vertices[i] = vl[0];
            vertices[(count - 1) - i] = vl[1];
        } else {
            // if it is the last line, add four vertices to the poly-line shape ..
            // the first two elements ..
            vertices[i] = vl[0];
            vertices[(count - 1) - i] = vl[1];
            // Add the last two points ..
            vertices[i + 1] = vl[3];
            vertices[(count - 1)-(i + 1)] = vl[2];

        }

    }
}

/*
 * 
 */
int main(int argc, char** argv) {
    // case 1
    //    Vec4d ol(11, 20, 15, 20);
    //case 2
    Vec4d ol(13, 21, 13, 16);
    //case 3
    Vec4d ol1(13, 16, 15, 20);
    Vec4d nl;
    double sv = 0.5;
    int dir = 1;
    shift_line(ol, nl, sv, dir);
    cout << "New coordinates:\t" << nl[0] << "," << nl[1] << "," << nl[2] << "," << nl[3] << endl;

    vector<Vec4d> lines;
    lines.push_back(ol);
    lines.push_back(ol1);
    const int size = 2 * lines.size() + 2;
    Point2d vertices[size];
    //    bounding_box(ol, vertices, sv);
    bounding_free_shape(lines, vertices, sv);
    for (int i = 0; i != size; i++) {
        Point2d p = vertices[i];
        cout << "( " << p.x << " , " << p.y << " )" << endl;
    }


    return 0;
}

