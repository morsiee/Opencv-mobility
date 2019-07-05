/* 
 * File:   main_lines.cpp
 * Author: essam
 *
 * Created on August 7, 2015, 6:45 AM
 */

#include <cstdlib>
#include <cmath>
#include<iostream>
#include <random>
#include <opencv/cv.h>
#include<opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
using namespace std;
using namespace cv;
//struct Point2d {
//    double x;
//    double y;
//    //    double z;
//
//    //    Point(double x1, double y1) : x(x1), y(y1) {
//    //    }
//};

//struct line2d {
//    double x1;
//    double y1;
//    //    double z1;
//    double x2;
//    double y2;
//    //    double z2;
//
//    line2d(double ix1, double iy1, double ix2, double iy2) : x1(ix1), y1(iy1), x2(ix2), y2(iy2) {
//    }
//};

//double dot(Point2d c1, Point2d c2) {
//    return (c1.x * c2.x + c1.y * c2.y /*+ c1.z * c2.z*/);
//}
//
//double norm(Point2d c1) {
//    return sqrt(dot(c1, c1));
//}

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

/*
 * 
 */
int main(int argc, char** argv) {
    Vec4d l1(1, 1, 3, 6);
    Vec4d l2(2, 2, 4.5, 5);


    cout << "Shortest distance:\t" << getShortestDistance(l1, l2) << endl;
    //    cout << "Position:\t" << positions(l1, l2) << endl;
    //
    //    cout << "swap lines ..." << endl;
    //    cout << "Position:\t" << positions(l2, l1) << endl;
    return 0;
}

