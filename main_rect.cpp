/* 
 * File:   main_rect.cpp
 * Author: essam
 *
 * Created on August 10, 2015, 8:48 AM
 */

#include <cstdlib>
#include<opencv/cv.h>
#include<iostream>
using namespace std;
using namespace cv;

int isPointInBoundingBox(float x1, float y1, float x2, float y2, float px, float py) {
    float left, top, right, bottom; // Bounding Box For Line Segment
    // For Bounding Box
    if (x1 < x2) {
        left = x1;
        right = x2;
    } else {
        left = x2;
        right = x1;
    }
    if (y1 < y2) {
        top = y1;
        bottom = y2;
    } else {
        top = y1;
        bottom = y2;
    }

    if ((px + 0.01) >= left && (px - 0.01) <= right &&
            (py + 0.01) >= top && (py - 0.01) <= bottom) {
        return 1;
    } else
        return 0;
}

int LineIntersection(float l1x1, float l1y1, float l1x2, float l1y2,
        float l2x1, float l2y1, float l2x2, float l2y2,
        float *m1, float *c1, float *m2, float *c2,
        float* intersection_X, float* intersection_Y) {
    float dx, dy;

    dx = l1x2 - l1x1;
    dy = l1y2 - l1y1;

    *m1 = dy / dx;
    // y = mx + c
    // intercept c = y - mx
    *c1 = l1y1 - *m1 * l1x1; // which is same as y2 - slope * x2

    dx = l2x2 - l2x1;
    dy = l2y2 - l2y1;

    *m2 = dy / dx;
    // y = mx + c
    // intercept c = y - mx
    *c2 = l2y1 - *m2 * l2x1; // which is same as y2 - slope * x2

    if ((*m1 - *m2) == 0)
        return 0;
    else {
        *intersection_X = (*c2 - *c1) / (*m1 - *m2);
        *intersection_Y = *m1 * *intersection_X + *c1;
    }
}

bool segmentsIntersection(Point a1, Point a2, Point b1, Point b2, Point& pnt) {

    float l1x1 = a1.x;
    float l1y1 = a1.y;
    float l1x2 = a2.x;
    float l1y2 = a2.y;

    float l2x1 = b1.x;
    float l2y1 = b1.y;
    float l2x2 = b2.x;
    float l2y2 = b2.y;

    float px, py;
    float m1, c1, m2, c2;
    float dx1, dx2, dy1, dy2;

    dx1 = l1x2 - l1x1;
    dx2 = l2x2 - l2x1;
    dy1 = l1y2 - l1y1;
    dy2 = l2y2 - l2y1;

    if (dx1 == 0) {
        // Line 1 is vertical ....
        m2 = dy2 / dx2;
        // intercept c = y - mx
        c2 = l2y1 - m2 * l2x1; // which is same as y2 - slope * x2
        pnt.x = l1x1;
        pnt.y = m2 * pnt.x + c2;

        //check if the obtained point lie on the line segment not the extension of the line ..
        if ((pnt.y < l2y1 && pnt.y > l2y2) || (pnt.y > l2y1 && pnt.y < l2y2)) {
            return true;
        }
    } else if (dx2 == 0) {
        // Line 2 is vertical ....
        m1 = dy1 / dx1;
        // y = mx + c
        // intercept c = y - mx
        c1 = l1y1 - m1 * l1x1; // which is same as y2 - slope * x2
        pnt.x = l2x1;
        pnt.y = m1 * pnt.x + c1;
        //check if the obtained point lie on the line segment not the extension of the line ..
        if ((pnt.y < l1y1 && pnt.y > l1y2) || (pnt.y > l1y1 && pnt.y < l1y2)) {
            return true;
        }

    } else {

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
            py = m1 * px + c1;
        }
        cout << "px: " << px << " py: " << py << endl;

        if (((py - m1 * px - c1) == 0) &&((py - m2 * px - c2) == 0)) {
            pnt.x = px;
            pnt.y = py;
            return true;
        };
    }
    return false;
}

/*
 * 
 */
int main(int argc, char** argv) {
    //    int w = 30;
    //    int h = 25;
    //    Rect roi(0, 0, w, h);

    Point a1(4, 2);
    Point a2(-1, -3);
    Point b1(0, 0);
    Point b2(10, 0);
    Point r;

    bool intersect = segmentsIntersection(a1, a2, b1, b2, r);
    cout << intersect << endl;
    cout << r.x << " : " << r.y << endl;
    //    cout << roi.x << " : " << roi.y;
    return 0;
}

