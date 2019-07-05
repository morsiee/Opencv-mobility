
/* 
 * File:   main_2.cpp
 * Author: essam
 *
 * Created on July 13, 2015, 4:31 PM
 */

#include <cstdlib>
#include<iostream>
//#include <cv.h>
//#include <opencv2/highgui/highgui.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <ctime>
#include<string>

using namespace cv;
using namespace std;
// Gets only the biggest segments

double XMIN = 387968.529665;
double YMIN = 3949750.693277;
double XMAX = 389547.318664;
double YMAX = 3951528.063522;

Mat threshSegments(Mat &src, double threshSize) {
    // FindContours:
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat srcBuffer, output;
    src.copyTo(srcBuffer);
    findContours(srcBuffer, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);

    vector<vector<Point> > allSegments;

    // For each segment:
    for (size_t i = 0; i < contours.size(); ++i) {
        drawContours(srcBuffer, contours, i, Scalar(200, 0, 0), 1, 8, hierarchy, 0, Point());
        Rect brect = boundingRect(contours[i]);
        rectangle(srcBuffer, brect, Scalar(255, 0, 0));

        int result;
        vector<Point> segment;
        for (unsigned int row = brect.y; row < brect.y + brect.height; ++row) {
            for (unsigned int col = brect.x; col < brect.x + brect.width; ++col) {
                result = pointPolygonTest(contours[i], Point(col, row), false);
                if (result == 1 || result == 0) {
                    segment.push_back(Point(col, row));
                }
            }
        }
        allSegments.push_back(segment);
    }

    output = Mat::zeros(src.size(), CV_8U);
    int totalSize = output.rows * output.cols;
    for (int segmentCount = 0; segmentCount < allSegments.size(); ++segmentCount) {
        vector<Point> segment = allSegments[segmentCount];
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
    std::vector<std::vector<Point> > contours;
    findContours(im.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        // Calculate contour area
        double area = contourArea(contours[i]);

        // Remove small objects by drawing the contour with black color
        if (area > 0 && area <= size)
            drawContours(im, contours, i, CV_RGB(0, 0, 0), -1);
    }
}

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
            Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
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
            Point(dilation_size, dilation_size));
    //    Apply the dilation operation
    dilate(src, src, element);
    //    imshow("Dilation Demo", dilation_dst);
}

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
            Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    morphologyEx(src, src, MORPH_CLOSE, element);
    //    imshow("Dilation Demo", dilation_dst);
}

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningIteration(Mat& im, int iter) {
    Mat marker = Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows - 1; i++) {
        for (int j = 1; j < im.cols - 1; j++) {
            uchar p2 = im.at<uchar>(i - 1, j);
            uchar p3 = im.at<uchar>(i - 1, j + 1);
            uchar p4 = im.at<uchar>(i, j + 1);
            uchar p5 = im.at<uchar>(i + 1, j + 1);
            uchar p6 = im.at<uchar>(i + 1, j);
            uchar p7 = im.at<uchar>(i + 1, j - 1);
            uchar p8 = im.at<uchar>(i, j - 1);
            uchar p9 = im.at<uchar>(i - 1, j - 1);

            int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i, j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinning(Mat& im) {
    im /= 255;

    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    } while (countNonZero(diff) > 0);

    im *= 255;
}

Mat& scanImageAndInvertIterator(Mat& im) {
    // accept only char type matrices
    CV_Assert(im.depth() != sizeof (uchar));

    const int channels = im.channels();
    cout << "channels \t" << channels << endl;

    switch (channels) {
        case 1:
        {

            MatIterator_<uchar> it, end;
            for (it = im.begin<uchar>(), end = im.end<uchar>(); it != end; ++it) {
                //                *it = table[*it];
                if (*it == 0) {
                    *it = 255;
                }


            }
            break;
        }
        case 3:
        {
            MatIterator_<Vec3b> it, end;
            for (it = im.begin<Vec3b>(), end = im.end<Vec3b>(); it != end; ++it) {
                //                (*it)[0] = table[(*it)[0]];
                //                (*it)[1] = table[(*it)[1]];
                //                (*it)[2] = table[(*it)[2]];

                if ((*it)[0] == 0 && (*it)[1] == 0 && (*it)[2] == 0) {
                    (*it)[0] = 255;
                    (*it)[1] = 255;
                    (*it)[2] = 255;
                }

            }
        }
    }

    return im;
}

void ReplaceStr(std::string& subject, const std::string& search,
        const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
}

double scaleX(double width, double x, double xmin, double xmax) {
    return width * (x - xmin) / (xmax - xmin);
}

double scaleY(double height, double y, double ymin, double ymax) {
    return height * (ymax - y) / (ymax - ymin);
}

void plot_line(Mat &im, vector<Point> &vertices) {
    for (std::vector<Point>::size_type i = 0; i != vertices.size() - 1; i++) {
        /* std::cout << someVector[i]; ... */
        Point p1 = vertices[i];
        Point p2 = vertices[i + 1];
        line(im, p1, p2, Scalar(255, 255, 255), LINE_8);
    }
}
//void invert(Mat& im) {
//    Size s = im.size.operator()();
//    for (int i = 0; i < im.rows; i++) {
//        for (int j = 0; j < im.cols; j++) {
//
//            if (im.at<uchar>(i, j) == 0) {
//                im.at<uchar>(i, j) = 255;
//            }
//        }
//    }
//}

/*
 * 
 */
int main(int argc, char** argv) {
    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    string path = argv[1];
    int thr = 60;
    //    string road = "w_z";
    string ext = ".bmp";
    //    string out = path.replace(path.find(ext),)
    //        int p_count = 36;
    string shape = "388967.38,3950114.37 388966.3,3950123.0199999996 388951.86,3950167.4800000004 388948.92,3950176.5300000003 388930.9,3950226.67 388929.69,3950230.41 388927.70499999996,3950236.5949999997 388925.015,3950245.09 388920.69,3950258.7350000003 388917.245,3950264.15 388911.53500000003,3950282.19 388909.66500000004,3950288.11 388903.185,3950308.64 388901.82999999996,3950318.23 388894.48,3950345.935 388889.345,3950356.3099999996";

    Mat cropped = imread(path, CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    scanImageAndInvertIterator(cropped);
    Size size = cropped.size.operator()();
    //    Mat mask = Mat::zeros(size, src.type());
    //
    //    cout << "Size width: " << size.width << " height: " << size.height << endl;
    //    Rect irect(Point(), size);
    //
    //    vector< vector<Point> > co_ordinates;
    vector<Point> vertices;
    //    //    Point2d p(388632.7862481127,3950313.751284572);
    clock_t begin = clock();
    int i = 0;
    stringstream ssin(shape);
    while (ssin.good()) {
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
        double x = scaleX(size.width, boost::lexical_cast<double>(xy[0]), XMAX, XMIN);
        double y = scaleY(size.height, boost::lexical_cast<double>(xy[1]), YMAX, YMIN);
        vertices.push_back(Point(x, y));
        ++i;
    }
    cout << "Vertices size\t" << vertices.size() << endl;

    //    plot_line(cropped, vertices);
    //    string tmp = path;
    //    ReplaceStr(tmp, ext, "-" + boost::lexical_cast<string>(thr) + "-thinning" + ext);
    //    imwrite(tmp, cropped);
    //
    //    //    vertices.push_back(P1);
    //    //    vertices.push_back(P2);
    //    //    vertices.push_back(P3);
    //    //    vertices.push_back(P4);
    //    //    vertices.push_back(P5);
    //    //    vertices.push_back(P6);
    //    co_ordinates.push_back(vertices);
    //    drawContours(mask, co_ordinates, 0, Scalar(255, 255, 255), CV_FILLED, 8);
    //    Mat wmask(src.size(), src.type(), Scalar(255));
    //
    //
    //    //    for (vector<Point>::iterator it = vertices.begin(); it != vertices.end(); ++it) {
    //    //        if (!(irect.contains(*it) && src.at<uchar>(it->x, it->y) == 0)) {
    //    //            cout << it->x << "," << it->y << endl;
    //    //        }
    //    //    }
    //    Rect rect = boundingRect(co_ordinates[0]);
    //
    //    Mat res;
    //    bitwise_and(src, mask, res);
    //    //
    //    //    cv::Mat locations; // output, locations of non-zero pixels 
    //    //    cv::findNonZero(res, locations);
    //
    //    //    Mat res(res_1.size(), res_1.type(), Scalar(255));
    //    //    inRange(res_1, Scalar(0), Scalar(0), res);
    //    //    bitwise_not(res, res);
    //    //
    //
    //
    //    //    rectangle(res, rect, Scalar(0, 0, 255), 5, LINE_8, 0);
    //
    //    Mat cropped(rect.width, rect.height, src.type());
    //    res(rect).copyTo(cropped);
    //invert the black area into wheat to be inverted into zero with image inversion.....
    //    scanImageAndInvertIterator(cropped);
    //    invert(cropped);
    Size s = cropped.size.operator()();
    //    string output1 = path + "/" + road + "-" + boost::lexical_cast<string>(thr) + "-out1" + ext;
    //    imwrite(output1, cropped);

    Mat imask(s.width, s.height, cropped.type());
    //    GaussianBlur(cropped,contours)
    bilateralFilter(cropped, imask, 13, 255, 150);
    //    bilateralFilter(cropped, imask, 7, 150, 50);

    //    bilateralFilter(cropped, imask, 15, 255, 255);
    //    blur(cropped, contours, Size(9, 9));

    //    medianBlur(imask, imask, 5);
    // reduce the intensity of the of the figure to reduce the effects of the shadow.
    imask.convertTo(imask, -1, 1, -70);
    //    Canny(cropped, contours,200,350);
    cvtColor(imask, imask, CV_RGB2GRAY);

    //    convert image into binary .....
    threshold(imask, imask, thr, 255, THRESH_BINARY);

    removeSmallBlobs(imask, 100);

    //    Mat sel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    //    erosion(imask, 2, 10);
    //    dilation(imask, 0, 15);

    imclose(imask, 0, 25);
    //    invert the binary image....
    bitwise_not(imask, imask);
    namedWindow("converted", 1);
    imshow("converted", imask);
    thinning(imask);

    plot_line(imask, vertices);
    //  recognize lines:
    //    vector<Vec4i> lines;
    //    Mat color_dst(imask.size.operator()(), imask.type());
    //    Canny(imask, imask, 50, 200, 3);
    //    HoughLinesP(imask, lines, 1, CV_PI / 180, 80, 30, 10);
    //    for (size_t i = 0; i < lines.size(); i++) {
    //        line(imask, Point(lines[i][0], lines[i][1]),
    //                Point(lines[i][2], lines[i][3]), Scalar(0, 255, 255), 3, 8);
    //    }

    //    vector<vector < Point> > contours;
    //    find the contours of the binary image...
    //    findContours(imask, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    //    drawContours(imask, contours, -1, Scalar(255), CV_FILLED, 4);

    //    for (vector< vector<Point> >::iterator it = contours.begin(); it != contours.end(); ++it) {
    //        Rect t_rect = boundingRect(*it);
    //        rectangle(imask, t_rect, Scalar(255), 2, 4, 0);
    //    }
    //    Mat res_ones;

    //    vector<vector<Point> > contours, out_contours;
    //    vector<Vec4i> hierarchy;
    //    findContours(imask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    //    cout << "number of lines:\t" << contours.size() << endl;
    //    out_contours.resize(contours.size());
    //    for (size_t k = 0; k < contours.size(); k++) {
    //        drawContours(imask, out_contours, k, Scalar(255-(k + 1)*7), CV_FILLED, 4);
    ////                approxPolyDP(Mat(contours[k]), out_contours[k], 3, false);
    //    }
    //    drawContours(imask, out_contours, 0, Scalar(255), CV_FILLED, 4);

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Elapsed time \t" << elapsed_secs << endl;
    cout << "New size width: " << s.width << " height: " << s.height << endl;
    //    string output = path + "/" + road + "-" + boost::lexical_cast<string>(thr) + "-binary" + ext;
    ReplaceStr(path, ext, "-" + boost::lexical_cast<string>(thr) + "-thinning" + ext);
    imwrite(path, imask);
    //    waitKey(0);
    return 0;
}

