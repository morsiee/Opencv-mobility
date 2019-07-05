
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
#include <omp.h>

using namespace cv;
using namespace std;

// Gets only the biggest segments

Mat threshSegments(Mat &src, double threshSize) {
    // FindContours:
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat srcBuffer, output;
    src.copyTo(srcBuffer);
    findContours(srcBuffer, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS);

    vector<vector<Point> > allSegments;

    // For each segment:
#pragma omp for
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
#pragma omp for
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
    cout<< "Contours size from remove small blobs:\t"<<contours.size()<<endl;
#pragma omp for
    for (int i = 0; i < contours.size(); i++) {
        // Calculate contour area
        double area = contourArea(contours[i]);

        // Remove small objects by drawing the contour with black color
        if (area > 0 && area <= size){
            cout<< "eliminate small blob ..."<< endl;
            drawContours(im, contours, i, CV_RGB(0, 0, 0), -1);
            
        }
    }
    std::vector<std::vector<Point> > contours1;
    findContours(im.clone(), contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cout<< "Contours size after remove small blobs:\t"<<contours1.size()<<endl;
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


    morphologyEx(src, src, MORPH_CLOSE, element);
    //    imshow("Dilation Demo", dilation_dst);
}

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
            Point(dilation_size, dilation_size));
    /// Apply the open followed by close to remove small elements and fill connected buildings....
    morphologyEx(src, src, MORPH_OPEN, element);

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
    //#pragma omp for
#pragma omp parallel for schedule(dynamic,1) collapse(2)
    for (int i = 1; i < im.rows - 1; i++) {
        //        #pragma omp for 
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
    clock_t begin = clock();
    do {
        //#pragma omp parallel
        //        {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        //        }
        absdiff(im, prev, diff);
        im.copyTo(prev);
    } while (countNonZero(diff) > 0);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Thinning elapsed time \t" << elapsed_secs << endl;
    im *= 255;
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
 * 
 * @param vertices
 * @param shape
 * @param size only added for debugging purpose ...
 */
void get_coordinates(vector<Point>& vertices, string shape, int p_count, Size size) {
    int i = 0;
    //    Point points[1][p_count];
    //    Point pa[p_count];
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
        vertices.push_back(Point(x, y));
        //        points[0][i] = Point(x, y);
        //        pa[i] = Point(x, y);
        ++i;
    }
}

/**
 * 
 * @param vertices
 * @param tx bounding rectangle top-left corner x
 * @param ty bounding rectangle top-left corner y
 */
void shift_coordinates(vector<Point>& vertices, int tx, int ty) {
    for (vector<Point>::size_type i = 0; i < vertices.size(); i++) {
        vertices[i] = Point(vertices[i].x - tx, vertices[i].y - ty);
    }
}

void plot_line(Mat& im, vector<Point>& vertices, Scalar color, int thickness) {
    if (vertices.size() == 1) return;
    for (vector<Point>::size_type i = 0; i < vertices.size() - 1; i++) {
        Point p1 = vertices[i];
        Point p2 = vertices[i + 1];
        line(im, p1, p2, color, thickness);
    }
}

/*
 * 
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    string path = argv[2];
    int thr = 100;
    string road = "199311841#1";
    string ext = ".png";
    int p_count = 40;
    string shape = "11522.376776458877,15103.0 11504.024386622477,14814.41164474354 11453.009514084446,13908.672590695121 11428.16626305738,13632.04085876345 11401.277063364134,13423.481394582857 11362.426527386293,13273.55277796619 11325.002850915282,13112.498711141932 11245.715290737206,12812.511276097875 10983.078741533796,12015.46634859921 10727.286491824929,11237.855957630849 10394.624883254446,10214.95998875001 10319.585734242113,9980.173276851165 10256.157685045275,9773.534099229992 10194.15241390104,9577.744603558393 10152.019861420627,9432.608267846776 10117.81991483328,9267.767804205068 10102.79329545622,9126.15907161077 10106.375273310825,8965.502869270635 10119.32822946215,8787.84555341935 10124.661994113723,8720.88171051332 9547.71695404924,8672.353311594512 9542.382896712166,8739.317129886975 9529.121127508628,8921.03803688513 9524.102690737334,9111.177129870164 9541.49301880146,9323.507933370985 9584.69680385682,9550.847772041963 9638.326560398227,9743.929768555941 9705.575597538678,9957.231491801977 9767.515130792914,10159.088676732705 9844.637163644162,10400.52697332764 10178.259599784902,11426.40129878578 10434.6881981491,12205.954907715233 10697.421207845402,13003.294992541772 10766.077318513728,13267.316369304315 10799.27627753809,13411.253657829828 10841.798895340764,13576.174016317009 10853.280776287473,13701.753869292312 10876.566768653574,13963.127228131065 10926.110590216515,14848.670642634233 10944.475355134973,15103.0";
    string lshape = "9836.18947408148,8696.617511053917 9830.855563087158,8763.581341653162 9817.748200410208,8943.270453075904 9813.447993096537,9118.668100740468 9829.65646681737,9295.637868790005 9868.358332638483,9491.728019946348 9916.239487148912,9660.837186059145 9980.86664129342,9865.382795514004 10043.550432517031,10069.630976793913 10119.63102344858,10307.743481038826 10452.773045804675,11332.128628208315 10708.88346984217,12110.710628157221 10971.568249292026,12907.903134321803 11045.540084713542,13189.907540223123 11080.851402461229,13342.403217898009 11121.537979351006,13499.827705447955 11140.723519672425,13666.89736402788 11164.788141369492,13935.899909413092 11215.067488419496,14831.541143690865 11233.426065796684,15103.0";

    Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    Size size = src.size.operator()();
    Mat mask = Mat::zeros(size, src.type());

    //    cout << "Size width: " << size.width << " height: " << size.height << endl;
    //    Rect irect(Point(), size);


    clock_t begin = clock();

    vector< vector<Point> > co_ordinates;
    vector<Point> vertices;
    get_coordinates(vertices, shape, p_count, size);
    co_ordinates.push_back(vertices);

    //    vector<vector<Point> >hull(co_ordinates.size());
    //    convexHull(Mat(co_ordinates[0]), hull[0], false);
    //    for (int i = 0; i < co_ordinates.size(); i++) {
    //        convexHull(Mat(co_ordinates[i]), hull[i], false);
    //    }

    drawContours(mask, co_ordinates, 0, Scalar(255, 255, 255), CV_FILLED, 8);
    //    vector<Point> t_coords = co_ordinates[0];
    //    Point* coords = &t_coords[0];
    //    fillPoly( mask, ppt, npt, 1, Scalar( 255, 255, 255 ), 8 );

    //    fillPoly(mask, pa, p_count, Scalar(255, 255, 255));
    //    string output = path + "/" + road + "-1-poly" + ext;
    //    imwrite(output, mask);

    //    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    //    imshow("Display window", mask); // Show our image inside it.
    //    waitKey(0); // Wait for a keystroke in the window

    //    Mat wmask(src.size(), src.type(), Scalar(255));


    //    for (vector<Point>::iterator it = vertices.begin(); it != vertices.end(); ++it) {
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
    res(rect).copyTo(cropped);
    cout << "Scan and invert" << endl;
    scanImageAndInvertIterator(cropped);
    //    invert(cropped);

    //    string output1 = path + "/" + road + "-" + boost::lexical_cast<string>(thr) + "-out1" + ext;
    //    imwrite(output1, cropped);

    Mat imask(rect.width, rect.height, src.type());
    //    GaussianBlur(cropped,contours)
    bilateralFilter(cropped, imask, 13, 255, 255);
    //    bilateralFilter(cropped, imask, 7, 150, 50);

    //    bilateralFilter(cropped, imask, 15, 255, 255);
    //    blur(cropped, contours, Size(9, 9));

    medianBlur(imask, imask, 9);

    // reduce the intensity of the of the figure to reduce the effects of the shadow.
    //    imask.convertTo(imask, -1, 1, -70);
    //    Canny(cropped, contours,200,350);
    cvtColor(imask, imask, CV_RGB2GRAY);

    //    convert image into binary .....
    threshold(imask, imask, thr, 255, THRESH_BINARY);
    string binary_output = path + "/" + road + "-" + boost::lexical_cast<string>(thr) + "-binary" + ext;
    imwrite(binary_output, imask);
    cout << "finish writing binary image .." << endl;
    //    adaptiveThreshold(imask, imask,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);

    cout << "Remove small blobs" << endl;
    removeSmallBlobs(imask, 200);
    //        threshSegments(imask, 250);
    //    cout << "end removing small blobs" << endl;
    //    Mat sel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    //    erosion(imask, 2, 10);
    //    dilation(imask, 0, 15);
    //    
    //    erosion(imask,0, 7);
    //    imopen(imask, 0, 7);
    //    dilation(imask, 0, 3);
    //    imopen(imask, 0, 7);
    imclose(imask, 0, 15);
    string morph_output = path + "/" + road + "-" + boost::lexical_cast<string>(thr) + "-morph" + ext;
    imwrite(morph_output, imask);
    cout << "finish writing morphological results..." << endl;
    //    invert the binary image....
    bitwise_not(imask, imask);
    cout << "Remove small blobs" << endl;
    removeSmallBlobs(imask, 350);
    
    cout << "Thinning ..." << endl;
    thinning(imask);
    cout << "End thinning " << endl;

    vector<Point> lvertices;
    get_coordinates(lvertices, lshape, p_count / 2, size);
    shift_coordinates(lvertices, tx, ty);
    plot_line(imask, lvertices, Scalar(255, 255, 255), 8);

    //    vector<vector < Point> > contours;
    //    find the contours of the binary image...
    //    findContours(imask, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    //    drawContours(imask, contours, -1, Scalar(255), CV_FILLED, 4);

    //    for (vector< vector<Point> >::iterator it = contours.begin(); it != contours.end(); ++it) {
    //        Rect t_rect = boundingRect(*it);
    //        rectangle(imask, t_rect, Scalar(255), 2, 4, 0);
    //    }
    //    Mat res_ones;

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Elapsed time \t" << elapsed_secs << endl;
    cout << "New size width: " << rect.width << " height: " << rect.height << endl;
    string output = path + "/" + road + "-" + boost::lexical_cast<string>(thr) + "-thinning" + ext;
    imwrite(output, imask);
    //    waitKey(0);
    return 0;
}

