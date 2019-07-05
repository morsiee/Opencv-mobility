
/* 
 * Real-time traffic snapshot estimation using single pass satellite imagery and disparity maps.
 * 
 * Author: essam
 *
 * Created on July 17, 2015, 4:31 PM
 * g++ -O2 -std=c++11 -fopenmp Utils.cpp preprocessing.cpp -o preprocessing.o `pkg-config --libs --cflags opencv` -lopencvblobslib -lboost_filesystem -lboost_system
 */
#include "Utils.h"
#include <cstdlib>
#include<iostream>
#include<random>
#include <stdio.h>
#include <fstream>
#include<cmath>
#include<random>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/photo/photo.hpp>
#include <opencvblobslib/blob.h>
#include <opencvblobslib/BlobResult.h>
#include <sstream>
#include <vector>
#include <ctime>
#include<string>
#include <omp.h>
#include <vector>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>

using namespace cv;
using namespace std;

#if 0

/**
 * blur image ..
 * @param src
 * @param dest
 */
void blur(Mat& src, Mat& dest) {
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
void iterative_imopen(Mat& im, int structure, int size) {
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
 * ehance image details ..
 * 
 * @param src
 * @param dest
 */
void imenhance(Mat& src, Mat& dest) {
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

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(Mat& src, Mat& dst, int threshold, int kernel_size) {
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
void fillBlobs(Mat& src, Mat& dst, int NUMCORES) {
    CBlobResult blobs;

    blobs = CBlobResult(src, Mat(), NUMCORES);
    //    cout << "Tempo MT: " << (getTickCount() - time) / getTickFrequency() << endl;
    CBlob *curblob;

    int numBlobs = blobs.GetNumBlobs();
    cout << "found: " << numBlobs << endl;
    for (int i = 0; i < numBlobs; i++) {
        curblob = blobs.GetBlob(i);
        //        curblob->FillBlob(imask, Scalar(255));

        t_contours hull;
        curblob->GetConvexHull(hull);
        //        if (curblob->Area(PIXELWISE) > 400)
        drawContours(dst, hull, -1, Scalar(255, 255, 255), CV_FILLED, 4);
    }
}
#endif

/*
 * 
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        cout << " Usage: [Road image] [Results path]." << endl;
        return -1;
    }

    Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    string path = argv[2];
    string ext = ".png";
    int NUMCORES = 4;
    int gray_th = 64;
    Size size = src.size.operator()();
    Utils utils;
    //    detailEnhance(Mat src, Mat dst, float sigma_s=10, float sigma_r=0.15f);

    Mat imask(size.width, size.height, src.type());

#if 1
    utils.imenhance(src, src);
    string enh_output = path + "/w_z_6" + ext;
    imwrite(enh_output, src);
#endif

#if 1
    utils.imblur(src, imask);
#else
    imask = src;
#endif
    // reduce the intensity of the of the figure to reduce the effects of the shadow.
    imask.convertTo(imask, -1, 1, -gray_th);
    //scanImageAndInvertIterator(imask);

    cvtColor(imask, imask, CV_RGB2GRAY);
    Mat edges;
    utils.CannyThreshold(imask, edges, 50, 3);

    string gray_output = path + "/gray" + ext;
    imwrite(gray_output, imask);

    string edges_output = path + "/edges" + ext;
    imwrite(edges_output, edges);

    threshold(imask, imask, gray_th, 224, THRESH_BINARY);

    //    iterative_imopen(imask, 0, 5);
    utils.iterative_imclose(imask, 2, 11);

    //    iterative_imclose(imask, 0, 4);
    string binary_output = path + "/binary" + ext;
    imwrite(binary_output, imask);
    cout << "finish writing binary image .." << endl;
    //    adaptiveThreshold(imask, imask,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);

#if 1
    bool blob = false;
    utils.fillBlobs(edges, edges, blob, NUMCORES);
    //    iterative_imopen(edges, 0, 5);
    utils.iterative_imclose(edges, 2, 11);

    string blob_output = path + "/blob" + ext;
    imwrite(blob_output, edges);

    Mat res;
    bitwise_and(edges, imask, res);

    //changed from 11 to 21
    utils.iterative_imclose(res, 2, 11);
    utils.iterative_imopen(res, 2, 21);

    string and_output = path + "/bitwise_and" + ext;
    imwrite(and_output, res);
    
//    blob = true;
    utils.fillBlobs(res, res, blob, NUMCORES);
    utils.iterative_imclose(res, 2, 21);

    string blob_and_output = path + "/fill blobs of bitwise_and" + ext;
    imwrite(blob_and_output, res);

    Mat not_res;
    bitwise_not(res, not_res);

    string not_output = path + "/not_res" + ext;
    imwrite(not_output, not_res);

#endif
    return 0;
}

