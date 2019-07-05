
/* 
 * Real-time traffic snapshot estimation using single pass satellite imagery and disparity maps.
 * 
 * Author: essam
 *
 * Created on July 17, 2015, 4:31 PM
 * g++ -O2 -std=c++11 -fopenmp Utils.cpp preprocessing_4.cpp -o preprocessing_4.o `pkg-config --libs --cflags opencv` -lopencvblobslib -lboost_filesystem -lboost_system
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
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void fillBlobs(Mat& src, Mat& dst, bool blob, int NUMCORES) {
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
        } else {
            t_contours hull;
            curblob->GetConvexHull(hull);
            Point center = curblob->getCenter();
            //                approxPolyDP();
            //        if (curblob->Area(PIXELWISE) > 400)
            drawContours(dst, hull, -1, Scalar(255, 255, 255), CV_FILLED, 4);
        }
        //        }
        //        


    }
}

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

#if 0
    utils.imenhance(src, src);
    string enh_output = path + "/w_z_6" + ext;
    imwrite(enh_output, src);
#endif

#if 1
    blur(src, imask, Size(3, 3));
    //    utils.imblur(src, imask);
#else
    //        bilateralFilter(src, imask, 7, 150, 50);

    imask = src;
#endif
    // reduce the intensity of the of the figure to reduce the effects of the shadow.
    //    imask.convertTo(imask, -1, 1, -gray_th);
    //scanImageAndInvertIterator(imask);

    cvtColor(imask, imask, CV_RGB2GRAY);

    // Apply Histogram Equalization
    //    equalizeHist(imask, imask);
    //    string eq_output = path + "/equalizeHist" + ext;
    //    imwrite(eq_output, imask);

    Mat edges;
    utils.CannyThreshold(imask, edges, 50, 3);

    string gray_output = path + "/gray" + ext;
    imwrite(gray_output, imask);



    threshold(imask, imask, gray_th, 255, THRESH_BINARY);

    //    iterative_imopen(imask, 0, 5);
    utils.iterative_imclose(imask, 0, 15);

    //    utils.imclose(edges, 1, 2);
    string binary_output = path + "/binary" + ext;
    imwrite(binary_output, imask);
    cout << "finish writing binary image .." << endl;
    //    adaptiveThreshold(imask, imask,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);

#if 1
    bool blob = true;
    utils.fillBlobs(edges, edges, blob, NUMCORES);

    utils.imclose(edges, 1, 2);
    utils.fillBlobs(edges, edges, blob, NUMCORES);
    //    utils.iterative_imopen(edges, 2, 4);
    //    
#if 0
    utils.iterative_imclose(edges, 0, 3);
    utils.CannyThreshold(edges, edges, 50, 3);
    string edges_output = path + "/edges" + ext;
    imwrite(edges_output, edges);
    blob = true;
    utils.fillBlobs(edges, edges, blob, NUMCORES);
#endif
    string blob_output = path + "/blob" + ext;
    imwrite(blob_output, edges);

    Mat not_blobs;
    bitwise_not(edges, not_blobs);
    string not_blob_output = path + "/Not blobs - 50 blur" + ext;
    imwrite(not_blob_output, not_blobs);


    //    blob = true;
    //    utils.fillBlobs(res, res, blob, NUMCORES);
    //    utils.iterative_imclose(res, 2, 15);
    //
    //    string blob_and_output = path + "/fill blobs of bitwise_and" + ext;
    //    imwrite(blob_and_output, res);



#endif
    return 0;
}

