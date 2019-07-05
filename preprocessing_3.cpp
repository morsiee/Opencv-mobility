
/* 
 * Real-time traffic snapshot estimation using single pass satellite imagery and disparity maps.
 * 
 * Author: essam
 *
 * Created on July 17, 2015, 4:31 PM
 * g++ -O2 -std=c++11 -fopenmp Utils.cpp preprocessing_3.cpp -o preprocessing_3.o `pkg-config --libs --cflags opencv` -lopencvblobslib -lboost_filesystem -lboost_system
 * ./preprocessing_3.o ~/work/w_z_6/all/w_z_6.bmp ~/work/w_z_6/tmp/
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
    utils.CannyThreshold(imask, edges, 40, 3);

    string gray_output = path + "/gray" + ext;
    imwrite(gray_output, imask);

    string edges_output = path + "/edges" + ext;
    imwrite(edges_output, edges);

    threshold(imask, imask, gray_th, 224, THRESH_BINARY);

    //    iterative_imopen(imask, 0, 5);
    utils.iterative_imclose(imask, 0, 11);

    //    iterative_imclose(imask, 0, 4);
    string binary_output = path + "/binary" + ext;
    imwrite(binary_output, imask);
    cout << "finish writing binary image .." << endl;
    //    adaptiveThreshold(imask, imask,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);

#if 1
    bool blob = false;
    utils.fillBlobs(edges, edges, blob, NUMCORES);
    //    iterative_imopen(edges, 0, 5);
    utils.iterative_imclose(edges, 0, 11);

    string blob_output = path + "/blob" + ext;
    imwrite(blob_output, edges);

    Mat res;
    bitwise_and(edges, imask, res);

    //changed from 11 to 21
    utils.iterative_imclose(res, 0, 11);
    //    utils.iterative_imopen(res, 2, 21);

    string and_output = path + "/bitwise_and" + ext;
    imwrite(and_output, res);

    blob = !blob;
    utils.fillBlobs(res, res, blob, NUMCORES);
    utils.iterative_imclose(res, 0, 15);

    string blob_and_output = path + "/fill blobs of bitwise_and" + ext;
    imwrite(blob_and_output, res);

    Mat not_res;
    bitwise_not(res, not_res);
    utils.iterative_imclose(res, 0, 11);

    string not_output = path + "/not_res" + ext;
    imwrite(not_output, not_res);

#endif
    return 0;
}

