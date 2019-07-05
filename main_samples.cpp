/* 
 * File:   main_samples.cpp
 * Author: essam
 *
 * Created on August 24, 2015, 12:56 AM
 */

#include <cstdlib>
#include <opencvblobslib/blob.h>
#include <opencvblobslib/BlobResult.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <boost/lexical_cast.hpp>
#include "Utils.h"
using namespace cv;
using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {

    Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR | IMREAD_ANYDEPTH);
    string path = argv[2];

    int NUMCORES = 4;
    int gray_th = 64;
    Mat imask = src;
//    imask.convertTo(imask, -1, 1, -gray_th);
    //scanImageAndInvertIterator(imask);
    Utils utils;
    
    cvtColor(imask, imask, CV_RGB2GRAY);
    Mat edges;
    utils.CannyThreshold(imask, edges, 60, 3);
//    threshold(imask, imask, gray_th, 224, THRESH_BINARY);

    CBlobResult blobs;
    blobs = CBlobResult(edges, Mat(), NUMCORES);
    //    cout << "Tempo MT: " << (getTickCount() - time) / getTickFrequency() << endl;
    CBlob *curblob;
    int numBlobs = blobs.GetNumBlobs();
    cout << "number of blobs:\t" << numBlobs << endl;
    //    cropped.setTo(Vec3b(0, 0, 0));
    for (int i = 0; i < numBlobs; i++) {
        //        Scalar mean, stddev;
        //        Vec3b color = Vec3b(255, 255, 255);
        curblob = blobs.GetBlob(i);
//        if (curblob->Area(PIXELWISE) > 3000) continue;
        CvRect cv_rect = curblob->GetBoundingBox();
        Rect rect(cv_rect.x, cv_rect.y, cv_rect.width, cv_rect.height);
        Point cp = curblob->getCenter();
        //        curblob->FillBlob(src, Scalar(255));
        int dim = rect.width;
        if (rect.width < rect.height) {
            dim = rect.height;
        }
        Mat sample(dim, dim, src.type());
        Rect srect(cp.x - dim / 2, cp.y - dim / 2, dim, dim);
        try {
            src(srect).copyTo(sample);
        } catch (cv::Exception) {
            continue;
        }
        string out = path + "/" + boost::lexical_cast<string>(i) + ".png";
        cout << "\t" << out << endl;
        imwrite(out, sample);

    }
    return 0;
}

