/* 
 * File:   MatchTemplate_1.cpp
 * Author: essam
 *
 * Created on August 16, 2015, 11:41 AM
 */

#include <cstdlib>
#include <iostream>
#include <dirent.h>
//#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
//#include <thread>
#include <unordered_map>
#include <boost/lexical_cast.hpp>
using namespace std;

void getFilesInDirectory(vector<string>& out, string directory) {
    DIR* dir;
    dirent* pdir;

    dir = opendir(directory.c_str()); // open current directory

    while (pdir = readdir(dir)) {
        string file(pdir->d_name);
        if (file.compare(".") == 0 || file.compare("..") == 0) {
            continue;
        }

        out.push_back(pdir->d_name);
        //        cout << pdir->d_name << endl;
    }
    closedir(dir);
}

int main(int argc, char** argv) {
    cv::Mat ref = cv::imread(argv[3], CV_LOAD_IMAGE_COLOR | cv::IMREAD_ANYDEPTH);
    if (ref.empty())
        return -1;
    for (int i = 0; i < 15; i++)
        cv::blur(ref, ref, cv::Size(5, 5));

    //"Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED"
    int match_method = atoi(argv[1]);
    double threshold = atof(argv[2]);
    //    cv::Mat gref;
    //    cv::cvtColor(ref, gref, CV_BGR2GRAY);

    vector<string> tpl_list;
    string dir = argv[4];
    getFilesInDirectory(tpl_list, dir);
    cout << "Number of files:\t" << tpl_list.size() << endl;
    //match locations ..
    int count = 0;
    unordered_map<int, cv::Point> matchLocMap;
    unordered_map<int, cv::Point> matchTplMap;



    for (vector<string>::size_type i = 0; i != tpl_list.size(); i++) {

        cout << i << " : " << tpl_list[i] << endl;
        string path = dir + "/" + tpl_list[i];
        cv::Mat tpl = cv::imread(path, CV_LOAD_IMAGE_COLOR | cv::IMREAD_ANYDEPTH);
        if (tpl.empty()) return -1;
        for (int i = 0; i < 9; i++)
            cv::blur(tpl, tpl, cv::Size(3, 3));
        
        //        cv::Mat gtpl;
        //        cv::cvtColor(tpl, gtpl, CV_BGR2GRAY);

        cv::Mat res(ref.rows - tpl.rows + 1, ref.cols - tpl.cols + 1, CV_32FC1);
        cv::matchTemplate(ref, tpl, res, match_method);
        cv::normalize(res, res, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        //        cv::matchTemplate(gref, gtpl, res, CV_TM_CCOEFF_NORMED);
        //        cv::threshold(res, res, 0.7, 1.0, CV_THRESH_TOZERO);


        while (true) {
            double minval, maxval;
            cv::Point minloc, maxloc, matchLoc;

            cv::minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);
            /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
            if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) {
                matchLoc = minloc;
            } else {
                matchLoc = maxloc;
            }
            if (maxval >= threshold) {
                cv::Point tmp(matchLoc.x + tpl.cols, matchLoc.y + tpl.rows);
                matchLocMap.insert(make_pair(count, matchLoc));
                matchTplMap.insert(make_pair(count, tmp));
                count++;
                //                cv::rectangle(
                //                        ref,
                //                        matchLoc,
                //                        cv::Point(matchLoc.x + tpl.cols, matchLoc.y + tpl.rows),
                //                        CV_RGB(0, 255, 0), 2
                //                        );
                cv::floodFill(res, matchLoc, cv::Scalar(0), 0, cv::Scalar(.1), cv::Scalar(1.));
            } else
                break;
        }


    }

    for (auto it = matchLocMap.begin(); it != matchLocMap.end(); ++it) {
        count = it->first;
        cv::Point lb = it->second;
        unordered_map<int, cv::Point>::const_iterator got = matchTplMap.find(count);
        if (got == matchTplMap.end()) return -1;
        cv::Point rt = got->second;
        cv::rectangle(ref, lb, rt, CV_RGB(0, 255, 0), 2);
    }

    //    cv::Mat ref = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR | cv::IMREAD_ANYDEPTH);
    //    cv::Mat tpl = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR | cv::IMREAD_ANYDEPTH);
    //

    cv::imwrite("/home/essam/work/res-" + boost::lexical_cast<string>(threshold) + "-" + boost::lexical_cast<string>(match_method) + ".png", ref);

    return 0;
}
