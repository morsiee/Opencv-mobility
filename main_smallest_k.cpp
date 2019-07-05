/* 
 * File:   main_smallest_k.cpp
 * Author: essam
 *
 * Created on August 23, 2015, 4:18 AM
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

using namespace std;
double avg_k_smallest(vector<double>& elements, int k);
/*
 * 
 */
int main(int argc, char** argv) {
    vector<double> v{5, 6, 4, 3, 2, 6, 7, 9, 3};

//    nth_element(v.begin(),v.begin() + v.size()/2, v.end());
//    cout << "The median is " << v[1] << '\n';
//
//    nth_element(v.begin(), v.begin() + 1, v.end(), greater<int>());
//    cout << "The second largest element is " << v[1] << '\n';
    cout << avg_k_smallest(v, 4)<< endl;
    return 0;
}

double avg_k_smallest(vector<double>& elements, int k) {
    double sum = 0;
    int count=0;
    nth_element(elements.begin(), elements.begin() + elements.size() / 2, elements.end());
    for (int i = 0; i < k && i < elements.size(); i++) {
        sum += elements[i];
        count++;
    }
    return sum / count;
}
