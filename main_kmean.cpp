/* 
 * File:   main_kmain.cpp
 * Author: essam
 *
 * Created on August 25, 2015, 3:05 AM
 */

#include <cstdlib>
#include <iostream>
#include <vector>
#include<boost/lexical_cast.hpp>
#include <dlib/clustering.h>
#include <dlib/rand.h>

using namespace std;
//using namespace dlib;

int test1();
void skmean(vector<double >& shift, vector<unsigned long>& assignments);

int main() {
#if 0
    typedef matrix<double, 1, 1> sample_type;
    // Now we are making a typedef for the kind of kernel we want to use.  I picked the
    // radial basis kernel because it only has one parameter and generally gives good
    // results without much fiddling.
    typedef radial_basis_kernel<sample_type> kernel_type;

    // Here we declare an instance of the kcentroid object.  It is the object used to 
    // represent each of the centers used for clustering.  The kcentroid has 3 parameters 
    // you need to set.  The first argument to the constructor is the kernel we wish to 
    // use.  The second is a parameter that determines the numerical accuracy with which 
    // the object will perform part of the learning algorithm.  Generally, smaller values 
    // give better results but cause the algorithm to attempt to use more dictionary vectors 
    // (and thus run slower and use more memory).  The third argument, however, is the 
    // maximum number of dictionary vectors a kcentroid is allowed to use.  So you can use
    // it to control the runtime complexity.  
    kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8);

    // Now we make an instance of the kkmeans object and tell it to use kcentroid objects
    // that are configured with the parameters from the kc object we defined above.
    kkmeans<kernel_type> test(kc);
    int num_clusters = 4;
    vector<sample_type> samples;
    vector<sample_type> initial_centers;

    double mysamples[] = {3, 7, 8.1, 9, -1.5, 2, 4, -3.5, 13, 9, -0.5, 2.5, 50, 20};
    for (int i = 0; i < (sizeof (mysamples) / sizeof (double)); i++) {
        sample_type m;
        m(0) = mysamples[i];
        samples.push_back(m);

    }


    // tell the kkmeans object we made that we want to run k-means with k set to 3. 
    // (i.e. we want 3 clusters)
    test.set_number_of_centers(num_clusters);

    // You need to pick some initial centers for the k-means algorithm.  So here
    // we will use the dlib::pick_initial_centers() function which tries to find
    // n points that are far apart (basically).  
    pick_initial_centers(num_clusters, initial_centers, samples, test.get_kernel());

    // now run the k-means algorithm on our set of samples.  
    test.train(samples, initial_centers);

    // now loop over all our samples and print out their predicted class.  In this example
    // all points are correctly identified.

    //    for (unsigned long i = 0; i < samples.size() / 3; ++i) {
    //        cout << test(samples[i]) << " ";
    //        cout << test(samples[i + num]) << " ";
    //        cout << test(samples[i + 2 * num]) << "\n";
    //    }

    // Now print out how many dictionary vectors each center used.  Note that 
    // the maximum number of 8 was reached.  If you went back to the kcentroid 
    // constructor and changed the 8 to some bigger number you would see that these
    // numbers would go up.  However, 8 is all we need to correctly cluster this dataset.
    for (int i = 0; i < num_clusters; i++) {

        printf("num dictionary vectors for center %s: %s\n",
                boost::lexical_cast<string>(i).c_str(),
                boost::lexical_cast<string>(test.get_kcentroid(i).dictionary_size()).c_str());
    }
    //    cout << "num dictionary vectors for center 0: " << test.get_kcentroid(0).dictionary_size() << endl;
    //    cout << "num dictionary vectors for center 1: " << test.get_kcentroid(1).dictionary_size() << endl;
    //    cout << "num dictionary vectors for center 2: " << test.get_kcentroid(2).dictionary_size() << endl;


    // Finally, we can also solve the same kind of non-linear clustering problem with
    // spectral_cluster().  The output is a vector that indicates which cluster each sample
    // belongs to.  Just like with kkmeans, it assigns each point to the correct cluster.

    vector<unsigned long> assignments = spectral_cluster(kernel_type(0.1), samples, num_clusters);

    for (int i = 0; i < assignments.size(); i++) {
        printf("%s : %s\n", boost::lexical_cast<string>(assignments[i]).c_str(), boost::lexical_cast<string>(samples[i](0)).c_str());
        //        cout <<<<" : "<< samples[i](0)<<endl;
    }
    //    cout << mat(assignments) << endl;
#else
    double mysamples[] = {3, 7, 8.1, 9, -1.5, 2, 4, -3.5, 13, 9, -0.5, 2.5, 50, 20};
    vector<double> shift(mysamples, mysamples + sizeof (mysamples) / sizeof (double));
    vector<unsigned long> assignments;

    skmean(shift, assignments);
    for (int i = 0; i < assignments.size(); i++) {
        printf("%s : %s\n", boost::lexical_cast<string>(assignments[i]).c_str(), boost::lexical_cast<string>(shift[i]).c_str());
    }
#endif
}

void skmean(vector<double>& shift, vector<unsigned long>& assignments) {
    typedef dlib::matrix<double, 1, 1> sample_type;
    // Now we are making a typedef for the kind of kernel we want to use.  I picked the
    // radial basis kernel because it only has one parameter and generally gives good
    // results without much fiddling.
    typedef dlib::radial_basis_kernel<sample_type> kernel_type;

    // Here we declare an instance of the kcentroid object.  It is the object used to 
    // represent each of the centers used for clustering.  The kcentroid has 3 parameters 
    // you need to set.  The first argument to the constructor is the kernel we wish to 
    // use.  The second is a parameter that determines the numerical accuracy with which 
    // the object will perform part of the learning algorithm.  Generally, smaller values 
    // give better results but cause the algorithm to attempt to use more dictionary vectors 
    // (and thus run slower and use more memory).  The third argument, however, is the 
    // maximum number of dictionary vectors a kcentroid is allowed to use.  So you can use
    // it to control the runtime complexity.  
    dlib::kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8);

    // Now we make an instance of the kkmeans object and tell it to use kcentroid objects
    // that are configured with the parameters from the kc object we defined above.
    dlib::kkmeans<kernel_type> test(kc);
    int num_clusters = 4;
    vector<sample_type> samples(shift.size());
    vector<sample_type> initial_centers;

    for (int i = 0; i < shift.size(); i++) {
        sample_type m;
        m(0) = shift[i];
        samples[i] = m;
    }

    // tell the kkmeans object we made that we want to run k-means with k set to 3. 
    // (i.e. we want 3 clusters)
    test.set_number_of_centers(num_clusters);

    // You need to pick some initial centers for the k-means algorithm.  So here
    // we will use the dlib::pick_initial_centers() function which tries to find
    // n points that are far apart (basically).  
    dlib::pick_initial_centers(num_clusters, initial_centers, samples, test.get_kernel());

    // now run the k-means algorithm on our set of samples.  
    test.train(samples, initial_centers);
    // Finally, we can also solve the same kind of non-linear clustering problem with
    // spectral_cluster().  The output is a vector that indicates which cluster each sample
    // belongs to.  Just like with kkmeans, it assigns each point to the correct cluster.

    assignments = dlib::spectral_cluster(kernel_type(0.1), samples, num_clusters);

}

int test1() {
    // Here we declare that our samples will be 2 dimensional column vectors.  
    // (Note that if you don't know the dimensionality of your vectors at compile time
    // you can change the 2 to a 0 and then set the size at runtime)
    typedef dlib::matrix<double, 2, 1> sample_type;

    // Now we are making a typedef for the kind of kernel we want to use.  I picked the
    // radial basis kernel because it only has one parameter and generally gives good
    // results without much fiddling.
    typedef dlib::radial_basis_kernel<sample_type> kernel_type;


    // Here we declare an instance of the kcentroid object.  It is the object used to 
    // represent each of the centers used for clustering.  The kcentroid has 3 parameters 
    // you need to set.  The first argument to the constructor is the kernel we wish to 
    // use.  The second is a parameter that determines the numerical accuracy with which 
    // the object will perform part of the learning algorithm.  Generally, smaller values 
    // give better results but cause the algorithm to attempt to use more dictionary vectors 
    // (and thus run slower and use more memory).  The third argument, however, is the 
    // maximum number of dictionary vectors a kcentroid is allowed to use.  So you can use
    // it to control the runtime complexity.  
    dlib::kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8);

    // Now we make an instance of the kkmeans object and tell it to use kcentroid objects
    // that are configured with the parameters from the kc object we defined above.
    dlib::kkmeans<kernel_type> test(kc);

    vector<sample_type> samples;
    vector<sample_type> initial_centers;

    sample_type m;

    dlib::rand rnd;

    // we will make 50 points from each class
    const long num = 50;

    // make some samples near the origin
    double radius = 0.5;
    for (long i = 0; i < num; ++i) {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2 * radius * rnd.get_random_double() - radius;
        m(1) = sign * sqrt(radius * radius - m(0) * m(0));

        // add this sample to our set of samples we will run k-means 
        samples.push_back(m);
    }

    // make some samples in a circle around the origin but far away
    radius = 10.0;
    for (long i = 0; i < num; ++i) {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2 * radius * rnd.get_random_double() - radius;
        m(1) = sign * sqrt(radius * radius - m(0) * m(0));

        // add this sample to our set of samples we will run k-means 
        samples.push_back(m);
    }

    // make some samples in a circle around the point (25,25) 
    radius = 4.0;
    for (long i = 0; i < num; ++i) {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2 * radius * rnd.get_random_double() - radius;
        m(1) = sign * sqrt(radius * radius - m(0) * m(0));

        // translate this point away from the origin
        m(0) += 25;
        m(1) += 25;

        // add this sample to our set of samples we will run k-means 
        samples.push_back(m);
    }

    // tell the kkmeans object we made that we want to run k-means with k set to 3. 
    // (i.e. we want 3 clusters)
    test.set_number_of_centers(3);

    // You need to pick some initial centers for the k-means algorithm.  So here
    // we will use the dlib::pick_initial_centers() function which tries to find
    // n points that are far apart (basically).  
    dlib::pick_initial_centers(3, initial_centers, samples, test.get_kernel());

    // now run the k-means algorithm on our set of samples.  
    test.train(samples, initial_centers);

    // now loop over all our samples and print out their predicted class.  In this example
    // all points are correctly identified.
    for (unsigned long i = 0; i < samples.size() / 3; ++i) {
        cout << test(samples[i]) << " ";
        cout << test(samples[i + num]) << " ";
        cout << test(samples[i + 2 * num]) << "\n";
    }

    // Now print out how many dictionary vectors each center used.  Note that 
    // the maximum number of 8 was reached.  If you went back to the kcentroid 
    // constructor and changed the 8 to some bigger number you would see that these
    // numbers would go up.  However, 8 is all we need to correctly cluster this dataset.
    cout << "num dictionary vectors for center 0: " << test.get_kcentroid(0).dictionary_size() << endl;
    cout << "num dictionary vectors for center 1: " << test.get_kcentroid(1).dictionary_size() << endl;
    cout << "num dictionary vectors for center 2: " << test.get_kcentroid(2).dictionary_size() << endl;


    // Finally, we can also solve the same kind of non-linear clustering problem with
    // spectral_cluster().  The output is a vector that indicates which cluster each sample
    // belongs to.  Just like with kkmeans, it assigns each point to the correct cluster.
    vector<unsigned long> assignments = dlib::spectral_cluster(kernel_type(0.1), samples, 3);
    cout << dlib::mat(assignments) << endl;

}


