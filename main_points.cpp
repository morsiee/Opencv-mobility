/* 
 * File:   main_points.cpp
 * Author: essam
 *
 * Created on August 19, 2015, 11:48 AM
 */

#include<vector>
#include<boost/shared_ptr.hpp>
#include<CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include<CGAL/Polygon_2.h>
#include<CGAL/create_offset_polygons_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K ;
typedef K::Point_2                   Point ;
typedef CGAL::Polygon_2<K>           Polygon_2 ;

/*
 * 
 */
int main(int argc, char** argv) {


    // Compute all intersection points.
    Point a(40, 30), b(90, 80), c(70, 70), d(140, 60), e(130, 80), f(100, 50), g(80, 30);
    Polygon_2 p;
    p.push_back(a);
    p.push_back(b);
    p.push_back(c);
    p.push_back(d);
    p.push_back(e);
    p.push_back(f);
    p.push_back(g);

   
    bool isSimple = p.is_simple();
    std::cout<< isSimple << std::endl; 
    
//    bool IsConvex = p.is_convex();
//    bool IsClockwise = (p.orientation() == CGAL::CLOCKWISE);
//    double Area = p.area();
//    std::cout << "polygon p is";
//    if (!IsSimple) std::cout << " not";
//    std::cout << " simple." << std::endl;
//    std::cout << "polygon p is";
//    if (!IsConvex) std::cout << " not";
//    std::cout << " convex." << std::endl;
//    std::cout << "polygon p is";
//    if (!IsClockwise) std::cout << " not";
//    std::cout << " clockwise oriented." << std::endl;
//    std::cout << "the area of polygon p is " << Area << std::endl;
//    std::cout << std::endl;
//    // apply some algorithms
//    Point q(1, 1);
//    std::cout << "created point q = " << q << std::endl;
//    std::cout << std::endl;
//    bool IsInside = (p.bounded_side(q) == CGAL::ON_BOUNDED_SIDE);
//    std::cout << "point q is";
//    if (!IsInside) std::cout << " not";
//    std::cout << " inside polygon p." << std::endl;
//    std::cout << std::endl;
//    // traverse the vertices and the edges
//    int n = 0;
//    for (VertexIterator vi = p.vertices_begin(); vi != p.vertices_end(); ++vi)
//        std::cout << "vertex " << n++ << " = " << *vi << std::endl;
//    std::cout << std::endl;
//    n = 0;
//    for (EdgeIterator ei = p.edges_begin(); ei != p.edges_end(); ++ei)
//        std::cout << "edge " << n++ << " = " << *ei << std::endl;



    return 0;
}

