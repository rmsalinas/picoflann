# picoflann
Picoflann is minimal library for Exact KdTree Search. It is an step forward [nanoflann](https://github.com/jlblancoc/nanoflann) but faster and easier to use.

## Computing times

Speed comparison with nanoflann and OpenCv on a data set of 10000 2D points:

```
Speed up in 3D points:

Index building SpeedUp
-------------------------------
PicoFlann vs NanoFlann:     1.46x faster
PicoFlann vs OpenCv Flann:  1.73x faster

Knn SpeedUp
-------------------------
PicoFlann vs NanoFlann:     4.14x faster
PicoFlann vs OpenCv Flann:  1.8x faster


Radius Search 
PicoFlann vs NanoFlann:        1.25x faster
PicoFlann vs OpenCv Flann:     3.21x faster


## Citing

If you use this project in academic research you must cite us. This project is part of the ucoslam project. Visit ucoslam.com for more information
about how to cite this project.

```
# Example 
```
     //Data type
    struct Point2f{
        Point2f(float X,float Y) { x=X;y=Y; }
        float x,y;
    };

    // Adapter.
    // Given an Point2f element, it returns the element of the dimension specified such that dim=0 is x and dim=1 is y
    struct PicoFlann_Point2fAdapter{
        inline  float operator( )(const Point2f &elem, int dim)const { return dim==0?elem.x:elem.y; }
    };

    //create the points randomly
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1000.0,1000.0);
    std::vector<Point2f> data;
    for(size_t i=0;i<1000;i++)
        data.push_back( Point2f ( distribution(generator),distribution(generator)));
    ///------------------------------------------------------------
    /// Create the kdtree
    picoflann::KdTreeIndex<2,PicoFlann_Point2fAdapter>  kdtree;//2 is the number of dimensions
    kdtree.build(data);
    //search 10 nearest neibors to point (0,0)
    std::vector<std::pair<uint32_t,double> > res=kdtree.searchKnn(data,Point2f(0,0),10);

    //radius search in a radius of 30 (the resulting distances are squared)
    res=kdtree.radiusSearch(data,Point2f(0,0),30);
    //another version
    kdtree.radiusSearch(data,res,Point2f(0,0),30);

    //you can save to a file
    std::ofstream file_out("out.bin",std::ios::binary);
    kdtree.toStream(file_out);

    //recover from the file
    picoflann::KdTreeIndex<2,PicoFlann_Point2fAdapter>  kdtree2;
    std::ifstream file_in("out.bin",std::ios::binary);
    kdtree2.fromStream(file_in);
    res=kdtree2.radiusSearch(data,Point2f(0,0),30);
 ```
 
