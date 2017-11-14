# picoflann
picoflann is minimal library for Exact KdTree Search. It is an step forward [nanoflann](https://github.com/jlblancoc/nanoflann) but faster and easier to use.

## Computing times

Speed comparison with nanoflann on a data set of 10000 2d points:

```
Time in milliseconds of a total of 10000 Repetitions.

Knn Search of 1000 elements
PicoFlann build= 1.37571 KnnSearch=1755.25
NanoFlann build= 2.01148 KnnSearch=7270.61

Radius search (we are a bit slower :S)
PicoFlann build= 1.408 radiusSearch=5.27617
NanoFlann build= 2.16247 radiusSearch=4.1935
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
    picoflann::KdTreeIndex<2>  kdtree;
    kdtree.build(data,PicoFlann_Point2fAdapter());//2 is the number of dimensions
    //search 10 nearest neibors to point (0,0)
    std::vector<std::pair<uint32_t,double> > res=kdtree.searchKnn(data,PicoFlann_Point2fAdapter(),Point2f(0,0),10);

    //radius search in a radius of 30 (the resulting distances are squared)
    res=kdtree.radiusSearch(data,PicoFlann_Point2fAdapter(),Point2f(0,0),30);
    //another version
    kdtree.radiusSearch(data,PicoFlann_Point2fAdapter(),res,Point2f(0,0),30);

    //you can save to a file
    std::ofstream file_out("out.bin",std::ios::binary);
    kdtree.toStream(file_out);

    //recover from the file
    picoflann::KdTreeIndex<2>  kdtree2;
    std::ifstream file_in("out.bin",std::ios::binary);
    kdtree2.fromStream(file_in);
    res=kdtree2.radiusSearch(data,PicoFlann_Point2fAdapter(),Point2f(0,0),30);
 ```
 
