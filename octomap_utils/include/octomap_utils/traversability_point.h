#ifndef TRAVERSABILITY_POINT_H
#define TRAVERSABILITY_POINT_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

struct EIGEN_ALIGN16 TraversabilityPoint    
{
    PCL_ADD_POINT4D;                  
    float intensity;
    float traversability;
    PCL_MAKE_ALIGNED_OPERATOR_NEW     
};

POINT_CLOUD_REGISTER_POINT_STRUCT(TraversabilityPoint,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (float, traversability, traversability)
)

#endif // TRAVERSABILITY_POINT_H