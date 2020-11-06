#include<ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/GetMap.h"
#include "nav_msgs/Odometry.h"
#include<tf/tf.h>
#include<tf/transform_listener.h>
#include <iostream>
#include <fstream>
#include<string>
#include<math.h>
#include<ctime>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Geometry>


//#include <eigen3/Eigen/Dense>
#include "common.h"
#include "tools_logger.hpp"
#include "tools_random.hpp"
#include<vector>
#include<algorithm>
#include "pcl_tools.hpp"
// PCL specific includes
//#include <pcl/ros/conversions.h>
#include<pcl/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/impl/io.hpp>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include</home/beihai/ros_in_qt/src/find_transformation/src/keyframe_process.hpp>
using namespace std;
#define PI 3.1415926
class Local_map
{
    public:
    bool cornericp_continue=true;
    bool surfaceicp_continue=true;

    int numbers;
    int iter_num=0;
    float radius=0.3;
    size_t corner_frame_index=0;
    size_t surface_frame_index=0;

    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> corner_icp;
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> surface_icp;

    Eigen::Matrix4d corner_t_hist = Eigen::Matrix4d::Identity ();
    Eigen::Matrix4d surface_t_hist = Eigen::Matrix4d::Identity ();

//    vector<double> initial_fitness_score;
//    vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> initial_trans_matrix;
    vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> initial_trans_candidates;

    pcl::PCDWriter writer;
    Eigen::Matrix4d corner_transformation_matrix = Eigen::Matrix4d::Identity ();
    Eigen::Matrix4d surface_transformation_matrix = Eigen::Matrix4d::Identity ();

    Eigen::Matrix4d corner_ndttrans = Eigen::Matrix4d::Identity ();
    Eigen::Matrix4d surface_ndttrans = Eigen::Matrix4d::Identity ();

    Eigen::Matrix4f init_ndtguess=Eigen::Matrix4f::Identity ();
//    Eigen::Matrix3d rotation_matrix=Eigen::Matrix3d::Identity ();
    Eigen::Quaterniond quater;

    //pcl::KdTreeFLANN<PointType>::Ptr corner_curr_kdtree;
    //pcl::KdTreeFLANN<PointType>::Ptr surface_curr_kdtree;
    ros::NodeHandle nh;
    pcl::PointCloud<PointType>::Ptr cornermap=boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::PointCloud<PointType>::Ptr current_corner_local_map=boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::search::KdTree<PointType>::Ptr cornermap_kdtree=boost::make_shared<pcl::search::KdTree<PointType>>();
    ros::Subscriber corner_sub;

    pcl::PointCloud<PointType>::Ptr surfacemap=boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::PointCloud<PointType>::Ptr current_surface_local_map=boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::search::KdTree<PointType>::Ptr surfacemap_kdtree=boost::make_shared<pcl::search::KdTree<PointType>>();
    ros::Subscriber surface_sub;
    ros::Publisher corner_odometry_publisher,surface_odometry_publisher;
    DBoW3::Vocabulary surface_vocabulary;
    DBoW3::Database surface_database;
    flann::Matrix<float> data_transform;

    void
    print4x4Matrix (const Eigen::Matrix4d & matrix)
    {
      printf ("Rotation matrix :\n");
      printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
      printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
      printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
      printf ("Translation vector :\n");
      printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
    }
    void
    print4x4Matrixf (const Eigen::Matrix4f & matrix)
    {
      printf ("Rotation matrix :\n");
      printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
      printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
      printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
      printf ("Translation vector :\n");
      printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
    }
    int local_map_creater(pcl::PointCloud< PointType >::Ptr in_laser_cloud_corner_from_map,
                          pcl::PointCloud< PointType >::Ptr in_laser_cloud_surf_from_map,
                          pcl::search::KdTree<PointType>::Ptr   kdtree_corner_from_map,
                          pcl::search::KdTree<PointType>::Ptr   kdtree_surf_from_map,
                          pcl::PointCloud< PointType >::Ptr laserCloudCornerStack,
                          pcl::PointCloud< PointType >::Ptr laserCloudSurfStack )
    {
        pcl::search::KdTree<PointType>::Ptr corner_curr_kdtree( new pcl::search::KdTree<PointType>);
        pcl::search::KdTree<PointType>::Ptr surface_curr_kdtree( new pcl::search::KdTree<PointType>);
//        pcl::search::KdTree<PointType>::Ptr corner_kdtree( new pcl::search::KdTree<PointType>);
//        pcl::search::KdTree<PointType>::Ptr surface_kdtree( new pcl::search::KdTree<PointType>);
//        corner_kdtree = &kdtree_corner_from_map;
//        surface_kdtree = &kdtree_surf_from_map;
//        pcl::PointCloud< PointType > corner_localmap;
        pcl::PointCloud<PointType>::Ptr corner_localmap (new pcl::PointCloud<PointType>);

        pcl::PointCloud< pcl::PointXYZ > corner_localmapXYZ;
        pcl::PointCloud< pcl::PointXYZ > laserCloudCornerStackXYZ;

        pcl::PointCloud< PointType > surf_localmap;
        PointType searchPoint;
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        std::cerr<<laserCloudCornerStack->points.size()<<std::endl;
        for(int i=0;i<laserCloudCornerStack->points.size();i++)
        {
            searchPoint=laserCloudCornerStack->points[i];
            if ( kdtree_corner_from_map->radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
            {
    //            for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
    //              std::cout << "    "  <<in_laser_cloud_corner_from_map->points[ pointIdxRadiusSearch[ i ] ].x
    //                        << " " << in_laser_cloud_corner_from_map->points[ pointIdxRadiusSearch[ i ] ].y
    //                        << " " << in_laser_cloud_corner_from_map->points[ pointIdxRadiusSearch[ i ] ].z
    //                        << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
                //numbers+=pointIdxRadiusSearch.size();
                //std::cout<<pointIdxRadiusSearch.size()<<std::endl;
                for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
                {
                    corner_localmap->points.push_back(in_laser_cloud_corner_from_map->points[i]);
                }
            }
        }
        std::cerr<<corner_localmap->points.size()<<std::endl;

        //Transform
        Eigen::Quaterniond rotation=Eigen::Quaterniond(1,0,0,0);
        Eigen::Vector3d trans=Eigen::Vector3d(10,10,0.1);
        for(int i=0;i<laserCloudCornerStack->points.size();i++)
        {
            Eigen::Vector3d point_curr(laserCloudCornerStack->points[i].x,laserCloudCornerStack->points[i].y,laserCloudCornerStack->points[i].z);
            Eigen::Vector3d point_w;
            point_w=rotation.inverse()*point_curr-rotation.inverse()*trans;
            laserCloudCornerStack->points[i].x=point_w.x();
            laserCloudCornerStack->points[i].y=point_w.y();
            laserCloudCornerStack->points[i].z=point_w.z();
            laserCloudCornerStack->points[i].intensity=laserCloudCornerStack->points[i].intensity;
        }
        for(int i=0;i<laserCloudSurfStack->points.size();i++)
        {
            Eigen::Vector3d point_curr(laserCloudSurfStack->points[i].x,laserCloudSurfStack->points[i].y,laserCloudSurfStack->points[i].z);
            Eigen::Vector3d point_w;
            point_w=rotation.inverse()*point_curr-rotation.inverse()*trans;
            laserCloudSurfStack->points[i].x=point_w.x();
            laserCloudSurfStack->points[i].y=point_w.y();
            laserCloudSurfStack->points[i].z=point_w.z();
            laserCloudSurfStack->points[i].intensity=laserCloudSurfStack->points[i].intensity;
        }

//        corner_curr_kdtree->setInputCloud (laserCloudCornerStack);
//        surface_curr_kdtree->setInputCloud (laserCloudSurfStack);
        //ICP
        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> corner_icp;
//        corner_icp.setSearchMethodSource(corner_curr_kdtree);
//        corner_icp.setSearchMethodTarget(kdtree_corner_from_map);
        //corner_icp.set
        corner_icp.setInputSource(laserCloudCornerStack);
        corner_icp.setInputTarget(in_laser_cloud_corner_from_map);
        
        corner_icp.align(*laserCloudCornerStack);

        if (corner_icp.hasConverged ())
        {
          std::cout << "\nICP has converged, score is " << corner_icp.getFitnessScore () << std::endl;
          corner_transformation_matrix = corner_icp.getFinalTransformation ().cast<double>();
          print4x4Matrix(corner_transformation_matrix);
        }

        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> surface_icp;
//        surface_icp.setSearchMethodSource(surface_curr_kdtree);
//        surface_icp.setSearchMethodTarget(kdtree_surf_from_map);

        surface_icp.setInputSource(laserCloudSurfStack);
        surface_icp.setInputTarget(in_laser_cloud_surf_from_map);

        surface_icp.align(*laserCloudSurfStack);

        if (surface_icp.hasConverged ())
        {
          std::cout << "\nICP has converged, score is " << surface_icp.getFitnessScore () << std::endl;
          surface_transformation_matrix = surface_icp.getFinalTransformation ().cast<double>();
          print4x4Matrix (surface_transformation_matrix);
        }

        //NDT
        pcl::NormalDistributionsTransform<PointType, PointType> corner_ndt;
        pcl::NormalDistributionsTransform<PointType, PointType> surface_ndt;
        corner_ndt.setInputSource(laserCloudCornerStack);
        corner_ndt.setResolution(1.0);
        corner_ndt.setInputTarget(in_laser_cloud_corner_from_map);
        pcl::PointCloud<PointType>::Ptr ndt_cornerout (new pcl::PointCloud<PointType>);
        corner_ndt.align (*ndt_cornerout, init_ndtguess);
        if (corner_ndt.hasConverged ())
        {
          std::cout << "\nNDT has converged, score is " << corner_ndt.getFitnessScore () << std::endl;
          corner_ndttrans = corner_ndt.getFinalTransformation ().cast<double>();
          print4x4Matrix (corner_ndttrans);
        }

        surface_ndt.setInputSource(laserCloudSurfStack);
        surface_ndt.setResolution(1.0);
        surface_ndt.setInputTarget(in_laser_cloud_surf_from_map);
        pcl::PointCloud<PointType>::Ptr ndt_surfaceout (new pcl::PointCloud<PointType>);
        surface_ndt.align (*ndt_surfaceout, init_ndtguess);
        if (surface_ndt.hasConverged ())
        {
          std::cout << "\nNDT has converged, score is " << surface_ndt.getFitnessScore () << std::endl;
          surface_ndttrans = surface_ndt.getFinalTransformation ().cast<double>();
          print4x4Matrix (surface_ndttrans);
        }
        //std::cout<<corner_localmap.points.size()<<std::endl;
//        writer.write("0.pcd",*corner_localmap);
        //corner_localmap->width = 1;
        //corner_localmap->height = corner_localmap->points.size();
        //pcl::io::savePCDFileASCII("00.pcd",*corner_localmap);
        return 0;
    }
    void test()
    {
      corner_frame_index++;
    }
    int cornericp_findtrans(pcl::PointCloud< PointType >::Ptr in_laser_cloud_corner_from_map,
                      pcl::search::KdTree<PointType>::Ptr   kdtree_corner_from_map,
                      pcl::PointCloud< PointType >::Ptr laserCloudCornerStack
                      )
    {
//      if (corner_frame_index<100)
//      {
//        *current_corner_local_map=*current_corner_local_map+*laserCloudCornerStack;
//        //std::cerr<<"local map size is: "<<current_corner_local_map->size()<<endl;
//      }
        vector<double> fitness_score;
        vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> trans_matrix;
        pcl::PointCloud<PointType>::Ptr corner_trans (new pcl::PointCloud<PointType>);
        pcl::transformPointCloud (*laserCloudCornerStack, *corner_trans, corner_t_hist);
        corner_icp.setInputSource(corner_trans);
        Eigen::Matrix4f init_icp=Eigen::Matrix4f::Identity ();
        while(cornericp_continue)
        {
            //corner_icp.setRANSACOutlierRejectionThreshold(0.8);
//            cerr<<"iter"<<iter_num<<endl;
            corner_icp.align(*corner_trans,init_icp);
            if (corner_icp.hasConverged())
            {
                fitness_score.push_back(corner_icp.getFitnessScore ());
                Eigen::Matrix4d trans_test=corner_icp.getFinalTransformation().cast<double>();
                trans_matrix.push_back(trans_test);
//                std::cout << "score at"<<iter_num<<"is" << corner_icp.getFitnessScore ();
                if((corner_icp.getFitnessScore ()>0.05)&&(iter_num<8))
                {
                    switch (iter_num) {
                    case 0:
                        init_icp(0,3)=-0.2;
                        init_icp(1,3)=0;
                        iter_num++;
                        break;
                    case 1:
                        init_icp(0,3)=-0.2;
                        init_icp(1,3)=0.2;
                        iter_num++;
                        break;
                    case 2:
                        init_icp(0,3)=0;
                        init_icp(1,3)=0.2;
                        iter_num++;
                        break;
                    case 3:
                        init_icp(0,3)=0.2;
                        init_icp(1,3)=0.2;
                        iter_num++;
                        break;
                    case 4:
                        init_icp(0,3)=0.2;
                        init_icp(1,3)=0;
                        iter_num++;
                        break;
                    case 5:
                        init_icp(0,3)=0.2;
                        init_icp(1,3)=-0.2;
                        iter_num++;
                        break;
                    case 6:
                        init_icp(0,3)=0;
                        init_icp(1,3)=-0.2;
                        iter_num++;
                        break;
                    case 7:
                        init_icp(0,3)=-0.2;
                        init_icp(1,3)=-0.2;
                        iter_num++;
                        break;
                    default:
                        break;
                    }
                    //print4x4Matrix(corner_t_hist);
                    cornericp_continue=true;
                }
                else
                {
                    //cerr<<"end"<<endl;
                    cornericp_continue=false;
                }
            }
        }
        iter_num=0;
        Eigen::Matrix4d corner_t_curr = trans_matrix[distance(begin(fitness_score),min_element(begin(fitness_score),end(fitness_score)))];
        //        cout<<"dis "<<distance(begin(fitness_score),min_element(begin(fitness_score),end(fitness_score)))<<endl;

        cornericp_continue=true;
        fitness_score.clear();
        trans_matrix.clear();
        corner_t_hist=corner_t_curr*corner_t_hist;

        //publish odometry
        Eigen::Matrix3d rotation_matrix=Eigen::Matrix3d::Identity();
        for(size_t rows=0;rows<3;rows++)
          for(size_t cols=0;cols<3;cols++)
            rotation_matrix(rows,cols)=corner_t_hist(rows,cols);
        Eigen::Quaterniond qua=Eigen::Quaterniond(rotation_matrix);
  //      cout<<"qua: "<<qua.w()<<" "<<qua.x()<<" "<<qua.y()<<" "<<qua.z()<<endl;
        nav_msgs::Odometry corner_odometry;
        corner_odometry.header.frame_id="camera_init";
        corner_odometry.pose.pose.orientation.x=qua.x();
        corner_odometry.pose.pose.orientation.y=qua.y();
        corner_odometry.pose.pose.orientation.z=qua.z();
        corner_odometry.pose.pose.orientation.w=qua.w();

        corner_odometry.pose.pose.position.x=corner_t_hist(0,3);
        corner_odometry.pose.pose.position.y=corner_t_hist(1,3);
        corner_odometry.pose.pose.position.z=corner_t_hist(2,3);

        corner_odometry_publisher.publish(corner_odometry);
        if(*min_element(begin(fitness_score),end(fitness_score))>0.05)
        {
          std::cout << "final corner icp score at "<<corner_frame_index<<" is "<<*min_element(begin(fitness_score),end(fitness_score))<<endl;
          std::cout<<"the transformation is: "<<qua.x()<<" "<<qua.y()<<" "<<qua.z()<<" "<<qua.w()<<" "<<corner_t_hist(0,3)<<" "<<corner_t_hist(1,3)<<" "<<corner_t_hist(2,3)<<endl;
        }
        corner_frame_index++;
        return 0;
    }
    int surfaceicp_findtrans(pcl::PointCloud< PointType >::Ptr in_laser_cloud_surface_from_map,
                      pcl::search::KdTree<PointType>::Ptr   kdtree_surface_from_map,
                      pcl::PointCloud< PointType >::Ptr laserClouSurfaceStack
                      )
    {
      if (surface_frame_index<100)
      {
        *current_surface_local_map=*current_surface_local_map+*laserClouSurfaceStack;
        //std::cerr<<"local map size is: "<<current_surface_local_map->size()<<endl;
      }
      else if (surface_frame_index==100) {
        std::cerr<<"surface database size is: "<<surface_database.size()<<endl;
        initial_find(current_surface_local_map,surface_database,data_transform,initial_trans_candidates);
        cerr<<"candidate size: "<<initial_trans_candidates.size()<<" surface local map size is: "<<current_surface_local_map->size()<<endl;
        for(size_t i=0;i<initial_trans_candidates.size();i++)
        {
//          cerr<<"initial transation matrix is:"<<endl;
//          print4x4Matrixf(initial_trans_candidates[i]);
          surface_icp.setInputSource(laserClouSurfaceStack);
          surface_icp.align(*laserClouSurfaceStack,initial_trans_candidates[i]);
          if(surface_icp.hasConverged() && (surface_icp.getFitnessScore()<0.05))
          {
            surface_t_hist=surface_icp.getFinalTransformation().cast<double>();
            cerr<<"final icp score is: "<<surface_icp.getFitnessScore()<<endl;
            break;
          }
        }
      }
      else
      {
        //cerr<<"surface frame index is: "<<surface_frame_index<<endl;
        vector<double> fitness_score;
        vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> trans_matrix;
        pcl::PointCloud<PointType>::Ptr surface_trans (new pcl::PointCloud<PointType>);
        pcl::transformPointCloud (*laserClouSurfaceStack, *surface_trans, surface_t_hist);
        surface_icp.setInputSource(surface_trans);
        Eigen::Matrix4f init_icp=Eigen::Matrix4f::Identity ();
        while(surfaceicp_continue)
        {
            //corner_icp.setRANSACOutlierRejectionThreshold(0.8);
            surface_icp.align(*surface_trans,init_icp);
            if (surface_icp.hasConverged())
            {
                fitness_score.push_back(surface_icp.getFitnessScore ());
                Eigen::Matrix4d trans_test=surface_icp.getFinalTransformation().cast<double>();
                trans_matrix.push_back(trans_test);
                //std::cout << "icp score at "<<iter_num<<" is" << surface_icp.getFitnessScore ()<<std::endl;
                if((surface_icp.getFitnessScore ()>0.05)&&(iter_num<8))
                {
                  //print4x4Matrixf(init_icp);
                    switch (iter_num) {
                    case 0:
                        init_icp(0,3)=-0.2;
                        init_icp(1,3)=0;
                        iter_num++;
                        break;
                    case 1:
                        init_icp(0,3)=-0.2;
                        init_icp(1,3)=0.2;
                        iter_num++;
                        break;
                    case 2:
                        init_icp(0,3)=0;
                        init_icp(1,3)=0.2;
                        iter_num++;
                        break;
                    case 3:
                        init_icp(0,3)=0.2;
                        init_icp(1,3)=0.2;
                        iter_num++;
                        break;
                    case 4:
                        init_icp(0,3)=0.2;
                        init_icp(1,3)=0;
                        iter_num++;
                        break;
                    case 5:
                        init_icp(0,3)=0.2;
                        init_icp(1,3)=-0.2;
                        iter_num++;
                        break;
                    case 6:
                        init_icp(0,3)=0;
                        init_icp(1,3)=-0.2;
                        iter_num++;
                        break;
                    case 7:
                        init_icp(0,3)=-0.2;
                        init_icp(1,3)=-0.2;
                        iter_num++;
                        break;
                    default:
                        break;
                    }
                    //print4x4Matrix(corner_t_hist);
                    surfaceicp_continue=true;
                }
                else
                {
                    //cerr<<"end"<<endl;
                    surfaceicp_continue=false;
                }
            }
        }
        iter_num=0;
        Eigen::Matrix4d surface_t_curr = trans_matrix[distance(begin(fitness_score),min_element(begin(fitness_score),end(fitness_score)))];
        if(*min_element(begin(fitness_score),end(fitness_score))>0.05)
        {
          std::cout << "final surface icp score at "<<surface_frame_index<<" is "<<*min_element(begin(fitness_score),end(fitness_score))<<endl;
        }
        //cout<<"dis "<<distance(begin(fitness_score),min_element(begin(fitness_score),end(fitness_score)))<<endl;
        surfaceicp_continue=true;
        fitness_score.clear();
        trans_matrix.clear();
        surface_t_hist=surface_t_curr*surface_t_hist;
      }

      //publish odometry
      Eigen::Matrix3d rotation_matrix=Eigen::Matrix3d::Identity();
      for(size_t rows=0;rows<3;rows++)
        for(size_t cols=0;cols<3;cols++)
          rotation_matrix(rows,cols)=surface_t_hist(rows,cols);
      Eigen::Quaterniond qua=Eigen::Quaterniond(rotation_matrix);
//      cout<<"qua: "<<qua.w()<<" "<<qua.x()<<" "<<qua.y()<<" "<<qua.z()<<endl;
      nav_msgs::Odometry surface_odometry;
      surface_odometry.header.frame_id="camera_init";
      surface_odometry.pose.pose.orientation.x=qua.x();
      surface_odometry.pose.pose.orientation.y=qua.y();
      surface_odometry.pose.pose.orientation.z=qua.z();
      surface_odometry.pose.pose.orientation.w=qua.w();

      surface_odometry.pose.pose.position.x=surface_t_hist(0,3);
      surface_odometry.pose.pose.position.y=surface_t_hist(1,3);
      surface_odometry.pose.pose.position.z=surface_t_hist(2,3);

      surface_odometry_publisher.publish(surface_odometry);
        surface_frame_index++;
        return 0;
    }

    int icp_test(pcl::PointCloud< PointType >::Ptr in_laser_cloud_corner_from_map,
                      pcl::PointCloud< PointType >::Ptr in_laser_cloud_surf_from_map,
                      pcl::search::KdTree<PointType>::Ptr   kdtree_corner_from_map,
                      pcl::search::KdTree<PointType>::Ptr   kdtree_surf_from_map,
                      pcl::PointCloud< PointType >::Ptr laserCloudCornerStack,
                      pcl::PointCloud< PointType >::Ptr laserCloudSurfStack )
    {
        Eigen::Quaternionf quarter=Eigen::Quaternionf(0.0839445,0.0294684, -0.0326215, 0.995579).normalized();
        Eigen::Translation3f translation(26.0923-0.2, 7.49656+0.2, 0.183726);
        //25.9302 7.87992 0.156416 -0.0272266 0.0282275 -0.994084 -0.101287
        //26.0923 7.49656 0.183726 0.0294684 -0.0326215 0.995579 0.0839445 0.0329776
        Eigen::Affine3f T=translation*quarter.toRotationMatrix();
        Eigen::Matrix4f init_=Eigen::Matrix4f::Identity ();
        init_(2,3)=0.1;
        print4x4Matrixf(init_);
        Eigen::Matrix4f init_guess=T.matrix();

        std::string cornerpath="/home/beihai/catkin_ws/src/loam_livox/pcd/laserCloudSurfStack/10118.pcd";
        pcl::PointCloud<PointType>::Ptr corner_curr (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr corner_trans (new pcl::PointCloud<PointType>);

        pcl::io::loadPCDFile<PointType> (cornerpath,*corner_curr);
        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> corner_icp;
        corner_icp.setInputTarget(in_laser_cloud_surf_from_map);
        corner_icp.setInputSource(corner_curr);
        corner_icp.setMaxCorrespondenceDistance(0.6);//points >0.5m ignored
        corner_icp.setMaximumIterations (50);
        corner_icp.setTransformationEpsilon (1e-8);
        corner_icp.setEuclideanFitnessEpsilon (0.05);
        corner_icp.align(*corner_trans,init_guess);
        if (corner_icp.hasConverged ())
        {
            std::cout << "\nICP has converged, score is " << corner_icp.getFitnessScore () << std::endl;
            print4x4Matrix(corner_icp.getFinalTransformation ().cast<double>());
        }

        return 0;
    }
    int find_initial()
    {
    
    }

    void corner_callback(const sensor_msgs::PointCloud2ConstPtr& input)
    {
      pcl::PointCloud<PointType>::Ptr corner_frame=boost::make_shared<pcl::PointCloud<PointType>>();// (new pcl::PointCloud<PointTypeIO>);
      pcl::fromROSMsg(*input,*corner_frame);
      cornericp_findtrans(cornermap,cornermap_kdtree,corner_frame);
      //std::cerr<<"corner frame id is: "<<corner_frame_index<<" corner map size is:"<<cornermap->size()<<endl;
    }
    void surface_callback(const sensor_msgs::PointCloud2ConstPtr& input)
    {
      pcl::PointCloud<PointType>::Ptr surface_frame=boost::make_shared<pcl::PointCloud<PointType>>();// (new pcl::PointCloud<PointTypeIO>);
      pcl::fromROSMsg(*input,*surface_frame);
      surfaceicp_findtrans(surfacemap,surfacemap_kdtree,surface_frame);
      //std::cerr<<"surface frame id is: "<<surface_frame_index<<" surface map size is:"<<surfacemap->size()<<endl;
    }
    Local_map()
    {
      pcl::io::loadPCDFile<PointType> ("/home/beihai/data/pcd/map/CYT02_cornermap01.pcd", *cornermap);
      cornermap_kdtree->setInputCloud(cornermap);
      corner_sub= nh.subscribe<sensor_msgs::PointCloud2>("cornerstack",10000,&Local_map::corner_callback,this);

      pcl::io::loadPCDFile<PointType> ("/home/beihai/data/pcd/map/CYT02_surfacemap01.pcd", *surfacemap);
      surfacemap_kdtree->setInputCloud(surfacemap);
      surface_sub= nh.subscribe<sensor_msgs::PointCloud2>("surfacestack",10000,&Local_map::surface_callback,this);

      surface_odometry_publisher=nh.advertise<nav_msgs::Odometry>("/surface_odometry",1000);
      corner_odometry_publisher=nh.advertise<nav_msgs::Odometry>("/corner_odometry",1000);

      database_initialize(surface_vocabulary,surface_database);
      flann::load_from_file(data_transform,"/home/beihai/data/vocab/transform_data.h5","transform_data");
      cerr<<"map,kdtree,vocabulary loaded"<<endl;

      cout<<"transform: "<<data_transform.rows<<" database size is: "<<surface_database.size()<<" vocab size is: "<<surface_vocabulary.size()<<endl;
      //set surface icp
      surface_icp.setInputTarget(surfacemap);
      surface_icp.setSearchMethodTarget(surfacemap_kdtree,true);
      surface_icp.setMaxCorrespondenceDistance(0.6);//points >0.5m ignored
      surface_icp.setMaximumIterations (50);
      surface_icp.setTransformationEpsilon (1e-8);
      surface_icp.setEuclideanFitnessEpsilon (0.05);
      std::cerr<<"surface icp has initialized"<<endl;
      //set corner icp
      corner_icp.setInputTarget(cornermap);
      corner_icp.setSearchMethodTarget(cornermap_kdtree,true);
      corner_icp.setMaxCorrespondenceDistance(0.6);//points >0.5m ignored
      corner_icp.setMaximumIterations (50);
      corner_icp.setTransformationEpsilon (1e-8);
      corner_icp.setEuclideanFitnessEpsilon (0.05);
      std::cerr<<"corner icp has initialized"<<endl;

    }
};
