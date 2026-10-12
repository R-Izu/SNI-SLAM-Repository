// 自動生成：STSL bd23507 の point-sam/examples/plane_seg.cpp の 1〜113 行目をそのまま写し、main を替えた
#include <iostream>
#include <string>
#include "utils/file_manager.h"
#include "global_definition/global_definition.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "glog/logging.h"
#include "segmentation/octree_seg.h"
#include "utils/tic_toc.h"
#include "utils/config.h"
#include "segmentation/voxel_clustering.h"
#include <pcl/filters/voxel_grid.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/features/normal_3d_omp.h>
#include "utils/tqdm.hpp"

#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

bool fileExists(const std::string &filePath) {
    return fs::exists(filePath);
}

void process(std::string config_path, bool save_oc_seg = true, bool save_ground = true) {
    config::readParameters(config_path);
    std::string pcd_folder = config::pcd_folder;
    std::string output_ground_folder = config::ground_folder;
    std::string output_ocseg_folder = config::oc_seg_folder;

    fs::path dir1 = output_ground_folder;
    if (!fs::exists(dir1)) {
        if (fs::create_directory(dir1)) {
            std::cout << "output_ground_folder Directory created successfully." << std::endl;
        } else {
            std::cerr << "Failed to create directory." << std::endl;
            return;
        }
    } else {
        std::cout << "Directory already exists." << std::endl;
    }
    fs::path dir2 = output_ocseg_folder;
    if (!fs::exists(dir2)) {
        if (fs::create_directory(dir2)) {
            std::cout << "output_ocseg_folder Directory created successfully." << std::endl;
        } else {
            std::cerr << "Failed to create directory." << std::endl;
            return;
        }
    } else {
        std::cout << "Directory already exists." << std::endl;
    }
    TicToc ticToc;
    std::vector<std::string> pcd_files;
    std::vector<std::string> pcd_names;
    for (const auto &entry : fs::directory_iterator(pcd_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pcd") {
            pcd_files.push_back(entry.path().string());
            pcd_names.push_back(entry.path().filename().string());
        }
    }
    int fileCount = pcd_files.size();
    for (int i : tq::trange(fileCount)) { // 使用tqdm样式的进度条
        std::string pcd_file = pcd_files[i];
        std::string pcd_name = pcd_names[i];

            string output_ocseg_file = output_ocseg_folder + "/" + pcd_name;
            string output_ground_file = output_ground_folder + "/" + pcd_name;

            if (!fileExists(output_ocseg_file) && !fileExists(output_ground_file)) {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1) {
                    PCL_ERROR("Couldn't read file %s\n", pcd_file.c_str());
                    continue;
                }

                ticToc.tic();
                OctreeSeg oc_seg(cloud);
                oc_seg.segmentation();
//                LOG(INFO) << "segmentation time: " << ticToc.toc() << " ms";
                pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_ocs = oc_seg.getLabeledRGBCloud();
                pcl::PointCloud<pcl::PointXYZRGBL>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
                pcl::PointCloud<pcl::PointXYZ>::Ptr oc_seg_ground_cloud = oc_seg.ground_cloud;

                if (oc_seg_ground_cloud->empty()) {
//                    std::cerr << "Point cloud is empty." << std::endl;
                    pcl::PointXYZ point;
                    point.x = 0.0;
                    point.y = 0.0;
                    point.z = 0.0;
                    oc_seg_ground_cloud->push_back(point);
                }

                if (save_oc_seg) {
                    string output_ocseg_file = output_ocseg_folder + "/" + pcd_name;
                    pcl::io::savePCDFileASCII(output_ocseg_file, *cloud_ocs);
                }
                if (save_ground) {
                    string output_ground_file = output_ground_folder + "/" +pcd_name;
                    pcl::io::savePCDFileASCII(output_ground_file, *oc_seg_ground_cloud);
                }
                // visualizeCloud<pcl::PointXYZRGBL>(cloud_ocs, pcd_file);
            } else {
                LOG(INFO) << "Files already exist, skipping: " << output_ocseg_file << ", " << output_ground_file;
            }
        }
    }


int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "usage: r46_plane_seg <pointsam config>" << std::endl; return 2; }
    process(argv[1], true, true);
    return 0;
}
