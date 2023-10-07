#include "utility.h"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))
typedef PointXYZIRPYT PointTypePose;

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
{
	pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

	int cloudSize = cloudIn->size();
	cloudOut->resize(cloudSize);

	Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(4)
	for (int i = 0; i < cloudSize; ++i)
	{
		const auto &pointFrom = cloudIn->points[i];
		cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
		cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
		cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
		cloudOut->points[i].intensity = pointFrom.intensity;
	}
	return cloudOut;
}

void keyframePose2Text(pcl::PointCloud<PointTypePose>::Ptr keyframePoses, const std::string &save_file, int type = 1)
{
	ofstream outFile;
	outFile.open(save_file, ios::out);
	for (size_t i = 0; i < keyframePoses->size(); i++)
	{
		if (type == 0)
		{
			outFile << i + 1 << ',' << keyframePoses->points[i].x << ',' << keyframePoses->points[i].y << ',' << keyframePoses->points[i].z << ','
					<< keyframePoses->points[i].roll << ',' << keyframePoses->points[i].pitch << ',' << keyframePoses->points[i].yaw << endl;
		}
		Eigen::AngleAxisf roll(keyframePoses->points[i].roll, Eigen::Vector3f::UnitX());
		Eigen::AngleAxisf pitch(keyframePoses->points[i].pitch, Eigen::Vector3f::UnitY());
		Eigen::AngleAxisf yaw(keyframePoses->points[i].yaw, Eigen::Vector3f::UnitZ());
		Eigen::Quaternionf quat = yaw * pitch * roll;
		if (type == 1)
		{
			outFile << i + 1 << ',' << keyframePoses->points[i].x << ',' << keyframePoses->points[i].y << ',' << keyframePoses->points[i].z << ','
					<< quat.w() << ',' << quat.x() << ',' << quat.y() << ',' << quat.z() << endl;
		}
		else if (type == 2)
		{
			Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
			mat.topLeftCorner(3,3) = quat.toRotationMatrix();
			mat.topRightCorner(3,1) = Eigen::Vector3f(keyframePoses->points[i].x, keyframePoses->points[i].y, keyframePoses->points[i].z);
			outFile << i + 1 << ',' << mat(0, 0) << ',' << mat(0, 1) << ',' << mat(0, 2) << ','
					<< mat(1, 0) << ',' << mat(1, 1) << ',' << mat(1, 2) << ','
					<< mat(2, 0) << ',' << mat(2, 1) << ',' << mat(2, 2) << ',' << endl;
		}
	}
	outFile.close();
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "lio_sam");
	ParamServer PS;
	std::string keyFrameMapsPath = std::getenv("HOME") + PS.savePCDDirectory;
	pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>);
	pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>);
	vector<pcl::PointCloud<PointType>::Ptr> cloudKeyFrames;

	if (pcl::io::loadPCDFile<PointTypePose>(keyFrameMapsPath + "/transformations.pcd", *cloudKeyPoses6D) == -1)
	{
		cout << "Read file fail!\n"
			 << endl;
		return -1;
	}
	cout << "总共有" << cloudKeyPoses6D->size() << "个关键帧!" << endl;
	cout << "读取关键帧中......" << endl;
	keyframePose2Text(cloudKeyPoses6D, keyFrameMapsPath + "/poses_keyframe.txt", 2);

	cloudKeyFrames.reserve(cloudKeyPoses6D->size());
	for (size_t i = 0; i < cloudKeyPoses6D->size(); i++)
	{
		pcl::PointCloud<PointType>::Ptr tmpKeyFrames(new pcl::PointCloud<PointType>);
		if (pcl::io::loadPCDFile<PointType>(keyFrameMapsPath + "/keyFrameCloud/keyFrame_" + std::to_string(i) + ".pcd", *tmpKeyFrames) == -1)
		{
			cout << "Read file fail! " << i << endl;
			return -1;
		}
		cloudKeyFrames.push_back(tmpKeyFrames);
	}

	cout << "处理中......" << endl;
	for (int i = 0; i < (int)cloudKeyFrames.size(); i++)
	{
		*cloudKeyFrames[i] = *transformPointCloud(cloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
		*globalMapCloud += *cloudKeyFrames[i];
	}

	pcl::io::savePCDFileBinary(keyFrameMapsPath + "AllkeyFramesMap.pcd", *globalMapCloud);
	cout << "****************************************************" << endl;
	cout << "Saving map to pcd files completed!" << endl;
	return 0;
}
