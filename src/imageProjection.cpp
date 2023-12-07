#include "utility.h"
#include "lio_sam/cloud_info.h"

struct VelodynePointXYZIRT
{
// #define VEL_TIMESTAMP_TYPE float
#define VEL_TIMESTAMP_TYPE double
// #define VEL_TIMESTAMP_FIELD time
#define VEL_TIMESTAMP_FIELD timestamp

    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    VEL_TIMESTAMP_TYPE VEL_TIMESTAMP_FIELD;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint16_t, ring, ring) (VEL_TIMESTAMP_TYPE, VEL_TIMESTAMP_FIELD, VEL_TIMESTAMP_FIELD)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint32_t, t, t) (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring) (std::uint16_t, noise, noise) (std::uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    // 订阅原始激光点云
    ros::Subscriber subLaserCloud;
    ros::Publisher pubLaserCloud;

    // 发布当前帧校正后点云，有效点
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    // imu数据队列（原始数据，转lidar系下）
    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    // imu里程计队列
    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    // 激光点云数据队列
    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    // 队列front帧，作为当前处理帧点云
    sensor_msgs::PointCloud2 currentCloudMsg;

    // 当前激光帧起止时刻间对应的imu数据，计算相对于起始时刻的旋转增量，以及时时间戳；用于插值计算当前激光帧起止时间范围内，每一时刻的旋转姿态
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    // 当前帧原始激光点云
    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    // 当期帧运动畸变校正之后的激光点云
    pcl::PointCloud<PointType>::Ptr fullCloud;
    // 从fullCloud中提取有效点
    pcl::PointCloud<PointType>::Ptr extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    // 当前激光帧起止时刻对应imu里程计位姿变换，该变换对应的平移增量；用于插值计算当前激光帧起止时间范围内，每一时刻的位置
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    // 当前帧激光点云运动畸变校正之后的数据，包括点云数据、初始位姿、姿态角等，发布给featureExtraction进行特征提取
    lio_sam::cloud_info cloudInfo;
    // 当前帧起始时刻
    double timeScanCur;
    // 当前帧结束时刻
    double timeScanEnd;
    // 当前帧header，包含时间戳信息
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;


public:
    ImageProjection():
    deskewFlag(0)
    {
        // 订阅原始lidar数据
        // ros::TransportHints().tcpNoDelay() :被用于指定hints，确定传输层的作用话题的方式:无延迟的TCP传输方式
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅imu里程计: 来自IMUPreintegration(IMUPreintegration.cpp中的类IMUPreintegration)发布的积分计算得到的每时刻imu位姿
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅原始lidar数据
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        // 发布当前激光帧运动畸变校正后的点云，有效点
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        // 发布当前激光帧运动畸变校正后的点云信息
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        // 分配内存
        allocateMemory();
        // 重置部分参数
        resetParameters();

        // setVerbosityLevel用于设置控制台输出的信息。(pcl::console::L_ALWAYS)不会输出任何信息;L_DEBUG会输出DEBUG信息;L_ERROR会输出ERROR信息
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        //根据params.yaml中给出的N_SCAN Horizon_SCAN参数值分配内存
        //用智能指针的reset方法在构造函数里面进行初始化
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        //cloudinfo是msg文件下自定义的cloud_info消息，对其中的变量进行赋值操作
        //(int size, int value):size-要分配的值数,value-要分配给向量名称的值
        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection,
        // 初始全部用FLT_MAX 填充，因此后文函数projectPointCloud中有一句if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) continue;
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}

    /**
     * 订阅原始imu数据
     * 1、imu原始测量数据转换到lidar系，加速度、角速度、RPY
     */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        // imu原始测量数据转换到lidar系，加速度、角速度、RPY
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        // 上锁，添加数据的时候队列不可用
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    /**
     * 订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿.
     */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    /**
     * 订阅原始lidar数据
     * 1、添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
     * 2、当前帧起止时刻对应的imu数据、imu里程计数据处理
     *   imu数据：
     *   1) 遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
     *   2) 用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
     *   imu里程计数据：
     *   1) 遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
     *   2) 用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
     * 3、当前帧激光点云运动畸变校正
     *   1) 检查激光点距离、扫描线是否合规
     *   2) 激光运动畸变校正，保存激光点
     * 4、提取有效激光点，存extractedCloud
     * 5、发布当前帧校正后点云，有效点
     * 6、重置参数，接收每帧lidar数据都要重置这些参数
     */
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 当前帧起止时刻对应的imu数据、imu里程计数据处理
        if (!deskewInfo())
            return;

        // 当前帧激光点云运动畸变校正
        // 1、检查激光点距离、扫描线是否合规
        // 2、激光运动畸变校正，保存激光点
        projectPointCloud();

        // 提取有效激光点，存extractedCloud
        cloudExtraction();

        // 发布当前帧校正后点云，有效点
        publishClouds();

        // 重置参数，接收每帧lidar数据都要重置这些参数
        resetParameters();
    }

    /**
     * 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
     */
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        // 取出激光点云队列中最早的一帧
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE)
        {
            // 转换成pcl点云格式 形参: (in,out)
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
            auto offset_timestamp = laserCloudIn->points[0].VEL_TIMESTAMP_FIELD;
            for (size_t i = 0; i < laserCloudIn->size(); i++)
            {
                laserCloudIn->points[i].VEL_TIMESTAMP_FIELD -= offset_timestamp;
            }
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.VEL_TIMESTAMP_FIELD = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        // 当前帧起始时刻
        timeScanCur = cloudHeader.stamp.toSec();
        // 当前帧结束时刻，注：点云中激光点的time记录相对于当前帧第一个激光点的时差，第一个点time=0
        timeScanEnd = timeScanCur + laserCloudIn->points.back().VEL_TIMESTAMP_FIELD;

        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        // 检查ring这个field是否存在，ring代表线数，0是最下面那条
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        // 检查是否存在time通道
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t" || field.name == "timestamp")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    /**
     * 当前帧起止时刻对应的imu数据、imu里程计数据处理
     */
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // 要求imu数据队列中时间覆盖当前帧激光数据，否则不往下处理了
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        // 旋转去畸变参数计算 
        imuDeskewInfo();

        // imu里程计去畸变参数计算
        odomDeskewInfo();

        return true;
    }

    /**
     * 当前帧对应imu数据处理
     * 1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
     * 2、用角速度、时间积分，计算每一imu时刻相对于初始时刻的旋转量，初始时刻旋转设为0
     * 注：imu数据都已经转换到lidar系下了
     */
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        // 从imu队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;
        // 遍历当前激光帧起止时刻（前后扩展0.01s）之间的imu数据
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            // 提取imu姿态角RPY，作为当前lidar帧初始姿态角
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

             // 超过当前激光帧结束时刻0.01s，结束
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            // 第一帧imu旋转角初始化
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            // 提取imu角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            // TODO: 改中值积分??
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;
        // 没有合规的imu数据
        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    /**
     * 当前帧对应imu里程计处理
     * 1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
     * 2、用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
     * 注：imu数据都已经转换到lidar系下了
     */
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;
        // 从imu里程计队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        // 要求必须有当前激光帧时刻之前的里程计数据
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        // 提取当前激光帧起始时刻的imu里程计
        nav_msgs::Odometry* startOdomMsg = nullptr;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = &odomQueue[i];
            // 找到第一个大于等于初始时刻的odom
            // timeScanCur，即当前帧点云的初始时刻
            if (ROS_TIME(startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        // 提取imu里程计姿态角
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg->pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        // 将当前激光帧起始时刻的imu里程计，做为当前帧lidar位姿的初始估计，后面用于mapOptmization
        cloudInfo.initialGuessX = startOdomMsg->pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg->pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg->pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;
        // 如果当前激光帧结束时刻之后没有imu里程计数据，返回
        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        // 提取当前激光帧结束时刻的imu里程计
        nav_msgs::Odometry* endOdomMsg = nullptr;
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = &odomQueue[i];
            // 找到第一个大于等于一帧激光结束时刻的odom
            if (ROS_TIME(endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }
        // 如果起止时刻对应imu里程计的方差不等，返回
        if (int(round(startOdomMsg->pose.covariance[0])) != int(round(endOdomMsg->pose.covariance[0])))
            return;
#if 1
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg->pose.pose.position.x, startOdomMsg->pose.pose.position.y, startOdomMsg->pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg->pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg->pose.pose.position.x, endOdomMsg->pose.pose.position.y, endOdomMsg->pose.pose.position.z, roll, pitch, yaw);

        // 起止时刻imu里程计的相对变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 相对变换，提取增量平移、旋转（欧拉角）
        float rollIncre, pitchIncre, yawIncre;
 
        // 给定的转换中，提取XYZ以及欧拉角,通过tranBt 获得增量值  后续去畸变用到
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);
#endif
        odomDeskewFlag = true;
    }

    /**
     * 在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
     */
    void findRotation(const double& pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        // 查找当前时刻在imuTime下的索引
        int imuPointerFront = 0;
        //imuDeskewInfo中，对imuPointerCur进行计数(计数到超过当前激光帧结束时刻0.01s)
        while (imuPointerFront < imuPointerCur)
        {
            //imuTime在imuDeskewInfo（deskewInfo中调用，deskewInfo在cloudHandler中调用）被赋值，从imuQueue中取值
            //pointTime为当前时刻，由此函数的函数形参传入,要找到imu积分列表里第一个大于当前时间的索引
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        // 设为离当前时刻最近的旋转增量
        // 如果计数为0或该次imu时间戳小于了当前时间戳(异常退出)
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // 前后时刻插值计算当前时刻的旋转增量
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    /**
     * 在当前激光帧起止时间范围内，计算某一时刻的平移（相对于起始时刻的平移增量）
     */
    void findPosition(const double& relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // 如果传感器移动速度较慢，例如人行走的速度，那么可以认为激光在一帧时间范围内，平移量小到可以忽略不计
        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.
#if 1
        if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
            return;

        float ratio = relTime / (timeScanEnd - timeScanCur);

        *posXCur = ratio * odomIncreX;
        *posYCur = ratio * odomIncreY;
        *posZCur = ratio * odomIncreZ;
#endif
    }

    /**
     * 激光运动畸变校正
     * 利用当前帧起止时刻之间的imu数据计算旋转增量，imu里程计数据计算平移增量，进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿
     */
    PointType deskewPoint(PointType *point, const double& relTime)
    {
         //这个来源于上文的时间戳通道和imu可用判断，没有或是不可用则返回点
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        // timeScanCur是当前帧点云的初始时刻
        // relTime是当前激光点相对于激光帧起始时刻的时间，pointTime则是当前激光点的时间戳
        double pointTime = timeScanCur + relTime;

        // 求解关于旋转和平移的运动畸变
        // 计算某一时刻的旋转（相对于建图起始时刻的旋转增量）
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        // 计算某一时刻的平移（相对于一帧激光内起始点时刻的平移增量）
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        // 这里的firstPointFlag来源于resetParameters函数，而resetParameters函数每次ros调用cloudHandler都会启动
        // 第一个点的位姿增量（0），求逆
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;//改成false以后，同一帧激光数据的下一次就不执行了
        }

        // transform points to start
        // 将points转到第一个激光点的坐标系下(去畸变)
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        // 当前时刻激光点与第一个激光点的运动增量 = Tij=Twi^-1 * Twj，w为本次scan第一个点的lidar坐标系
        Eigen::Affine3f transBt = transStartInverse * transFinal;

	    // 去畸变补偿，根据lidar位姿变换 Tij，修正点云位置: Tij * Pj
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    /**
     * 当前帧激光点云运动畸变校正
     * 1、检查激光点距离、扫描线是否合规
     * 2、激光运动畸变校正，保存激光点
     */
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            //laserCloudIn就是原始的点云话题中的数据
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            // 距离检查
            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            // 扫描线检查
            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            // 扫描线如果有降采样，跳过采样的扫描线这里要跳过
            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            // 已经存过该点，不再处理
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            // 点云去畸变
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].VEL_TIMESTAMP_FIELD);

            // 矩阵存激光点的距离
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 转换成一维索引，存校正之后的激光点
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    /**
     * 提取有效激光点，存extractedCloud
     */
    void cloudExtraction()
    {
        // 有效激光点数量
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 提取特征的时候，每一行的前5个和最后5个不考虑
            // 记录每根扫描线起始第5个激光点在一维数组中的索引
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    // 记录激光点对应的Horizon_SCAN方向上的索引
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    // 激光点距离
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    // 加入有效激光点
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            // 记录每根扫描线倒数第5个激光点在一维数组中的索引
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }

    /**
     * 发布当前帧校正后点云，有效点
    */
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        //publishCloud在utility.h头文件中,需要传入发布句柄pubExtractedCloud，提取出的有效点云，该帧时间戳，
        //pubExtractedCloud定义在构造函数中，用来发布去畸变的点云.
        //extractedCloud主要在cloudExtraction中被提取，点云被去除了畸变，
        //另外每行头五个和后五个不要（仍然被保存，但是之后在提取特征时不要,因为要根据前后五个点算曲率）
        //cloudHeader.stamp 来源于currentCloudMsg,cloudHeader在cachePointCloud中被赋值currentCloudMsg.header
        //而currentCloudMsg是点云队列cloudQueue中提取的
        //lidarFrame:在utility.h中被赋为base_link,
        //在publishCloud函数中，tempCloud.header.frame_id="base_link"(lidarFrame)
        //之后用发布句柄pubExtractedCloud来发布去畸变的点云
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        //发布自定义cloud_info信息
        pubLaserCloudInfo.publish(cloudInfo);

        //pubExtractedCloud发布的只有点云信息，而pubLaserCloudInfo发布的为自定义的很多信息
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");
    //对于一些只订阅一个话题的简单节点来说，我们使用ros::spin()进入接收循环，
    //每当有订阅的话题发布时，进入回调函数接收和处理消息数据。
    //但是更多的时候，一个节点往往要接收和处理不同来源的数据，并且这些数据的产生频率也各不相同，
    //当我们在一个回调函数里耗费太多时间时，会导致其他回调函数被阻塞，导致数据丢失。
    //这种场合需要给一个节点开辟多个线程，保证数据流的畅通。

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
