#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <open3d/Open3D.h>

using namespace open3d;

// Backend interface: defines what all backends must implement
class IPosegraphBackend {
public:
    virtual ~IPosegraphBackend() = default;

    virtual void AddNode(const core::Tensor& trans) = 0;
    virtual void AddOdometryEdge(const core::Tensor& odometry,
                                 const core::Tensor& info,
                                 int i, int j,
                                 bool uncertain) = 0;
    virtual void Optimize() = 0;
    virtual size_t CountNodes() const = 0;
    virtual void Save(const std::string& path) const = 0;
};

#ifdef USE_GTSAM
class imuNoise;

class GTSAMPosegraphBackend : public IPosegraphBackend {
public:
    using Matrix6d = Eigen::Matrix<double,6,6>;

    GTSAMPosegraphBackend(const open3d::core::Tensor& initial_pose_tensor,
                          std::shared_ptr<imuNoise> imu = nullptr)
        : imu_(imu)
    {
        using namespace gtsam;
        Eigen::Matrix4d initial_pose = TensorToMatrix4d(initial_pose_tensor);

        pose_noise_ = noiseModel::Diagonal::Sigmas((Vector(6) << Vector6::Constant(0.0001)).finished());
        vel_noise_ = noiseModel::Isotropic::Sigma(3,0.1);
        bias_noise_ = noiseModel::Isotropic::Sigma(6,10);

        Pose3 pose0(initial_pose);
        graph_.add(PriorFactor<Pose3>(Symbol('x',0), pose0, pose_noise_));
        initial_.insert(Symbol('x',0), pose0);

        if (imu_) {
            Vector3 vel0 = Vector3::Zero();
            bias_ = imu_->get_bias();
            params_ = imu_->get_params();

            graph_.add(PriorFactor<Vector3>(Symbol('v',0), vel0, vel_noise_));
            graph_.add(PriorFactor<imuBias::ConstantBias>(Symbol('b',0), bias_, bias_noise_));
            initial_.insert(Symbol('v',0), vel0);
            initial_.insert(Symbol('b',0), bias_);
        }
    }

    void AddNode(const open3d::core::Tensor& trans) override {
        using namespace gtsam;

        Eigen::Matrix4d pose_mat = TensorToMatrix4d(trans);
        Pose3 pose(pose_mat);

        int next_index = CountNodes();
        if (!initial_.exists(Symbol('x', next_index))) {
            initial_.insert(Symbol('x', next_index), pose);
        }
    }

    void AddOdometryEdge(const open3d::core::Tensor& odometry,
                         const open3d::core::Tensor& info,
                         int i, int j,
                         bool uncertain) override
    {
        using namespace gtsam;

        double coef = uncertain ? 100.0 : 1.0;
        Eigen::Matrix6d info_mat = TensorToMatrix6d(info);
        Eigen::Matrix6d cov = (info_mat + 1e-6 * Eigen::Matrix6d::Identity()).inverse();

        auto odom_noise = noiseModel::Gaussian::Covariance(coef * cov);
        auto robust_noise = noiseModel::Robust::Create(
            noiseModel::mEstimator::Huber::Create(1.0),
            odom_noise);

        Eigen::Matrix4d odom_mat = TensorToMatrix4d(odometry);
        Pose3 odom_pose(odom_mat.inverse());

        graph_.add(BetweenFactor<Pose3>(Symbol('x',i), Symbol('x',j), odom_pose, robust_noise));
    }

    void AddImuEdge(const std::vector<Eigen::Vector4d>& accel,
                    const std::vector<Eigen::Vector4d>& gyro,
                    int i, int j)
    {
        using namespace gtsam;

        if (!imu_) {
            throw std::runtime_error("IMU configuration not set.");
        }

        double prev = gyro.front()(0);
        auto preint = std::make_shared<PreintegratedCombinedMeasurements>(params_, bias_);

        for (size_t k=1; k<gyro.size(); ++k) {
            Vector3 g = gyro[k].tail<3>();
            Vector3 a = accel[k].tail<3>();
            double dt = gyro  - prev;
            if (dt > 0) {
                preint->integrateMeasurement(a, g, dt);
                prev = gyro ;
            }
        }

        Point3 p0 = initial_.at<Pose3>(Symbol('x',i)).translation();
        Point3 p1 = initial_.at<Pose3>(Symbol('x',j)).translation();
        double dt = gyro.back()(0) - gyro.front()(0);
        Vector3 v = (p1 - p0) / dt;

        graph_.add(CombinedImuFactor(Symbol('x',i), Symbol('v',i),
                                     Symbol('x',j), Symbol('v',j),
                                     Symbol('b',i), Symbol('b',j),
                                     *preint));

        initial_.insert(Symbol('v',j), v);
        initial_.insert(Symbol('b',j), bias_);
    }

    void Optimize() override {
        using namespace gtsam;

        LevenbergMarquardtParams params;
        params.setVerbosity("TERMINATION");
        params.setMaxIterations(500);
        params.setRelativeErrorTol(1e-10);
        params.setAbsoluteErrorTol(1e-10);

        LevenbergMarquardtOptimizer optimizer(graph_, initial_, params);
        result_ = optimizer.optimize();
    }

    size_t CountNodes() const override {
        using namespace gtsam;
        size_t count = 0;
        for (const auto& key : initial_.keys()) {
            if (Symbol(key).chr() == 'x') {
                ++count;
            }
        }
        return count;
    }

    void Save(const std::string& path) const override {
        auto pg = ConvertToOpen3D();
        open3d::io::WritePoseGraph(path, pg);
    }

    open3d::pipelines::registration::PoseGraph ConvertToOpen3D() const {
        using namespace gtsam;
        using namespace open3d::pipelines::registration;

        PoseGraph pg;

        for (const auto& key : result_.keys()) {
            if (Symbol(key).chr() == 'x') {
                Pose3 pose = result_.at<Pose3>(key);
                pg.nodes_.emplace_back(pose.matrix());
            }
        }

        for (size_t i=0; i<graph_.size(); ++i) {
            auto factor = boost::dynamic_pointer_cast<BetweenFactor<Pose3>>(graph_[i]);
            if (factor) {
                auto k1 = Symbol(factor->key1()).index();
                auto k2 = Symbol(factor->key2()).index();
                Eigen::Matrix4d odometry = factor->measured().matrix();
                Eigen::Matrix6d info = Eigen::Matrix6d::Identity();
                bool uncertain = std::abs(int(k1)-int(k2)) != 1;

                pg.edges_.emplace_back(k1, k2, odometry, info, uncertain);
            }
        }

        return pg;
    }

private:
    static Eigen::Matrix4d TensorToMatrix4d(const open3d::core::Tensor& t) {
        O3D_ASSERT(t.GetShape() == std::vector<int64_t>{4,4});
        Eigen::Matrix4d m;
        t.ToFlatVector<double>().asEigenMatrix().resize(4,4);
        return Eigen::Map<const Eigen::Matrix4d>(t.GetDataPtr<double>());
    }

    static Eigen::Matrix6d TensorToMatrix6d(const open3d::core::Tensor& t) {
        O3D_ASSERT(t.GetShape() == std::vector<int64_t>{6,6});
        Eigen::Matrix6d m;
        t.ToFlatVector<double>().asEigenMatrix().resize(6,6);
        return Eigen::Map<const Eigen::Matrix6d>(t.GetDataPtr<double>());
    }

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_;
    gtsam::Values result_;

    gtsam::SharedNoiseModel pose_noise_;
    gtsam::SharedNoiseModel vel_noise_;
    gtsam::SharedNoiseModel bias_noise_;

    gtsam::imuBias::ConstantBias bias_;
    gtsam::PreintegrationParams params_;

    std::shared_ptr<imuNoise> imu_;
};

#endif 

// Example: Open3D backend
// always implicit Tensor conversions
class Open3dPosegraphBackend : public IPosegraphBackend {
public:
    explicit Open3dPosegraphBackend(const core::Tensor& initial_pose) {
        pose_graph_.nodes_.emplace_back(core::eigen_converter::TensorToEigenMatrixXd(initial_pose));
    }

    void AddNode(const core::Tensor& trans) override {
        pose_graph_.nodes_.emplace_back(core::eigen_converter::TensorToEigenMatrixXd(trans));
    }

    void AddOdometryEdge(const core::Tensor& odometry,
                         const core::Tensor& info,
                         int i, int j,
                         bool uncertain) override {
        pose_graph_.edges_.emplace_back(i, j, 
            core::eigen_converter::TensorToEigenMatrixXd(odometry), 
            core::eigen_converter::TensorToEigenMatrixXd(info), uncertain);
    }

    void Optimize() override {

        double max_correspondence_distance = 0.01;
        double preference_loop_closure = 0.2;

        auto method = open3d::pipelines::registration::GlobalOptimizationLevenbergMarquardt();
        auto criteria = open3d::pipelines::registration::GlobalOptimizationConvergenceCriteria();
        auto option = open3d::pipelines::registration::GlobalOptimizationOption(
            max_correspondence_distance,
            0.25, // edge prune threshold
            preference_loop_closure,
            0 // reference node
        );

        open3d::pipelines::registration::GlobalOptimization(
            pose_graph_,
            method,
            criteria,
            option
        );
    }

    size_t CountNodes() const override {
        return pose_graph_.nodes_.size();
    }

    void Save(const std::string& path) const override {
        open3d::io::WritePoseGraph(path, pose_graph_);
    }

    // Optional: if you want to get the underlying Open3D PoseGraph
    const open3d::pipelines::registration::PoseGraph& GetPoseGraph() const {
        return pose_graph_;
    }

private:
    open3d::pipelines::registration::PoseGraph pose_graph_;
};

// Main templated Posegraph class
template <typename BackendT>
class Posegraph {
public:
    template<typename... Args>
    explicit Posegraph(Args&&... args)
        : backend_(std::make_unique<BackendT>(std::forward<Args>(args)...))
    {}

    void AddNode(const core::Tensor trans) {
        backend_->AddNode(trans);
    }

    void AddOdometryEdge(const core::Tensor odometry,
                         const core::Tensor info,
                         int i, int j,
                         bool uncertain) {
        backend_->AddOdometryEdge(odometry, info, i, j, uncertain);
    }

    void Optimize() {
        backend_->Optimize();
    }

    size_t CountNodes() const {
        return backend_->CountNodes();
    }

    void Save(const std::string& path) const {
        backend_->Save(path);
    }

    const open3d::pipelines::registration::PoseGraph& GetPoseGraph(){
        return backend_->GetPoseGraph();
    }

    // If you want access to the backend instance itself
    BackendT& Backend() { return *backend_; }

private:
    std::unique_ptr<BackendT> backend_;
};

template class Posegraph<Open3dPosegraphBackend>;