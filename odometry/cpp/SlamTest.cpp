#include "CONFIG.h"
#include "TestDataset.h"
#include "pose_graph.h"
#include "TestUtil.h"

#include <nlohmann/json.hpp>

void computeMetricsSlam(SubDataset data, core::Device device, SlamMethod method, t::pipelines::odometry::Method o_method){

    float depthscale = 5000.0;
    float depth_max = 7.0;
    
    auto dataset = Tum_dataset(data);
    auto Trajectory = std::vector<core::Tensor>();

    t::pipelines::slam::Model model(0.01, 16, 1000, dataset.get_init_pose(), device, true);
    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critirias = {10,10,10};
    core::Tensor pose = dataset.get_init_pose();
    Trajectory.push_back(pose);
    t::pipelines::slam::Frame ray_cast_frame(480, 640, dataset.get_intrinsics(), device);
    t::pipelines::slam::Frame source_frame(480, 640, dataset.get_intrinsics(), device);
    t::geometry::RGBDMImage source =  dataset.get_RGBDMImage(0).To(device);
    source_frame.SetDataFromImage("color", source.color_);
    source_frame.SetDataFromImage("depth", source.depth_);
    source_frame.SetDataFromImage("mask", source.mask_);
    model.UpdateFramePose(0, dataset.get_init_pose());


    for (int i = 0; i < 10; i++){
        switch (method)
        {
        case SlamMethod::WeightMaskout:
            model.WeightMaskedIntegrate(source_frame, depthscale, depth_max);
            break;
        
        case SlamMethod::Raw:
            model.Integrate(source_frame, depthscale, depth_max);
            break;
        
        case SlamMethod::Maskout:
            model.MaskedIntegrate(source_frame, depthscale, depth_max);
            break;
        }
    }
    
    Posegraph<Open3dPosegraphBackend> graph(pose);

    Trajectory.push_back(pose);
    t::pipelines::odometry::OdometryResult result;
    core::Tensor current_T_frame_to_world;
    float weight_threshold;

    double IntergationTime = 0;
    double TrackingTime = 0;
    double RaycastTime = 0;
    double total = 0;

    for(int i = 1; i< dataset.get_size() -1 ; i++){
        
        weight_threshold = std::min(i*1.0f, 4.0f);
        t::geometry::RGBDMImage source = dataset.get_RGBDMImage(i).To(device);

        source_frame.SetDataFromImage("color", source.color_);
        source_frame.SetDataFromImage("depth", source.depth_);
        
        source_frame.SetDataFromImage("mask", source.mask_);

        auto start = std::chrono::high_resolution_clock::now();
        model.SynthesizeModelFrame(ray_cast_frame, depthscale, 0.1, depth_max, 8.0f, true, weight_threshold); 
        auto end =  std::chrono::high_resolution_clock::now();
        RaycastTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        auto image = t::geometry::Image(ray_cast_frame.GetData("color"));
        
        current_T_frame_to_world = model.GetCurrentFramePose(); 
        bool succes;
        try{ 
            auto start = std::chrono::high_resolution_clock::now();
            switch (method){
                case SlamMethod::WeightMaskout:
                    result = model.TrackMaskedFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;
                case SlamMethod::Raw:
                    result = model.TrackFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            o_method, critirias);
                    break;
                case SlamMethod::Maskout:
                    result =  model.TrackMaskedFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            o_method, critirias);
                    break;}
                auto end =  std::chrono::high_resolution_clock::now();
                TrackingTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                succes = true;
            
        }   
        catch(std::runtime_error){
            std::cout<<"tracking lost at" << dataset.get_timestamp(i) << std::endl;
            result = t::pipelines::odometry::OdometryResult();
            succes = false;
        }

        if (succes){
            model.UpdateFramePose(i, current_T_frame_to_world.Matmul(result.transformation_));
            auto start = std::chrono::high_resolution_clock::now();
            switch (method){
                case SlamMethod::WeightMaskout:
                    model.WeightMaskedIntegrate(source_frame, depthscale, depth_max, 8.0, 21, 15);
                    break;
                case SlamMethod::Raw:
                    model.Integrate(source_frame, depthscale, depth_max);
                    break;
                case SlamMethod::Maskout:
                    model.MaskedIntegrate(source_frame, depthscale, depth_max);
                    break;}
            auto end =  std::chrono::high_resolution_clock::now();
            IntergationTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        else{
            model.UpdateFramePose(i, current_T_frame_to_world);
        }

        // if (IsLargeTranslation(result.transformation_, 0.1)){
        //     std::cout<<i<<std::endl;
        // }

        pose = model.GetCurrentFramePose();
        Trajectory.push_back(pose);
    } 

    //io::WritePoseGraph("graph.json", graph.GetPoseGraph());

    auto pcd = model.ExtractPointCloud(7.0f);

    visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((pcd).ToLegacy())});
    //t::io::WritePointCloud("p.pcd", pcd);

    nlohmann::json returnJson;
    returnJson["ATE_Trans"] = dataset.ComputeATETrans(Trajectory, false);
    returnJson["ATE_Rot"] = dataset.ComputeATERot(Trajectory);
    returnJson["RPE_Trans"] = dataset.ComputeRPETrans(Trajectory, 1);
    returnJson["RPE_Rot"] = dataset.ComputeRPERot(Trajectory, 1);

    returnJson["total_time"] = (IntergationTime + TrackingTime) ;
    returnJson["avg_time"] = (IntergationTime + TrackingTime) /dataset.get_size();
    returnJson["avg_integration_time"] = IntergationTime /dataset.get_size();
    returnJson["avg_tracking_time"] = TrackingTime /dataset.get_size();
    returnJson["avg_raycast_time"] = RaycastTime/dataset.get_size();

    std::ofstream file("data/output.json");
    file << returnJson.dump(4); 
    file.close();

}

int main(){

    std::ifstream config_file("data/testConfigs/slam_config.json");
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open config.json");
    }

    nlohmann::json config;
    config_file >> config;

    core::Device device = get_device(config["device"]);
    SlamMethod method = get_slammethod(config["slam_method"]);
    SubDataset dataset_type = get_dataset(config["dataset"]);
    t::pipelines::odometry::Method odom_method = get_method(config["method"]);

    std::cout << fmt::format("On config: Device {}, Integrationmethod {}, Dataset {}",
        config["device"].get<std::string>(),  config["slam_method"].get<std::string>(),
        config["dataset"].get<std::string>())<< std::endl;

    computeMetricsSlam(dataset_type, device, method, odom_method);
}
