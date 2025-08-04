#include "CONFIG.h"
#include "TestDataset.h"
#include "pose_graph.h"
#include "TestUtil.h"

#include <nlohmann/json.hpp>

void test_slam(SubDataset data, core::Device device, SlamMethod method){

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

    for(int i = 1; i< dataset.get_size() -1 ; i++){
        
        weight_threshold = std::min(i*1.0f, 15.0f);
        t::geometry::RGBDMImage source = dataset.get_RGBDMImage(i).To(device);

        source_frame.SetDataFromImage("color", source.color_);
        source_frame.SetDataFromImage("depth", source.depth_);

        source_frame.SetDataFromImage("mask", source.mask_);
        model.SynthesizeModelFrame(ray_cast_frame, depthscale, 0.1, depth_max, 8.0f, true, weight_threshold); 

        current_T_frame_to_world = model.GetCurrentFramePose(); 
        bool succes;
        try{ 
            switch (method){
                case SlamMethod::WeightMaskout:
                    result = model.TrackMaskedFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;
                case SlamMethod::Raw:
                    result = model.TrackFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;
                case SlamMethod::Maskout:
                    result =  model.TrackMaskedFrameToModel(source_frame, ray_cast_frame, depthscale, depth_max, 0.07, 
                            t::pipelines::odometry::Method::PointToPlane, critirias);
                    break;}
                succes = true;
            
        }   
        catch(std::runtime_error){
            std::cout<<"tracking lost at" << dataset.get_timestamp(i) << std::endl;
            result = t::pipelines::odometry::OdometryResult();
            succes = false;
        }

        if (succes){
            model.UpdateFramePose(i, current_T_frame_to_world.Matmul(result.transformation_));
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
        }
        else{
            model.UpdateFramePose(i, current_T_frame_to_world);
        }

        if (IsLargeTranslation(result.transformation_, 0.1)){
            std::cout<<i<<std::endl;
        }

        //std::cout << dataset.get_timestamp(i) << std::endl;
        auto information = t::pipelines::odometry::ComputeOdometryInformationMatrix(
            source.depth_.To(device), ray_cast_frame.GetDataAsImage("depth"), 
            dataset.get_intrinsics(core::Device("CPU:0")), result.transformation_, 0.1, depthscale, depth_max);
        
        //std::cout << dataset.get_timestamp(i) << std::endl;

        graph.AddOdometryEdge(result.transformation_, information, i, i+1, false);
        graph.AddNode(pose);

        pose = model.GetCurrentFramePose();
        Trajectory.push_back(pose);

    } 

    io::WritePoseGraph("graph.json", graph.GetPoseGraph());

    auto pcd = model.ExtractPointCloud(7.0f);
    visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((pcd).ToLegacy())});
    t::io::WritePointCloud("p.pcd", pcd);

    std::cout << "realative ATE: " << dataset.ComputeATE(Trajectory, false)<< std::endl;
    std::cout << "RPE: " << dataset.ComputeRPE(Trajectory, 1)<< std::endl;
}

int main(){

    std::ifstream config_file("configs/slam_config.json");
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open config.json");
    }

    nlohmann::json config;
    config_file >> config;

    core::Device device = get_device(config["device"]);
    SlamMethod method = get_slammethod(config["slam_method"]);
    SubDataset dataset_type = get_dataset(config["dataset"]);

    std::cout << fmt::format("On config: Device {}, Integrationmethod {}, Dataset {}",
        config["device"].get<std::string>(),  config["slam_method"].get<std::string>(),
        config["dataset"].get<std::string>())<< std::endl;

    test_slam(dataset_type, device, method);
}
