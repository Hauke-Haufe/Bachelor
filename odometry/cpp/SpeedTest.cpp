#include "CONFIG.h"
#include "TestDataset.h"

#include<chrono>
#include <nlohmann/json.hpp>

double DOdometry(Tum_dataset& dataset, 
                int i, int j, t::pipelines::odometry::Method method, 
                core::Device device,
                std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
                bool& sucess,
                core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))) ){

    auto source = dataset.get_RGBDMImage(i);
    auto target = dataset.get_RGBDMImage(j);

    try{
        auto start = std::chrono::high_resolution_clock::now();
        auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(),
                init, 
                5000.0F,
                7.0F,
                critiria,
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        sucess = true;
        auto end =  std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count();
    }
     catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
        return 0.0;
    }
    
}

double Odometry(Tum_dataset& dataset, 
        int i, int j, t::pipelines::odometry::Method method, 
        core::Device device, 
        std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critiria, 
        bool& sucess,
        core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")))){

    auto source = dataset.get_RGBDImage(i);
    auto target = dataset.get_RGBDImage(j);

    //auto Pcd = t::geometry::PointCloud::CreateFromDepthImage(source.depth_, dataset.get_intrinsics(device), 
    //    core::Tensor::Eye(4, core::Float32, ((open3d::core::Device)("CPU:0"))), 1000.0f, 10.0f);
    //visualization::DrawGeometries({std::make_shared<geometry::PointCloud>((Pcd).ToLegacy())});
    try{
        auto start = std::chrono::high_resolution_clock::now();
        auto result = t::pipelines::odometry::RGBDOdometryMultiScale(source.To(device), 
                target.To(device),
                dataset.get_intrinsics(),
                init, 
                5000.0F,
                7.0F,
                critiria,
                method,
                open3d::t::pipelines::odometry::OdometryLossParams());

        sucess = true;
        auto end =  std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count();
    }
    catch(std::runtime_error){
        std::cout<<"tracking lost at" <<dataset.get_timestamp(i)<<std::endl;
        sucess = false;
        return 0.0;
    }
}

void testSpeedOdom(SubDataset data,t::pipelines::odometry::Method method, MaskMethod m_mehtod, core::Device  device){
    auto dataset = Tum_dataset(data);
    auto dtype = core::Dtype::Float32;

    std::vector<t::pipelines::odometry::OdometryConvergenceCriteria> critirias = {6,3,1};

    bool success;
    double time = 0;
    int success_count = 0;
    for(int i = 0; i< dataset.get_size() -1 ; i++){
        double result;
        switch (m_mehtod){
        
            case MaskMethod::CompleteMask:
                result = DOdometry(dataset, i, i+1,  method, device, critirias, success);
                break;

            case MaskMethod::NoMask:
                result = Odometry(dataset, i, i+1, method, device, critirias, success);
                break;
        }

        if (success){
            time += result;
            success_count += 1;
        }
        
    } 
    
    nlohmann::json returnJson;
    returnJson["total_time"] = time ;
    returnJson["avg_time"] = time/success_count;

    std::ofstream file("data/output.json");
    file << returnJson.dump(4); 
    file.close();
}

void runSpeedTestOdometry(){
    std::ifstream config_file("configs/odometry_config.json");
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open config.json");
    }

    nlohmann::json config;
    config_file >> config;

    core::Device device = get_device(config["device"]);
    t::pipelines::odometry::Method method = get_method(config["method"]);
    MaskMethod mask_mehtod = get_maskmethod(config["maskingmethod"]);
    SubDataset dataset_type = get_dataset(config["dataset"]);

    std::cout << fmt::format("On config: Device {}, Method {}, Maskoutmethod {}, Dataset {}",
        config["device"].get<std::string>(),  config["method"].get<std::string>(), 
        config["maskingmethod"].get<std::string>(),config["dataset"].get<std::string>())<< std::endl;
    
    testSpeedOdom(dataset_type, method, mask_mehtod, device);
}

void testSpeedSlam(SubDataset data, core::Device device, SlamMethod method, t::pipelines::odometry::Method o_method){

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

    t::pipelines::odometry::OdometryResult result;
    core::Tensor current_T_frame_to_world;
    float weight_threshold;

    double avgIntergationTime = 0;
    double avgTrackingTime = 0;
    double avgTime = 0;
    double total = 0;

    int count = 0;
    for(int i = 1; i< dataset.get_size() -1 ; i++){
        
        weight_threshold = std::min(i*1.0f, 15.0f);
        t::geometry::RGBDMImage source = dataset.get_RGBDMImage(i).To(device);

        source_frame.SetDataFromImage("color", source.color_);
        source_frame.SetDataFromImage("depth", source.depth_);
        
        source_frame.SetDataFromImage("mask", source.mask_);
        model.SynthesizeModelFrame(ray_cast_frame, depthscale, 0.1, depth_max, 8.0f, true, weight_threshold); 

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
                avgTrackingTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                succes = true;
            
        }   
        catch(std::runtime_error){
            std::cout<<"tracking lost at" << dataset.get_timestamp(i) << std::endl;
            result = t::pipelines::odometry::OdometryResult();
            succes = false;
        }

        if (succes){
            count += 1;

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
            avgTrackingTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        else{
            model.UpdateFramePose(i, current_T_frame_to_world);
        }
    } 
    
    nlohmann::json returnJson;
    returnJson["total_time"] = (avgIntergationTime + avgTrackingTime) ;
    returnJson["avg_time"] = (avgIntergationTime + avgTrackingTime) *(dataset.get_size()/count);
    returnJson["avg_integration_time"] = avgIntergationTime /count;
    returnJson["avg_tracking_time"] = avgTrackingTime /count;

    std::ofstream file("data/output.json");
    file << returnJson.dump(4); 
    file.close();
}

void runSpeedTestSlam(){
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
    
    testSpeedSlam(dataset_type, device, method, odom_method);   
}

int main(int argc, char* argv[]){

    if (argc > 1) {
        std::string param = argv[1]; // first argument after program name

        if(param == "odometry"){
            runSpeedTestOdometry();
        }
        else if(param == "slam"){
            runSpeedTestSlam();
        }
        else{
            throw std::runtime_error("invalid command line Argument");
        }

    } else {
        throw std::runtime_error("invalid command line Argument");
    }


    return 0;
}