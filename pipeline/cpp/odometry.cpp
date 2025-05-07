#include <open3d/Open3D.h>
#include<iostream>
using namespace open3d;

std::string Path = "/home/nb-messen-07/Desktop/SpatialMapping/data/images";

template <typename Derived>
core::Tensor EigentoTensorF64i(Eigen::MatrixBase<Derived> matrix){

    constexpr int rows = Derived::RowsAtCompileTime;
    constexpr int  cols = Derived::ColsAtCompileTime;
    core::SizeVector size({rows, cols});

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_row = matrix;

    std::cout<<mat_row<<std::endl;
    std::cin;

    return core::Tensor(
        static_cast<double*>(mat_row.data()), 
        core::Dtype::Float64, size
    );
    
}

template <typename Derived>
core::Tensor EigentoTensorF64(Eigen::MatrixBase<Derived>& matrix){
    
    constexpr int rows = Derived::RowsAtCompileTime;
    constexpr int  cols = Derived::ColsAtCompileTime;
    core::SizeVector size({rows, cols});

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_row = matrix;
    core::Tensor tensor(size, core::Dtype::Float64);

    for (int i = 0; i< rows; ++i){
        for (int j = 0 ; j< cols; ++j){
            tensor[i][j] = mat_row(i,j);
        }
    }
    return tensor;

    
}



void visualizeImaget(t::geometry::RGBDImage image){

    auto legacy_image = image.ToLegacy();
    auto image_ptr = std::make_shared<geometry::Image>(legacy_image.depth_);

    std::vector<std::shared_ptr<const geometry::Geometry>> images;
    images.push_back(image_ptr);

    visualization::DrawGeometries(images);
}

t::pipelines::odometry::OdometryResult Odometry(
    t::geometry::RGBDImage& source,
    t::geometry::RGBDImage& target,    
    core::Tensor& intrinsics
){
    
        //hier kann man noch convergence Critiria angeben und co    
        
        //visualizeImaget(source);
        //visualizeImaget(target);

        //std::cout << intrinsics.ToString();
        //std::cin;

        //utility::VerbosityContextManager contex(utility::VerbosityLevel::Debug);
        //contex.Enter();
        core::Device device("CUDA:0");


        return t::pipelines::odometry::RGBDOdometryMultiScale(
            source.To(device),
            target.To(device),
            intrinsics
        );
};

pipelines::registration::PoseGraph Multiway_registration(
    std::vector<std::shared_ptr<t::geometry::RGBDImage>> Images, 
    core::Tensor &instrinsics
){
    
    auto Posegraph = pipelines::registration::PoseGraph();
    auto trans_odometry = Eigen::Matrix4d::Identity();
    
    pipelines::registration::PoseGraphNode Node(trans_odometry);
    
    Posegraph.nodes_.emplace_back(Node);
    
    bool uncertain;

    for(int source_id= 0; source_id < Images.size(); source_id++){

        for (int target_id = source_id +1; target_id < source_id+7; target_id ++){
            

            if(target_id == source_id +1){
                bool uncertain = false;
            }
            else{
                bool uncertain = true;
            }

            auto result = Odometry(*(Images[source_id]), *(Images[target_id]), instrinsics);
            


            //assert(result.transformation_.GetShape() == std::vector<int64_t>({4,4}));
            //assert(result.transformation_.GetDtype() == core::Float64);

            double* dataptr = result.transformation_.GetDataPtr<double>();
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Transform(dataptr, 4,4);
            
            // schreib die Funktionen auf Tensor Images um pain
            auto information = Eigen::Matrix6d::Identity();

            auto Edge = pipelines::registration::PoseGraphEdge(
                source_id,
                target_id,
                Transform, 
                information, 
                uncertain 
            );

            Posegraph.nodes_.emplace_back(Transform);
            Posegraph.edges_.emplace_back(Edge);  
            }

        }

    return Posegraph;    
}
    

pipelines::registration::PoseGraph optimize_posegraph(
    pipelines::registration::PoseGraph& Posegraph){
    
    double max_correspondence_distance = 0.01;
    double preference_loop_closure = 0.2;

    auto method = pipelines::registration::GlobalOptimizationLevenbergMarquardt();
    auto criteria = pipelines::registration::GlobalOptimizationConvergenceCriteria();

    auto option = pipelines::registration::GlobalOptimizationOption(
        max_correspondence_distance,
        0.25,
        preference_loop_closure,
        0
    );

    pipelines::registration::GlobalOptimization(Posegraph, method, criteria, option);

    return Posegraph;
};

std::unique_ptr<t::geometry::VoxelBlockGrid> create_vgb(){

    std::vector<std::string> atr_names = {"tsdf", "weight", "color"};
    std::vector<core::Dtype> atr_types = {core::Dtype::Float32,core::Dtype::Float32, core::Dtype::Float32 };
    std::vector<core::SizeVector> channels = {{1}, {1}, {3}};
    core::Device device("CUDA:0");

    return std::make_unique<t::geometry::VoxelBlockGrid>(
        atr_names, 
        atr_types,
        channels,
        0.005,
        16,
        10000,
        device
    );

}

std::unique_ptr<t::geometry::VoxelBlockGrid> integrate(
    pipelines::registration::PoseGraph& Posegraph,
    core::Tensor& instrinsics,
    std::string path){
    
    std::unique_ptr<t::geometry::VoxelBlockGrid> vgb;
    vgb = create_vgb();

    t::geometry::Image color_image;
    t::geometry::Image depth_image;
    
    for(int i = 0; Posegraph.nodes_.size(); i++){

        std::stringstream color_path_ss;
        std::stringstream depth_path_ss;
        color_path_ss << path << "/color/image" << i << ".png";
        depth_path_ss << path << "/depth/image" << i << ".png";

        std::string color_path = color_path_ss.str();
        std::string depth_path = depth_path_ss.str();

        t::io::ReadImage(color_path, color_image);
        t::io::ReadImage(depth_path, depth_image);
        

        auto pose = Posegraph.nodes_[i].pose_;
        Eigen::Matrix4d inverse = pose.inverse();
        Eigen::Matrix4d inverse_transpose = inverse.transpose();
        core::SizeVector size(4,4);

        core::Tensor inverse_t(static_cast<void*>(inverse_transpose.data()), core::Dtype::Float64, size);

        auto frustum_block_coords = vgb->GetUniqueBlockCoordinates(
            depth_image,
            instrinsics,
            inverse_t
        );

        vgb->Integrate(
            frustum_block_coords,
            depth_image,
            color_image,
            instrinsics,
            inverse_t
        );

    }

    return vgb;
}