#include <torch/extension.h>

// 声明 .cu 中的实现
torch::Tensor group_pixels_cuda(torch::Tensor center_points, torch::Tensor offsets);
torch::Tensor merge_semantic_and_instance_cuda(
    torch::Tensor sem_seg,
    torch::Tensor ins_seg,
    torch::Tensor semantic_thing_seg,
    torch::Tensor thing_ids,
    const int64_t label_divisor,
    const int64_t stuff_area,
    const int64_t void_label);
std::tuple<torch::Tensor, torch::Tensor> find_instance_center_cuda_bind(
    torch::Tensor center_heatmap,
    double threshold,
    int64_t nms_kernel,
    int64_t top_k);
std::tuple<torch::Tensor, torch::Tensor> get_panoptic_segmentation_cuda(
    torch::Tensor sem_seg,
    torch::Tensor center_heatmap,
    torch::Tensor offsets,
    torch::Tensor thing_ids,
    int64_t label_divisor,
    int64_t stuff_area,
    int64_t void_label,
    double threshold,
    int64_t nms_kernel,
    int64_t top_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_pixels", &group_pixels_cuda, "group_pixels (CUDA)");
    m.def("merge_semantic_and_instance", &merge_semantic_and_instance_cuda, "merge_semantic_and_instance (CUDA)");
    m.def("find_instance_center", &find_instance_center_cuda_bind, "find_instance_center (CUDA)");
    m.def("get_panoptic_segmentation", &get_panoptic_segmentation_cuda, "get_panoptic_segmentation (CUDA)");
}
