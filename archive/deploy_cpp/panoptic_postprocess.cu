#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

/*
 * CUDA 加速的 Panoptic-DeepLab 后处理核心：
 * 1) find_instance_center_cuda: 阈值 + maxpool NMS + topk/nonzero
 * 2) group_pixels_cuda: 每像素线程遍历 K 个中心找最近中心
 * 3) merge_semantic_and_instance_cuda: 多数类投票 + 实例编号 + 写 panoptic
 * 4) get_panoptic_segmentation_cuda: 组合上述步骤
 *
 * 假设输入形状：
 *   sem_seg:        [1, H, W] int64
 *   center_heatmap: [1, H, W] float
 *   offsets:        [2, H, W] float
 */

// ---------------------- group_pixels ----------------------
template <typename scalar_t>
__global__ void group_pixels_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> centers,   // [K,2]
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> offsets,   // [2,H,W]
    torch::PackedTensorAccessor64<int64_t, 3, torch::RestrictPtrTraits> out               // [1,H,W]
) {
    const int W = offsets.size(2);
    const int H = offsets.size(1);
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int K = centers.size(0);
    const scalar_t cy = static_cast<scalar_t>(y) + offsets[0][y][x];
    const scalar_t cx = static_cast<scalar_t>(x) + offsets[1][y][x];

    scalar_t best = static_cast<scalar_t>(1e30);
    int64_t best_k = 0;
    for (int k = 0; k < K; ++k) {
        const scalar_t dy = centers[k][0] - cy;
        const scalar_t dx = centers[k][1] - cx;
        const scalar_t d2 = dy * dy + dx * dx;
        if (d2 < best) {
            best = d2;
            best_k = k;
        }
    }
    out[0][y][x] = best_k + 1; // id=0 给 stuff
}

torch::Tensor group_pixels_cuda(torch::Tensor center_points, torch::Tensor offsets) {
    c10::cuda::CUDAGuard device_guard(offsets.device());
    const auto H = offsets.size(1);
    const auto W = offsets.size(2);
    auto out = torch::zeros({1, H, W}, offsets.options().dtype(torch::kInt64));

    const dim3 threads(16, 16);
    const dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(offsets.scalar_type(), "group_pixels_cuda", [&] {
        group_pixels_kernel<scalar_t><<<blocks, threads>>>(
            center_points.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            offsets.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            out.packed_accessor64<int64_t, 3, torch::RestrictPtrTraits>());
    });
    return out;
}

// ---------------------- merge_semantic_and_instance ----------------------
__global__ void hist_kernel(
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> sem_seg,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> ins_seg,
    const torch::PackedTensorAccessor64<uint8_t, 2, torch::RestrictPtrTraits> semantic_thing_seg,
    const torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> thing_lut,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> hist // [max_ins+1, num_classes]
) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int H = sem_seg.size(0);
    const int W = sem_seg.size(1);
    if (y >= H || x >= W) return;

    const auto ins_id = ins_seg[y][x];
    if (ins_id <= 0) return;
    if (!semantic_thing_seg[y][x]) return;

    const auto cls = sem_seg[y][x];
    if (cls < 0 || cls >= thing_lut.size(0)) return;
    if (!thing_lut[cls]) return;
    atomicAdd(&hist[ins_id][cls], 1);
}

__global__ void stuff_area_kernel(
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> sem_seg,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> ins_seg,
    const torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> thing_lut,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> area // [num_classes]
) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int H = sem_seg.size(0);
    const int W = sem_seg.size(1);
    if (y >= H || x >= W) return;
    if (ins_seg[y][x] != 0) return;

    const auto cls = sem_seg[y][x];
    if (cls < 0 || cls >= thing_lut.size(0)) return;
    if (thing_lut[cls]) return;
    atomicAdd(&area[cls], 1);
}

__global__ void majority_newid_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> hist, // [max_ins+1, C]
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> thing_counter, // [C]
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> inst_major, // [max_ins+1]
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> inst_newid   // [max_ins+1]
) {
    const int ins_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ins_id <= 0 || ins_id >= hist.size(0)) return;
    const int C = hist.size(1);

    int best_c = 0;
    int best_cnt = -1;
    for (int c = 0; c < C; ++c) {
        const int cnt = hist[ins_id][c];
        if (cnt > best_cnt) {
            best_cnt = cnt;
            best_c = c;
        }
    }
    inst_major[ins_id] = best_c;
    const int new_id = atomicAdd((int*)&thing_counter[best_c], 1) + 1;
    inst_newid[ins_id] = new_id;
}

__global__ void write_panoptic_kernel(
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> sem_seg,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> ins_seg,
    const torch::PackedTensorAccessor64<uint8_t, 2, torch::RestrictPtrTraits> semantic_thing_seg,
    const torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> thing_lut,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> inst_major,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> inst_newid,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> stuff_area_cnt,
    const int64_t label_divisor,
    const int64_t stuff_area_thr,
    const int64_t void_label,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> pan_seg // [H,W]
) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int H = sem_seg.size(0);
    const int W = sem_seg.size(1);
    if (y >= H || x >= W) return;

    const auto ins_id = ins_seg[y][x];
    const auto cls = sem_seg[y][x];

    if (ins_id > 0 && semantic_thing_seg[y][x] && cls >= 0 && cls < thing_lut.size(0) && thing_lut[cls]) {
        const int c = inst_major[ins_id];
        const int n = inst_newid[ins_id];
        pan_seg[y][x] = static_cast<int64_t>(c) * label_divisor + static_cast<int64_t>(n);
    } else {
        if (cls >= 0 && cls < thing_lut.size(0) && !thing_lut[cls] && stuff_area_cnt[cls] >= stuff_area_thr) {
            pan_seg[y][x] = static_cast<int64_t>(cls) * label_divisor;
        } else {
            pan_seg[y][x] = void_label;
        }
    }
}

torch::Tensor merge_semantic_and_instance_cuda(
    torch::Tensor sem_seg,              // int64 [1,H,W] or [H,W]
    torch::Tensor ins_seg,              // int64 [1,H,W] or [H,W]
    torch::Tensor semantic_thing_seg,   // bool/int [1,H,W] or [H,W]
    torch::Tensor thing_ids,            // int64 list of thing class ids
    const int64_t label_divisor,
    const int64_t stuff_area,
    const int64_t void_label
) {
    c10::cuda::CUDAGuard device_guard(sem_seg.device());
    if (sem_seg.dim() == 3) sem_seg = sem_seg.squeeze(0);
    if (ins_seg.dim() == 3) ins_seg = ins_seg.squeeze(0);
    if (semantic_thing_seg.dim() == 3) semantic_thing_seg = semantic_thing_seg.squeeze(0);

    const auto H = sem_seg.size(0);
    const auto W = sem_seg.size(1);

    const auto max_cls = sem_seg.max().item<int64_t>() + 1;
    auto thing_lut = torch::zeros({max_cls}, torch::dtype(torch::kUInt8).device(sem_seg.device()));
    if (thing_ids.numel() > 0) {
        thing_lut.index_put_({thing_ids}, 1);
    }

    const int64_t max_ins = ins_seg.max().item<int64_t>();
    auto hist = torch::zeros({max_ins + 1, max_cls}, torch::dtype(torch::kInt32).device(sem_seg.device()));
    auto inst_major = torch::zeros({max_ins + 1}, torch::dtype(torch::kInt32).device(sem_seg.device()));
    auto inst_newid = torch::zeros({max_ins + 1}, torch::dtype(torch::kInt32).device(sem_seg.device()));
    auto thing_counter = torch::zeros({max_cls}, torch::dtype(torch::kInt32).device(sem_seg.device()));
    auto stuff_area_cnt = torch::zeros({max_cls}, torch::dtype(torch::kInt32).device(sem_seg.device()));
    auto semantic_thing_seg_u8 = semantic_thing_seg.to(torch::kUInt8);

    const dim3 threads(16, 16);
    const dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    hist_kernel<<<blocks, threads>>>(
        sem_seg.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        ins_seg.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        semantic_thing_seg_u8.packed_accessor64<uint8_t, 2, torch::RestrictPtrTraits>(),
        thing_lut.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
        hist.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>());

    stuff_area_kernel<<<blocks, threads>>>(
        sem_seg.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        ins_seg.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        thing_lut.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
        stuff_area_cnt.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

    const dim3 threads_ins(256);
    const dim3 blocks_ins((max_ins + threads_ins.x) / threads_ins.x + 1);
    majority_newid_kernel<<<blocks_ins, threads_ins>>>(
        hist.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
        thing_counter.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        inst_major.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        inst_newid.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

    auto pan_seg = torch::zeros({H, W}, torch::dtype(torch::kInt64).device(sem_seg.device()));
    write_panoptic_kernel<<<blocks, threads>>>(
        sem_seg.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        ins_seg.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        semantic_thing_seg_u8.packed_accessor64<uint8_t, 2, torch::RestrictPtrTraits>(),
        thing_lut.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
        inst_major.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        inst_newid.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        stuff_area_cnt.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        label_divisor,
        stuff_area,
        void_label,
        pan_seg.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>());

    return pan_seg.unsqueeze(0); // [1,H,W]
}

// ---------------------- find_instance_center ----------------------
std::tuple<torch::Tensor, torch::Tensor> find_instance_center_cuda(
    torch::Tensor center_heatmap,  // [1,H,W] float
    double threshold,
    int64_t nms_kernel,
    c10::optional<int64_t> top_k_opt
) {
    c10::cuda::CUDAGuard device_guard(center_heatmap.device());
    auto heat = at::threshold(center_heatmap, threshold, -1);
    const int64_t pad = (nms_kernel - 1) / 2;
    auto pooled = at::max_pool2d(heat, {nms_kernel, nms_kernel}, {1, 1}, {pad, pad});
    heat = at::where(heat == pooled, heat, at::full_like(heat, -1));
    heat = heat.squeeze(0); // [H,W]

    torch::Tensor centers;
    if (top_k_opt.has_value()) {
        const int64_t top_k = top_k_opt.value();
        auto flat = heat.flatten();
        auto topk = std::get<0>(at::topk(flat, top_k));
        auto thr = topk[-1].clamp_min(0);
        centers = at::nonzero(heat > thr); // [K,2] (y,x)
    } else {
        centers = at::nonzero(heat > 0);
    }
    return {centers, heat};
}

// ---------------------- get_panoptic_segmentation ----------------------
std::tuple<torch::Tensor, torch::Tensor> get_panoptic_segmentation_cuda(
    torch::Tensor sem_seg,           // int64 [1,H,W]
    torch::Tensor center_heatmap,    // float [1,H,W]
    torch::Tensor offsets,           // float [2,H,W]
    torch::Tensor thing_ids,         // int64 list
    int64_t label_divisor,
    int64_t stuff_area,
    int64_t void_label,
    double threshold,
    int64_t nms_kernel,
    int64_t top_k
) {
    c10::cuda::CUDAGuard device_guard(sem_seg.device());
    auto center_res = find_instance_center_cuda(center_heatmap, threshold, nms_kernel, top_k);
    auto centers = std::get<0>(center_res); // [K,2]

    torch::Tensor instance = torch::zeros_like(sem_seg, torch::kInt64);
    if (centers.size(0) > 0) {
        auto ins_id = group_pixels_cuda(centers.to(offsets.device()).to(offsets.scalar_type()), offsets);
        instance = ins_id;
    }
    // foreground mask: thing 类别
    auto thing_mask = torch::zeros_like(sem_seg, torch::kUInt8);
    if (thing_ids.numel() > 0) {
        for (int64_t i = 0; i < thing_ids.size(0); ++i) {
            const auto id = thing_ids[i].item<int64_t>();
            thing_mask |= (sem_seg == id);
        }
    }
    auto panoptic = merge_semantic_and_instance_cuda(
        sem_seg.to(torch::kInt64),
        instance.to(torch::kInt64),
        thing_mask.to(torch::kUInt8),
        thing_ids,
        label_divisor,
        stuff_area,
        void_label);
    return {panoptic, centers.unsqueeze(0)}; // panoptic [1,H,W], centers [1,K,2]
}

// 声明给 pybind 使用
std::tuple<torch::Tensor, torch::Tensor> find_instance_center_cuda_bind(
    torch::Tensor center_heatmap,
    double threshold,
    int64_t nms_kernel,
    int64_t top_k) {
    c10::optional<int64_t> topk_opt = top_k > 0 ? c10::optional<int64_t>(top_k) : c10::nullopt;
    auto out = find_instance_center_cuda(center_heatmap, threshold, nms_kernel, topk_opt);
    return out;
}
