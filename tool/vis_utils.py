import numpy as np
import torch
import open3d as o3d
from IPython.display import clear_output
from tool.visualization_helpers import get_new_pallete
import Scene_rep
from IPython.display import clear_output
import plotly.graph_objects as go
class Vis_color:
    def __init__(self, use_vis):
        self.use_vis = use_vis
        if not self.use_vis:
            return None
        self.vis_image = o3d.visualization.Visualizer()
        self.vis_image.create_window(window_name="input color image", width=320, height=240, left=1750)

        self.pallete = get_new_pallete(100)  # list(3 * cls_num)

    # @param color_image: Tensor(H, W, 3), dtype=float32
    def update(self, color_image):
        color_img_nd = (color_image.cpu().numpy() * 255).astype(np.uint8)

        if not self.use_vis:
            return
        geometry_image = o3d.geometry.Image(color_img_nd)
        self.vis_image.add_geometry(geometry_image)
        self.vis_image.poll_events()
        self.vis_image.update_renderer()
        geometry_image.clear()


class Vis_pointcloud:
    def __init__(self, use_vis, args, device="cuda:0"):
        self.use_vis = use_vis
        self.args = args
        self.device = device
        if not self.use_vis:
            return None

        self.add_geo_flag = False
        self.pcd = o3d.geometry.PointCloud()
        self.scene_points = None
        self.scene_points_color = None


    def get_text_embeddings(self):
        text_embed_path = self.args.vocab_feature_file
        text_embeddings = torch.load(text_embed_path)
        return text_embeddings

    def get_color_pallete(self, label_num, brighten=True):
        pallete = get_new_pallete(label_num)  # list(3 * label_num)
        pallete = np.array(pallete).reshape((-1, 3))  # RGB value of each label, [0~255], ndarray(label_num, 3), dtype=int64

        if brighten:
            pallete_float = pallete.astype("float32") / 255.
            pallete_float = np.power(pallete_float, 1 / 2.2)
            pallete = (pallete_float * 255).astype("int64")
        return pallete

    def set_uniform_color(self, scene_colors, rgb=[156, 156, 156]):
        rgb = np.array(rgb)
        scene_colors[:] = rgb
        return scene_colors
    
    def vis_one_object(self,point_ids, scene_points):
        points = scene_points[point_ids]
        color = (torch.rand(3) * 0.7 + 0.3) * 255
        colors = torch.tile(color, (points.shape[0], 1))
        pts_mean = torch.mean(points, dim=0)
        return point_ids, points, colors, color, pts_mean
    
    def adjust_colors_to_pastel(self, rgb_array, factor=0.9):
        pastel_rgb_array = rgb_array * factor + (1 - factor)
        return pastel_rgb_array
    
    def process_mask_boundary_pts(self, pred_inst_masks, return_list=False):
        if isinstance(pred_inst_masks, list):
            pred_inst_masks = torch.stack(pred_inst_masks, dim=0)

        # Step 1: sort each predicted instance by mask size, with ascending order
        pred_inst_size = torch.sum(pred_inst_masks, dim=-1)
        pred_inst_idx_asc = torch.argsort(pred_inst_size)
        pred_inst_masks = pred_inst_masks[pred_inst_idx_asc]  # predicted instances' masks (sort by mask size, ascending order), Tensor(pred_inst_num, n), dtype=bool

        # Step 2: for each boundary points, it will be only assigned to 1 pred instance (with minimal mask size)
        pred_inst_masks_new = self.keep_min_rows(pred_inst_masks)
        if return_list:
            pred_inst_masks_new = [inst_mask for inst_mask in pred_inst_masks]

        return pred_inst_masks_new

    def keep_min_rows(self,mask_tensor):
        cumsum = mask_tensor.cumsum(dim=0)  # cumulative sum for each col, Tensor(m, n), dtype=int
        first_true_mask = (cumsum == 1)

        mask_tensor_new = mask_tensor & first_true_mask
        return mask_tensor_new
    
    # @brief: update members "self.scene_points" and "self.scene_points_color" to latest reconstruction and segmentation results;
    # @param points: ndarray(scene_pts_num, 3);
    # @param instance_list: list of Instance obj;
    # @param instance_pts_mask_list: list of Tensor(scene_pts_num, ), dtype=bool;
    def show_current_seg_pc(self, scene_points, instance_list, instance_pts_mask_list, inst_rgb_list):
        
        if isinstance(scene_points, np.ndarray):
            scene_points = torch.from_numpy(scene_points).to(self.device)

        num_instances = len(instance_list)

        pred_inst_masks = self.process_mask_boundary_pts(instance_pts_mask_list)
        inst_text = [instance.get_instance_text for instance in instance_list]

        # Step 4: paint each instance's corresponding scene points
        scene_colors = torch.zeros_like(scene_points)
        scene_colors = torch.pow(scene_colors, 1 / 2.2)
        scene_colors = scene_colors * 255
        instance_colors = 200. * torch.ones_like(scene_colors)  # set background to gray

        centers=[]
        for idx in range(num_instances):
            instance_mask = pred_inst_masks[idx].cpu().numpy()
            corr_point_ids = np.where(instance_mask)[0]
            point_ids, points, colors, label_color, center = self.vis_one_object(corr_point_ids, scene_points)

            if inst_rgb_list is None:
                instance_colors[point_ids] = label_color.to(self.device)
            else:
                instance_colors[point_ids] = inst_rgb_list[idx].to(self.device)
            
            if center is None:
            # fallback: mean of points if center missing
                if isinstance(points, torch.Tensor):
                    c = points.mean(dim=0).cpu().numpy()
                else:
                    c = np.mean(points, axis=0)
            else:
                if hasattr(center, "cpu"):
                    c = center.cpu().numpy()
                else:
                    c = np.asarray(center)
            centers.append(c.tolist())
            
        instance_colors = instance_colors.cpu().numpy() / 255.
        instance_colors = self.adjust_colors_to_pastel(instance_colors)  # adjust color brightness
        self.scene_points_color = instance_colors

        self.scene_points = scene_points.cpu().numpy()
        self._centers = np.array(centers) if len(centers) > 0 else None
        self._inst_texts = inst_text
        
    def update(self):
        if not self.use_vis:
            return

        if self.scene_points is None or self.scene_points_color is None:
            return

        # build Open3D pointcloud

        self.pcd.points = o3d.utility.Vector3dVector(self.scene_points)
        # assume scene_points_color in 0..1 range already (as in your code)
        self.pcd.colors = o3d.utility.Vector3dVector(self.scene_points_color)

        # try to generate a Plotly figure via draw_plotly
        fig = None
        try:
            # draw_plotly often returns a plotly.graph_objects.Figure
            fig = o3d.visualization.draw_plotly([self.pcd], mesh_show_wireframe=False, width=1600, height=900)
        except Exception:
            # some environments may raise or not return a Figure; ignore and fall back
            fig = None

        # Prepare text labels if available
        centers = getattr(self, "_centers", None)   # expected shape (M,3) or None
        texts = getattr(self, "_inst_texts", None)  # list of strings or None

        # If draw_plotly returned a figure object, add text trace to it
        if fig is not None and hasattr(fig, "add_trace"):
            if centers is not None and len(centers) > 0:
                centers = np.asarray(centers)
                labels = texts if texts is not None else [f"inst_{i}" for i in range(len(centers))]
                text_trace = go.Scatter3d(
                    x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                    mode="text",
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=12, color="black"),
                    hoverinfo="none"
                )
                fig.add_trace(text_trace)

            # clear previous notebook output and show updated figure
            clear_output(wait=True)
            fig.show()
            return

        # Fallback: draw_plotly didn't return a Figure — create plotly Figure directly
        # Convert colors to "rgb(r,g,b)" strings for Plotly
        pts = self.scene_points
        cols = (np.clip(self.scene_points_color, 0.0, 1.0) * 255).astype(int)
        color_strings = ["rgb(%d,%d,%d)" % tuple(c) for c in cols]

        trace_pts = go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color=color_strings),
            hoverinfo="none"
        )

        data = [trace_pts]
        if centers is not None and len(centers) > 0:
            centers = np.asarray(centers)
            labels = texts if texts is not None else [f"inst_{i}" for i in range(len(centers))]
            trace_text = go.Scatter3d(
                x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                mode="text",
                text=labels,
                textposition="top center",
                textfont=dict(size=12, color="black"),
                hoverinfo="none"
            )
            data.append(trace_text)

        fig2 = go.Figure(data=data)
        fig2.update_layout(scene=dict(aspectmode="auto"), width=1600, height=900)

        clear_output(wait=True)
        fig2.show()


























        # if not self.add_geo_flag:
        #     self.vis.add_geometry(self.pcd)
        #     self.add_geo_flag = True
        # else:
        #     self.vis.update_geometry(self.pcd)

        # self.vis.poll_events()
        # self.vis.update_renderer()
