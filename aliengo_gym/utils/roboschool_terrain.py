# flat_terrain_only.py

import numpy as np
from isaacgym import terrain_utils


class Terrain:
    def __init__(
        self,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=0.0,
        terrain_length=10.0,
        terrain_width=5.0,
        mesh_type="trimesh",
        slope_treshold=1.5,
    ):
        self.type = mesh_type
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.border_size = border_size
        self.cfg = self
        self.terrain_length = terrain_length
        self.terrain_width = terrain_width
        self.slope_treshold = slope_treshold
        self.wall_height = 1.0
        self.wall_thickness = 0.5

        if self.type == "none":
            return

        self.length_per_env_pixels = int(self.terrain_length / self.horizontal_scale)
        self.width_per_env_pixels = int(self.terrain_width / self.horizontal_scale)
        self.border = int(self.border_size / self.horizontal_scale)
        self.wall_height_px = int(self.wall_height / self.vertical_scale)
        self.wall_thickness_px = max(1, int(self.wall_thickness / self.horizontal_scale))

        self.tot_rows = self.width_per_env_pixels + 2 * self.border
        self.tot_cols = self.length_per_env_pixels + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        terrain = terrain_utils.SubTerrain(
            "flat",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.vertical_scale,
            horizontal_scale=self.horizontal_scale,
        )
        terrain.height_field_raw[:, :] = 0

        start_x = self.border
        end_x = self.border + self.width_per_env_pixels
        start_y = self.border
        end_y = self.border + self.length_per_env_pixels

        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # walls around the terrain
        self.height_field_raw[start_x:start_x + self.wall_thickness_px, start_y:end_y] = self.wall_height_px
        self.height_field_raw[end_x - self.wall_thickness_px:end_x, start_y:end_y] = self.wall_height_px
        self.height_field_raw[start_x:end_x, start_y:start_y + self.wall_thickness_px] = self.wall_height_px
        self.height_field_raw[start_x:end_x, end_y - self.wall_thickness_px:end_y] = self.wall_height_px

        # internal wall
        inner_wall_length = 2.0
        inner_wall_thickness = 0.5

        inner_wall_length_px = max(1, int(inner_wall_length / self.horizontal_scale))
        inner_wall_thickness_px = max(1, int(inner_wall_thickness / self.horizontal_scale))

        # place it approximately in the center of the flat area
        center_x = (start_x + end_x) // 2 + 5
        center_y = (start_y + end_y) // 2 - 15

        # vertical wall (long in y direction, thick in x direction)
        wall_x0 = center_x - inner_wall_thickness_px // 2
        wall_x1 = wall_x0 + inner_wall_thickness_px

        wall_y0 = center_y - inner_wall_length_px // 2
        wall_y1 = wall_y0 + inner_wall_length_px

        # keep it inside the usable area and away from the border walls
        margin = self.wall_thickness_px + 1
        wall_x0 = max(start_x + margin, wall_x0)
        wall_x1 = min(end_x - margin, wall_x1)
        wall_y0 = max(start_y + margin, wall_y0)
        wall_y1 = min(end_y - margin, wall_y1)

        self.height_field_raw[wall_x0:wall_x1, wall_y0:wall_y1] = self.wall_height_px

        # =========================
        # fixed internal obstacles
        # =========================

        obstacle_height_px = self.wall_height_px

        def clamp_to_inner_area(x0, x1, y0, y1, margin=1):
            x0 = max(start_x + self.wall_thickness_px + margin, x0)
            x1 = min(end_x   - self.wall_thickness_px - margin, x1)
            y0 = max(start_y + self.wall_thickness_px + margin, y0)
            y1 = min(end_y   - self.wall_thickness_px - margin, y1)
            return x0, x1, y0, y1

        # -------------------------
        # box 1
        # size: 0.8 m x 0.6 m
        # -------------------------
        box1_w_px = max(1, int(0.8 / self.horizontal_scale))
        box1_l_px = max(1, int(0.6 / self.horizontal_scale))

        box1_center_x = start_x + int(1.2 / self.horizontal_scale)
        box1_center_y = start_y + int(1.5 / self.horizontal_scale)

        x0 = box1_center_x - box1_w_px // 2
        x1 = x0 + box1_w_px
        y0 = box1_center_y - box1_l_px // 2
        y1 = y0 + box1_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        # -------------------------
        # box 2
        # size: 1.0 m x 0.5 m
        # -------------------------
        box2_w_px = max(1, int(1.0 / self.horizontal_scale))
        box2_l_px = max(1, int(0.5 / self.horizontal_scale))

        box2_center_x = start_x + int(3.5 / self.horizontal_scale)
        box2_center_y = start_y + int(3.2 / self.horizontal_scale) + 25

        x0 = box2_center_x - box2_w_px // 2
        x1 = x0 + box2_w_px
        y0 = box2_center_y - box2_l_px // 2
        y1 = y0 + box2_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        # -------------------------
        # box 3
        # size: 1.0 m x 1.0 m
        # -------------------------
        box3_w_px = max(1, int(1.0 / self.horizontal_scale))
        box3_l_px = max(1, int(1.0 / self.horizontal_scale))

        box3_center_x = start_x + int(3.5 / self.horizontal_scale)
        box3_center_y = start_y + int(3.2 / self.horizontal_scale) + 40

        x0 = box3_center_x - box3_w_px // 2
        x1 = x0 + box3_w_px
        y0 = box3_center_y - box3_l_px // 2
        y1 = y0 + box3_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        self.heightsamples = self.height_field_raw
        self.env_origins = np.zeros((1, 1, 3), dtype=np.float32)
        self.env_origins[0, 0] = [
            2.5,
            1.0,
            0.3,
        ]

        self.vertices = None
        self.triangles = None
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.horizontal_scale,
                self.vertical_scale,
                self.slope_treshold,
            )