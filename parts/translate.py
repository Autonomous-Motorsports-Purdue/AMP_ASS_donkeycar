import numpy as np

class Translate():
    def __init__(self, zed_calib, height, pitch):
        self.height = height
        self.pitch = pitch
        self.camera_matrix = np.array([[zed_calib["fx"], 0, zed_calib["cx"]],
                                [0, zed_calib["fy"], zed_calib["cy"]],
                                [0, 0, 1]])

    def run(self, point):
        v_col, u_row = point # u=y, v=x
        v_col *= 2
        u_row *= 2
        print(f"row, col: {u_row}, {v_col}")
        r = np.deg2rad(self.pitch)
        R_wc = np.array([[0, np.sin(r), np.cos(r)],
            [-1, 0, 0],
            [0, -np.cos(r), np.sin(r)]], dtype=np.float64)

        # --- 3. camera optical centre in world coords ---
        C_w = np.array([0.0, 0.0, self.height], dtype=np.float64)

        # Pixel → ray in camera frame
        ray_cam = np.linalg.inv(self.camera_matrix) @ np.array([v_col, u_row, 1.0])

        # Ray → world frame
        ray_world = R_wc @ ray_cam

        # Intersect with ground plane (Z = 0)
        if abs(ray_world[2]) < 1e-9:
            raise ValueError("Ray is parallel to the ground plane.")
        lam = -C_w[2] / ray_world[2]
        P_w = C_w + lam * ray_world
        P_w[2] = 0.0                    # enforce exact planarity
        return P_w

