import numpy as np

class FlowRotator:
    def __init__(self, x_flow, y_flow, z_flow, center_flow):
        self.x_flow = x_flow
        self.y_flow = y_flow
        self.z_flow = z_flow
        self.center_flow = center_flow

    def rotate_flow(self, flow_field:np.array, rotation_matrix:np.array)->np.array:
        xx, yy, zz = np.meshgrid(np.arange(self.x_flow),
                                np.arange(self.y_flow),
                                np.arange(self.z_flow), indexing='ij')
        coords = np.stack(
            (xx-self.center_flow[0,0], yy-self.center_flow[0,1], zz-self.center_flow[0,2]),
            axis=-1
            ) 
        rot_coords = self._rotate_coords(coords, rotation_matrix)

        valid_indices, valid_coords = self._get_valid_coords_and_indices(rot_coords)
        valid_flow_vals = flow_field[xx, yy, zz][valid_indices]

        flow_field_rotated = self._rotate_flow_vals(valid_coords, valid_flow_vals, rotation_matrix)

        return flow_field_rotated

    def _rotate_coords(self, coords:np.array, rotation_matrix:np.array)->np.array:
        rot_coords = np.dot(coords.reshape((-1, 3)), rotation_matrix)
        rot_coords = rot_coords.reshape(self.x_flow, self.y_flow, self.z_flow, 3)
        rot_coords[:, :, :, 0] += self.center_flow[0, 0]
        rot_coords[:, :, :, 1] += self.center_flow[0, 1]
        rot_coords[:, :, :, 2] += self.center_flow[0, 2]
        return rot_coords

    def _rotate_flow_vals(self, valid_coords:np.array, valid_flow_vals:np.array, rotation_matrix:np.array)->np.array:
        flow_field_rotated = np.empty([self.x_flow, self.y_flow, self.z_flow, 3])  
        flow_field_rotated[:] = np.nan
        flow_field_rotated[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = (np.dot(valid_flow_vals, rotation_matrix))
        return flow_field_rotated