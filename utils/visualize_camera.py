import numpy as np
import copy

def generate_circle_trajectory():
    reference_position = np.array([ 0.28423298,    -0.9363399,   4.29988871])

    num_steps = 200
    angle_step = 4 * np.pi / num_steps
    trajectory = []
    for step in range(num_steps):
        angle = step * angle_step

        # # Calculate the new position with rotation around the Y-axis
        # rotation_y = np.array([
        #     [np.cos(angle),  0, np.sin(angle)],
        #     [0,              1, 0],
        #     [-np.sin(angle), 0, np.cos(angle)]
        # ])
        # position_vector = rotation_y @ reference_position

        radius = np.linalg.norm(reference_position)

        dx = radius * np.cos(angle)
        dz = radius * np.sin(angle)
        dy = radius * 0.1 * np.cos(angle) * 0
        position_vector = np.array([dx, dy, dz])

        rotation_matrix = pos_to_rotation(position_vector)
        # rotation_matrix = np.eye(3)

        trajectory.append((rotation_matrix, position_vector))
    return trajectory


def pos_to_rotation(position_vector):
    # Z-axis (camera is pointing in the positive Z direction)
    # z_axis = -position_vector.copy()
    z_axis = position_vector.copy()
    z_axis /= np.linalg.norm(z_axis)  # Normalize
    # Y-axis (up direction)
    y_axis = np.array([0, 1, 0])
    # X-axis (right direction)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    # Recompute the Y-axis to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)

    # Assemble the rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # rotation_matrix = rotation_matrix.transpose()

    return rotation_matrix


def visualize_poses(poses, size=0.1, bounding_box=None):
    """
    Visualize camera poses in the axis-aligned bounding box, which
    can be utilized to tune the aabb size.
    """
    # poses: [B, 4, 4]

    import trimesh

    axes = trimesh.creation.axis(axis_length=4)
    # box = trimesh.primitives.Box(bounds=[[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]).as_outline()
    box = trimesh.primitives.Box(bounds=bounding_box).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


if __name__ == "__main__":
    trajectory = generate_circle_trajectory()
    poses = []
    for rotation, position in trajectory:
        pose = np.vstack((np.hstack((rotation, position[:, None])), np.array([0, 0, 0, 1])))
        poses.append(pose)
    visualize_poses(poses)