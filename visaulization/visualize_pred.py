import numpy as np
import open3d as o3d
import os
import warnings


print(os.getcwd())

colors = [
    [1, 1, 0],  # cone: yellow
    [1, 0, 0],  # vehicle: red
    [0, 0, 1],  # human: blue
    [0, 1, 0]  # cone: yellow
]



def load_label(label_path: str):
    annotation_file = open(label_path, 'r')
    bboxes = []
    lines = annotation_file.readlines()
    for line in lines[1:]:
        tokens = line.split(",")
        x = float(tokens[0])
        y = float(tokens[1])
        z = float(tokens[2])
        h = float(tokens[3])
        w = float(tokens[4])
        l = float(tokens[5])
        # velX = float(tokens[6])
        # velY = float(tokens[7])
        yaw = float(tokens[6])
        cls = int(tokens[7])
        score = float(tokens[8])
        if score > 0.2:
            bboxes.append([x, y, z, h, w, l, yaw, cls])

        # box = [h,w,l,x,y,z,rot]
    return bboxes


def load_bin(bin_path: str):
    with open(bin_path, 'rb') as f:
        points = np.fromfile(f, dtype=np.float32)
    points = points.reshape(-1, 4)
    return points


def create_bounding_box(center, extent, yaw):
    # Create an oriented bounding box
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = extent
    # Apply rotation
    R = bbox.get_rotation_matrix_from_xyz((0, 0, yaw))
    bbox.rotate(R, center=center)
    return bbox


def filter_bounding_boxes(point_cloud, bounding_boxes, min_points_threshold=50):
    """
    Filters out bounding boxes with fewer points than the specified threshold.
    
    Args:
    - point_cloud (o3d.geometry.PointCloud): The original point cloud data.
    - bounding_boxes (list of dict): List of predicted bounding boxes, each containing 'center', 'size', and 'rotation'.
    - min_points_threshold (int): The minimum number of points required to keep a bounding box.
    
    Returns:
    - filtered_bounding_boxes (list of dict): The filtered list of bounding boxes.
    """
    filtered_bounding_boxes = []

    # Convert the point cloud to a NumPy array
    points = np.asarray(point_cloud.points)

    for bbox in bounding_boxes:
        # Extract bounding box parameters
        center = bbox['center']
        size = bbox['size']  # [width, length, height]
        rotation = bbox['rotation']  # Assume rotation is around the Z-axis

        # Create an oriented bounding box in Open3D
        oriented_bbox = o3d.geometry.OrientedBoundingBox(center, 
                                                         o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0, 0, rotation]), 
                                                         size)
        
        # Get indices of points inside the bounding box
        indices = oriented_bbox.get_point_indices_within_bounding_box(points)
        num_points_in_bbox = len(indices)

        # Check if the number of points meets the threshold
        if num_points_in_bbox >= min_points_threshold:
            filtered_bounding_boxes.append(bbox)

    return filtered_bounding_boxes


def visualize(points: np.array, boxes: np.array, labels=None):
    # create pcd object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    geometries = [pcd]
    for i, box in enumerate(boxes):
        center = box[:3]
        extent = box[3:6]
        yaw = box[6]  # assuming yaw is provided in the boxes array

        # Create an oriented bounding box in Open3D
        oriented_bbox = o3d.geometry.OrientedBoundingBox(center, 
                                                         o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0, 0, yaw]), 
                                                         extent)

        # Get indices of points inside the bounding box
        indices = oriented_bbox.get_point_indices_within_bounding_box(pcd.points)
        num_points_in_bbox = len(indices)
        print(f"number of points in the box: {num_points_in_bbox}")

        # Check if the number of points meets the threshold
        if num_points_in_bbox < 40:
            continue

        bbox = create_bounding_box(center, extent, yaw)
        if labels is not None:
            # Set bounding box color based on label
            bbox.color = colors[labels[i]-1]
        else:
            bbox.color = [1,0,0]
        geometries.append(bbox)
    # Visualize point cloud
    o3d.visualization.draw_geometries(geometries)


# def draw_visualization(points, bboxes):
    # Convert NumPy array to Open3D point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[:,:3])

    # # draw bboxes in o3d
    # entities_to_draw = []
    # colors = []
    # for box in bboxes:
    #     center_point = box[0:3]
    #     h, w, l = box[3], box[4], box[5]
    #     rotation = -box[6]

    #     # Standard 3x3 rotation matrix around the Z axis
    #     rotation_matrix = np.array(
    #         [[np.cos(rotation), -np.sin(rotation), 0.0], [np.sin(rotation), np.cos(rotation), 0.0], [0.0, 0.0, 1.0]])

    #     box3d = o3d.geometry.OrientedBoundingBox(center_point, rotation_matrix, [h, w, l])

    #     lines = [[0, 1],  # bottom back
    #             [1, 7],  # right back
    #             [7, 2],  # top back
    #             [2, 0],  # left back
    #             [3, 6],  # bottom front
    #             [6, 4],  # right front
    #             [4, 5],  # top front
    #             [5, 3],  # left front
    #             [0, 3],  # bottom left
    #             [1, 6],  # bottom right
    #             [2, 5],  # top left
    #             [7, 4]  # top right
    #             ]
    #     # colors = [[1, 0, 0] for i in range(len(lines))]  # Red color for each line
    #     if box[-1] == 0:
    #         color = [255, 0, 0]  # Box color would be red box.color = [R,G,B]
    #     elif box[-1] == 1:
    #         color = [0, 0, 255]
    #     else:
    #         color = [0, 255, 0]
    #     colors.append(color)
    #     line_set = o3d.geometry.LineSet(
    #         points=o3d.utility.Vector3dVector(box3d.get_box_points()),
    #         lines=o3d.utility.Vector2iVector(lines),
    #     )
    #     line_set.colors = o3d.utility.Vector3dVector(colors)

    #     entities_to_draw.append(line_set)
    # all_geometries = [pcd] + entities_to_draw
    # o3d.visualization.draw_geometries(all_geometries)

# # visualize a single frame
# points = load_bin('data/bin/000000.bin')
# bboxes = load_label('data/outputs/000000.txt')
# draw_visualization(points, bboxes)
# visualize all frames
bin_ROOT = "data/bin"
output_ROOT = "data/outputs"
for bin_file in sorted(os.listdir(bin_ROOT)):
    points = load_bin(os.path.join(bin_ROOT, bin_file))
    label_file_name = bin_file.split('.')[0] + '.txt'
    if os.path.exists(os.path.join(output_ROOT, label_file_name)):
        bboxes = load_label(os.path.join(output_ROOT, label_file_name))
        print(f"==========visualize frame {bin_file}=============")
        visualize(points, bboxes)
    else:
        warnings.warn(
            f"output lable for file {os.path.join(bin_ROOT, bin_file)} is not found!", UserWarning)

#
