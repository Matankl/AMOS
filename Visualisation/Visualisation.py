import numpy as np
import cv2
from skimage import measure
import pyvista as pv
import os


def create_3d_models(image_paths, label_color_map=None, offset = 0):
    """
    Create 3D mesh models for each label from a stack of 2D label images.

    Parameters
    ----------
    image_paths : list of str
        A sorted list of file paths pointing to the 2D label images.
        Each image contains integer labels (0 for background, 1–15 for organs).
    label_color_map : dict, optional
        A dictionary mapping label -> color (string or RGB tuple).
        For example: {1: "red", 2: (0, 1, 0), ...}
        If None, default colors will be assigned.

    Returns
    -------
    dict
        A dictionary where:
        - keys are integer labels (1–15)
        - values are dicts containing:
            "mesh": pyvista.PolyData mesh for that label
            "color": color for that label
    """
    # 1) Read all slices and stack into a 3D volume
    volume_slices = []
    for path in image_paths:
        # Read as grayscale or as is (assuming single-channel label images)
        label_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        volume_slices.append(label_img)
    volume = np.stack(volume_slices, axis=0)  # shape: (depth, height, width)

    # 2) Prepare output dictionary
    mesh_dict = {}

    # If no label_color_map is provided, define some default colors
    if label_color_map is None:
        default_colors = [
            "red", "green", "blue", "yellow", "magenta", "cyan",
            "orange", "purple", "brown", "pink", "lime", "teal",
            "navy", "olive", "gray"
        ]
        label_color_map = {}
        # Fill color map for labels 1–15
        for lbl in range(1, 16):
            color_index = (lbl - 1) % len(default_colors)
            label_color_map[lbl] = default_colors[color_index]

    # 3) For each label (1–15), extract a surface mesh
    for lbl in range(1, 16):
        # Create a binary mask for the current label
        label_mask = (volume == lbl).astype(np.uint8)
        if np.any(label_mask):
            # Run marching cubes to get vertices and faces
            # The level=0.5 is used to identify boundaries in the binary volume
            verts, faces, normals, values = measure.marching_cubes(
                label_mask,
                level=0.5,
                spacing=(1.0, 1.0, 1.0)  # adjust if voxel spacing is different
            )

            # Apply offset to the z-coordinates
            height_offset = offset  # Define your height offset between slices
            verts[:, 0] += verts[:, 0] * height_offset  # Adjust z-coordinates

            # Convert faces from (n, 3) to a format suitable for PyVista
            # PyVista expects [n_vertices_per_face, v1, v2, v3, ...]
            faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)

            # Create a PyVista PolyData mesh
            mesh = pv.PolyData(verts, faces_pv)

            # smoothing the edges
            smoothed_mesh = mesh.smooth(n_iter=20, relaxation_factor=0.10)  # Adjust parameters as needed

            # Store the mesh along with the color in the dictionary
            mesh_dict[lbl] = {
                "mesh": smoothed_mesh,
                "color": label_color_map.get(lbl, "white")
            }
        else:
            # If that label doesn't exist in the volume, skip it
            continue

    return mesh_dict

def show_3d_model(mesh_dict, labels_to_show=None, background_color="white"):
    """
    Show an interactive 3D view of the meshes generated for each label.

    Parameters
    ----------
    mesh_dict : dict
        Dictionary returned by create_3d_models.
        Keys = labels, Values = { "mesh": <pyvista.PolyData>, "color": <color> }
    labels_to_show : list of int, optional
        Which labels to show. If None, show all available labels.
    background_color : str or tuple
        Background color for the 3D viewer. Default is "white".
    """
    plotter = pv.Plotter()
    plotter.background_color = background_color

    # If no specific labels are requested, show all
    if labels_to_show is None:
        labels_to_show = list(mesh_dict.keys())

    for lbl, mesh_info in mesh_dict.items():
        if lbl in labels_to_show:
            mesh = mesh_info["mesh"]
            color = mesh_info["color"]
            plotter.add_mesh(
                mesh,
                color=color,
                opacity=1.0,  # adjust if you want semi-transparency
                show_edges=False  # set to True to see mesh edges
            )

    plotter.show()

if __name__ == "__main__":

    amos_test_label_path = r"D:\Database\Images\amos22 (1)\amos22\Test\label"
    ct_names = set([name.split("_")[1] for name in os.listdir(amos_test_label_path)])
    ct_images = [[name for name in os.listdir(amos_test_label_path) if ct_num in name] for ct_num in ct_names]

    key = lambda x: int(x.split("_")[-1].replace("slice", "").replace(".png", ""))
    sorted_image_paths = []
    for ct in ct_images:
        ct.sort(key=key)
        sorted_image_paths.append(ct)


    # 2) (Optional) Define a custom color map for some or all labels
    #    Label -> color (string or (r, g, b) in [0,1])
    # organ_colors = {
    #     1: "red",
    #     2: "green",
    #     3: "blue",
    #     4: "yellow",
    #     5: (0.5, 0, 0.5),   # purple in RGB
    #     # ...
    # }

    # 3) Generate the 3D mesh dictionary
    organ_colors = None
    sorted_single_ct_images = sorted_image_paths[0]
    add_path = lambda path : os.path.join(amos_test_label_path, path)
    sorted_single_ct_images = list(map(add_path, sorted_single_ct_images))

    print(f"creating mesh")
    meshes = create_3d_models(sorted_single_ct_images, label_color_map=organ_colors, offset= 5)

    # 4) Show the 3D scene, e.g. only labels 1, 2, and 5
    print(f"showing mesh")
    show_3d_model(meshes, labels_to_show=None)
