import os
import pyvista as pv
import numpy as np
import argparse

def read_obj(obj_file):
    """
    Read the OBJ file using PyVista and return the mesh points and faces.
    """
    mesh = pv.read(obj_file)
    points = mesh.points  # Points in 3D space
    faces = mesh.faces  # Faces of the mesh (triangular facets)

    # Faces are in the form of [num_faces, p0, p1, p2], reshape to extract only the vertices
    faces = faces.reshape(-1, 4)[:, 1:]
    return points, faces

def find_neighbors(faces):
    """
    Find the neighbors for each triangle face based on shared edges.
    This will help in identifying the fourth vertex of each tetrahedron.
    """
    edges = {}
    # Create a dictionary to store edges
    for i, face in enumerate(faces):
        for j in range(3):
            edge = tuple(sorted([face[j], face[(j + 1) % 3]]))  # Sort so that the order of vertices doesn't matter
            if edge not in edges:
                edges[edge] = []
            edges[edge].append(i)
    
    # Find neighbors for each face based on shared edges
    neighbors = {}
    for edge, faces_list in edges.items():
        if len(faces_list) == 2:  # A valid neighboring edge
            neighbors[faces_list[0]] = faces_list[1]
            neighbors[faces_list[1]] = faces_list[0]
    
    return neighbors

def extract_tetrahedra_from_faces(points, faces, neighbors):
    tetrahedra = []
    for i, face in enumerate(faces):
        if i not in neighbors:
            continue
        neighbor_face = faces[neighbors[i]]
        common = set(face) & set(neighbor_face)
        if len(common) == 2:  # Only proceed if two vertices are shared
            non_shared_face = list(set(face) - common)
            non_shared_neighbor = list(set(neighbor_face) - common)
            if len(non_shared_face) == 1 and len(non_shared_neighbor) == 1:
                tetra = list(common) + non_shared_face + non_shared_neighbor
                if len(set(tetra)) == 4:  # Ensure uniqueness
                    tetrahedra.append(tetra)
    return tetrahedra

def generate_mid_side_nodes(points, tetrahedra):
    """
    Generate mid-side nodes for TET10 elements based on TET4 elements.
    Returns updated points and the new connectivity for TET10 elements.
    """
    mid_side_nodes = {}  # Map of edge to mid-side node index
    updated_points = points.tolist()  # List of all points, initially the same as original points
    tet10_elements = []  # Connectivity for TET10 elements

    for tet in tetrahedra:
        tet10 = list(tet)  # Start with the four corner nodes
        edges = [
            (tet[0], tet[1]), (tet[1], tet[2]), (tet[2], tet[0]),  # Base triangle
            (tet[0], tet[3]), (tet[1], tet[3]), (tet[2], tet[3])   # Edges to the apex
        ]
        
        # Add mid-side nodes for each edge
        for edge in edges:
            edge = tuple(sorted(edge))  # Sort edge to ensure consistent key
            if edge not in mid_side_nodes:
                # Calculate midpoint
                mid_x = (points[edge[0]][0] + points[edge[1]][0]) / 2
                mid_y = (points[edge[0]][1] + points[edge[1]][1]) / 2
                mid_z = (points[edge[0]][2] + points[edge[1]][2]) / 2
                mid_node_id = len(updated_points)  # Index of the new mid-side node
                updated_points.append([mid_x, mid_y, mid_z])
                mid_side_nodes[edge] = mid_node_id
            tet10.append(mid_side_nodes[edge])
        
        tet10_elements.append(tet10)

    return np.array(updated_points), tet10_elements

def extract_surface(tetrahedra):
    """
    Extracts the surface faces from the tetrahedral mesh.
    Only external faces (not shared by other tetrahedra) are included.
    """
    # Surface codes corresponding to the local face indices
    surface_codes = {
        (0, 1, 2): "S1",  # Face (0, 1, 2)
        (0, 1, 3): "S2",  # Face (0, 1, 3)
        (0, 2, 3): "S3",  # Face (0, 2, 3)
        (1, 2, 3): "S4"   # Face (1, 2, 3)
    }

    # Dictionary to track how many times each face (global nodes) appears across tetrahedra
    face_count = {}
    
    # Step 1: Loop through the tetrahedra to extract all faces with global node IDs
    for elem_id, elem in enumerate(tetrahedra):
        # Generate faces from global node IDs for this tetrahedron
        faces = [
            tuple(sorted((elem[0], elem[1], elem[2]))),  # Face (0,1,2)
            tuple(sorted((elem[0], elem[1], elem[3]))),  # Face (0,1,3)
            tuple(sorted((elem[0], elem[2], elem[3]))),  # Face (0,2,3)
            tuple(sorted((elem[1], elem[2], elem[3])))   # Face (1,2,3)
        ]
        
        # Track how many times each face occurs across tetrahedra
        for face in faces:
            if face not in face_count:
                face_count[face] = []
            face_count[face].append(elem_id)  # Keep track of the element IDs with zero-based indexing
    
    # Step 2: Collect the surface elements (faces shared by only one tetrahedron)
    surface_elements = []
    for face, elem_ids in face_count.items():
        if len(elem_ids) == 1:  # External face (not shared)
            elem_id = elem_ids[0]  # Only one element has this face
            # Step 2.1: Assign local indices (0, 1, 2, 3) to the global node IDs of the face
            local_indices = {node: idx for idx, node in enumerate(tetrahedra[elem_id])}
            local_face = tuple(sorted([local_indices[node] for node in face]))
            
            # Step 2.2: Get the corresponding surface code for the local face
            surface_face = surface_codes.get(local_face)
            if surface_face:
                surface_elements.append((elem_id + 1, surface_face))  # Abaqus IDs are 1-based
    
    return surface_elements

 
def save_mesh(refined_points, tetrahedra, surface_elements, output_file):
    """
    Save refined mesh (nodes and elements) to an Abaqus .inp file.
    """
    with open(output_file, 'w') as f_inp:
        f_inp.write("** File generated by tet4to10.py\n")
        
        # Write the nodes section
        f_inp.write("*NODE\n")
        for idx, point in enumerate(refined_points, start=1):
            f_inp.write(f"{idx}, {point[0]:.6e}, {point[1]:.6e}, {point[2]:.6e}\n")
        
        # Write the elements section (tetrahedral elements)
        f_inp.write("*ELEMENT, TYPE=C3D10\n")
        for elem_id, elem in enumerate(tetrahedra, start=1):
            f_inp.write(f"{elem_id}, " + ", ".join(map(str, [v + 1 for v in elem])) + "\n")
        
        # Write the surface elements section
        f_inp.write("*SURFACE, TYPE=ELEMENT, NAME=Surface\n")
        for elem_id, surface_face in surface_elements:
            f_inp.write(f"{elem_id},{surface_face}\n")
            
def process_and_save_files(source_folder, destination_folder, file_extension="obj"):
    """
    Process all OBJ files in the source folder, generate tetrahedral meshes, and save them in the destination folder.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    obj_files = [f for f in os.listdir(source_folder) if f.endswith(f'.{file_extension}')]

    for obj_file in obj_files:
        obj_file_path = os.path.join(source_folder, obj_file)
        
        # Read the OBJ file and extract points and faces
        points, faces = read_obj(obj_file_path)
        
        # Find neighbors (shared edges between faces)
        neighbors = find_neighbors(faces)
        
        # Extract tetrahedra from faces (using already existing points)
        tetrahedra = extract_tetrahedra_from_faces(points, faces, neighbors)
        
        # Generate mid-side nodes and TET10 elements
        refined_points, tet10_elements = generate_mid_side_nodes(points, tetrahedra)
        
        surface = extract_surface(tet10_elements)
        
        # Save to Abaqus format
        output_file = os.path.join(destination_folder, f"{os.path.splitext(obj_file)[0]}_tet10.inp")
        save_mesh(refined_points, tet10_elements, surface, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='diff_med_fold_512_128_128')
    parser.add_argument('-e', '--extension', type=str, default='obj')
    
    args = parser.parse_args()

    source_folder = f"Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Generated/tet4/{args.model}"
    destination_folder = f"Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Generated/tet10/{args.model}"

    process_and_save_files(source_folder, destination_folder, file_extension=args.extension)
