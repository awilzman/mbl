# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:03:05 2024

@author: arwilzman
"""
import os
import open3d as o3d
import numpy as np
import argparse

def crawl_directory_for_stl(directory):
    """Recursively find all STL files in a directory."""
    stl_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.stl'):
                stl_files.append(os.path.join(root, file))
    return stl_files

def rotate_mesh(mesh, rotation_matrix):
    """Rotate the mesh using a given rotation matrix."""
    mesh.rotate(rotation_matrix, center=(0, 0, 0))

def view_and_rotate_stl(file_path):
    """Load, view, and optionally rotate an STL file."""
    print(f"Loading {file_path}")
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()

    while True:
        # Display the mesh
        o3d.visualization.draw_geometries([mesh])

        # Prompt user for rotation options
        choice = input("Enter rotation axis (x, y, z), 'reset' to reset, or nothing to move to the next file: ").strip().lower()

        if choice == '':
            break
        elif choice == 'reset':
            mesh = o3d.io.read_triangle_mesh(file_path)
            mesh.compute_vertex_normals()
        else:
            angle = float(input(f"Enter rotation angle in degrees around {choice}-axis: "))
            angle_rad = np.deg2rad(angle)

            if choice == 'x':
                R = mesh.get_rotation_matrix_from_xyz((angle_rad, 0, 0))
            elif choice == 'y':
                R = mesh.get_rotation_matrix_from_xyz((0, angle_rad, 0))
            elif choice == 'z':
                R = mesh.get_rotation_matrix_from_xyz((0, 0, angle_rad))
            else:
                print("Invalid choice. Please enter 'x', 'y', or 'z'.")
                continue

            rotate_mesh(mesh, R)

def main(directory):
    """Crawl directory for STL files and allow the user to view and rotate them."""
    stl_files = crawl_directory_for_stl(directory)
    
    if not stl_files:
        print("No STL files found in the directory.")
        return

    for file_path in stl_files:
        view_and_rotate_stl(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl a directory for STL files and view/rotate them.")
    parser.add_argument('-d', '--directory', type=str, required=True)
    
    args = parser.parse_args(['-d','../Data/Volumes/'])
    main(args.directory)