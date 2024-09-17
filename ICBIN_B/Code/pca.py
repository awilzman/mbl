# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:34:26 2024

@author: arwilzman
"""
import os
import argparse
import shapeworks as sw

def main(directory):
    # List STL files and parse filenames
    stl_files = [f for f in os.listdir(directory) if f.endswith('.stl')]

    bone_groups = {}
    for file in stl_files:
        parts = file.split('_')
        bone_id = parts[0]
        side = parts[1]
        bone_type = parts[2]
        
        key = (side, bone_type)
        if key not in bone_groups:
            bone_groups[key] = []
        bone_groups[key].append(os.path.join(directory, file))
    
    # Process each group of STL files
    all_particle_systems = {}
    for (side, bone_type), files in bone_groups.items():
        # Load and preprocess meshes
        meshes = [sw.Mesh(file) for file in files]
        for mesh in meshes:
            mesh.remesh(5000)  # Adjust the number according to your needs
            mesh.smooth(iterations=10)  # Optional smoothing step
        
        # Initialize and run Particle System
        particle_system = sw.ParticleSystem()
        for mesh in meshes:
            particle_system.add_mesh(mesh)
        
        particle_system.set_parameters(iterations=1000, procrustes_interval=10)
        particle_system.initialize()
        particle_system.optimize()
        
        # Save results
        mean_shape = particle_system.mean_shape()
        mean_shape_file = f'{side}_{bone_type}_mean_shape.stl'
        mean_shape.write(mean_shape_file)
        all_particle_systems[(side, bone_type)] = particle_system

        # Optionally, analyze PCA modes or other results
        pca_modes = particle_system.pca_modes()
        # Further processing or visualization of PCA modes can be done here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='../')
    args = parser.parse_args()
    directory = args.directory + 'Data/Volumes/'
    
    main(directory)