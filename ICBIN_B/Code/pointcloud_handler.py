# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:25:45 2024
handler? But I don't even know her!
@author: Andrew
"""
#%% Initialize
import argparse
import os
import open3d as o3d
import pandas as pd
import numpy as np
import h5py
from sklearn.decomposition import IncrementalPCA
from alphashape import alphashape
from scipy.spatial import Delaunay
import meshio
import pyvista as pv

#%% init
def inc_PCA(bone):
    ipca = IncrementalPCA(n_components=3, batch_size=1000)
    original_coordinates=bone[['x', 'y', 'z']].values
    ipca.fit(original_coordinates)
    # Transform the original coordinates using the fitted PCA
    reoriented_coordinates = ipca.transform(original_coordinates)
    # Calculate the rotation matrix to align principal axes to z, x, y
    rotation_matrix = ipca.components_  # The columns are the principal components
    bone.loc[:,['x','y','z']]=reoriented_coordinates
    return bone, rotation_matrix

def create_stl(points, filename, depth=16):

    # Determine the bounds of the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)  # Ensure normals are consistently oriented
    
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    
    mesh.compute_vertex_normals()
    
    o3d.io.write_triangle_mesh(filename, mesh)
    
def convert_to_tet10(points, tetrahedra, filtered_densities):
    point_index_map = {tuple(p): i for i, p in enumerate(points)}
    midpoints = {}
    tet10_elements = []
    tet10_densities = []

    for tet in tetrahedra:
        tet_vertices = [tuple(points[v]) for v in tet]
        tet_density = np.mean([filtered_densities[v] for v in tet])
        tet10 = list(tet)
        for i in range(4):
            for j in range(i + 1, 4):
                edge = (tet_vertices[i], tet_vertices[j])
                if edge not in midpoints:
                    midpoints[edge] = (np.array(edge[0]) + np.array(edge[1])) / 2
                    midpoint_tuple = tuple(midpoints[edge])
                    if midpoint_tuple not in point_index_map:
                        point_index_map[midpoint_tuple] = len(points)
                        points = np.vstack([points, midpoints[edge]])
                tet10.append(point_index_map[tuple(midpoints[edge])])
        tet10_elements.append(tet10)
        tet10_densities.append(tet_density)

    return points, tet10_elements, tet10_densities

def create_tet10_vtk(points, vtk_filename, depth=16, density_threshold=0.1, visualize=False):
    # Separate points and density values
    points = np.array(points)
    xyz_points = points[:, :3]
    densities = points[:, 3]
    
    # Use density values to filter the original point cloud
    filtered_indices = densities >= np.quantile(densities, density_threshold)
    filtered_points = xyz_points[filtered_indices]
    filtered_densities = densities[filtered_indices]

    # Create a point cloud from the filtered 3D points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=16, max_nn=64))
    pcd.orient_normals_consistent_tangent_plane(k=30)  # Ensure normals are consistently oriented
    
    # Perform Poisson surface reconstruction to get mesh and densities
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    
    
    if visualize:
        o3d.visualization.draw_geometries([mesh], window_name="Poisson Surface Reconstruction")
        
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    
    if visualize:
        o3d.visualization.draw_geometries([mesh], window_name="Poisson Surface Reconstruction")
        
    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices)
    
    # Generate Delaunay triangulation of the filtered point cloud
    delaunay = Delaunay(vertices)

    # Convert the tetrahedra to Tet10 elements
    points, tet10_elements, tet10_densities = convert_to_tet10(vertices, delaunay.simplices, filtered_densities)

    # Create a Mesh object for Tet10 elements
    tet10_mesh = meshio.Mesh(
        points=points,
        cells=[("tetra10", np.array(tet10_elements))],
        cell_data={"density": [tet10_densities]}
    )

    # Write the Tet10 mesh to a VTK file
    meshio.write(vtk_filename, tet10_mesh)
    #%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory', type=str,default='../')
    parser.add_argument('-samp', type=str,default='')
    parser.add_argument('-t','--truncate', action='store_true')
    parser.add_argument('-trunc', type=int,default=30000)
    parser.add_argument('-sdep', type=int,default=6)
    parser.add_argument('-skip', type=bool,default=True)
    parser.add_argument('-reseg', type=bool,default=False)
    parser.add_argument('-recloud', action='store_true')
    parser.add_argument('-vis', action='store_true', default=False)
    parser.add_argument('-thresh', type=int, default=600,help='HU Threshold')
    parser.add_argument('-c_slope', type=float, default=0.000357,help='calibration slope')
    parser.add_argument('-c_int', type=float, default=-0.0012625,help='calibration intercept')
    parser.add_argument('-alpha', type=float, default=0.5,help='surface finish')
    parser.add_argument('-stl', action='store_true', default=False)
    parser.add_argument('-vtk', action='store_true', default=False)
    args = parser.parse_args(['-t','-stl'
                              ])
    if args.reseg:
        args.recloud = True
    if args.recloud:
        print('WIP')
    
    directory = args.directory + 'Data/'
    volumes = directory + 'Volumes/'
    samps = os.listdir(directory + 'Unfiltered_PCs/Cadaver/') + \
            os.listdir(directory + 'Unfiltered_PCs/Runner/')
            
    study_dirs = {'R15': 'Runner', 'MTSFX': 'Runner'}
    default_dir = 'Cadaver'
    
    if args.samp:
        if args.samp not in samps:
            raise ValueError(f"Sample '{args.samp}' not available")
        samps = [args.samp]
    
    for s in samps:
        stl_only = False
        sample_dir = next((study_dirs[code] for code in study_dirs if code in s), default_dir)
        MTs = os.listdir(f"{directory}Unfiltered_PCs/{sample_dir}/{s}")
        if not MTs:
            print(f'No samples found in {s}')
            continue
        
        if not os.path.exists(f'{directory}Compressed/{s}'):
            os.makedirs(f'{directory}Compressed/{s}')
        else:
            MT_comp = os.listdir(f'{directory}Compressed/{s}')
            a = [m.split('.')[0] for m in MTs]
            b = [m.split('.')[0] for m in MT_comp]
            if a == b and args.skip:
                print(f'skipped {s}')
                if args.stl:
                    print('still making stls')
                stl_only = True
        
        for MT in MTs:
            MTname = MT.split('.')[0]
            if 'L' in MTname:
                MTside = 'L'
            elif 'R' in MTname:
                MTside = 'R'
            else:
                print(f'Error in {s}/{MT}, name is not right')
                continue
            
            pc = pd.read_csv(f"{directory}Unfiltered_PCs/{sample_dir}/{s}/{MT}", header=None, sep=',')
            if len(pc.columns) == 4:
                pc.columns = ['x', 'y', 'z', 'd']
            else:
                print(f'{s}/{MT} file cannot be read')
                continue
            
            pc['d'] = np.where(
                pc['d'] > args.thresh, pc['d'] * args.c_slope + 
                args.c_int, args.thresh * args.c_slope + args.c_int)
            
            if args.truncate:
                pc['P'] = pc['d'] / pc['d'].sum()
                pc = pc.sample(n=args.trunc, weights='P', random_state=42)
                pc = pc.drop(columns=['P'])
            
            reor_pc, _ = inc_PCA(pc)
            
            try:
                MTno = int(MTname.replace(MTside,''))
            except:
                if 'MT1' in MTname:
                    MTno = 1
                elif 'MT2' in MTname:
                    MTno = 2
                elif 'MT3' in MTname:
                    MTno = 3
                elif 'MT4' in MTname:
                    MTno = 4
                elif 'MT5' in MTname:
                    MTno = 5
                    
            # Convert the point cloud DataFrame to Open3D format
            current_pcd = o3d.geometry.PointCloud()
            current_pcd.points = o3d.utility.Vector3dVector(reor_pc[['x', 'y', 'z']].values)
            
            # Continue with processing and saving...
            if args.vtk:
                create_tet10_vtk(reor_pc, directory + f'Mesh/{s}_{MT}.vtk', args.sdep, 0.1, args.vis)
            
            points = reor_pc[['x', 'y', 'z']].values
            alpha = args.alpha
            alpha_shape_mesh = alphashape(points, alpha)
            surf_pc = pd.DataFrame(np.asarray(alpha_shape_mesh.vertices), 
                                   columns=['x', 'y', 'z']).drop_duplicates()
            
            # Visualization and saving the output
            if args.vis:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(reor_pc[['x', 'y', 'z']].to_numpy())
                color_mapping = (reor_pc['d'] - reor_pc['d'].min()) / (reor_pc['d'].max() - reor_pc['d'].min())
                colors = o3d.utility.Vector3dVector(np.vstack((color_mapping, 1 - color_mapping, np.zeros_like(color_mapping))).T)
                point_cloud.colors = colors
                o3d.visualization.draw_geometries([point_cloud])
    
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(surf_pc[['x', 'y', 'z']].to_numpy())
                o3d.visualization.draw_geometries([point_cloud])
            
            if args.truncate:
                if len(surf_pc) > args.trunc // 5:
                    surf_pc = surf_pc.sample(n=args.trunc // 5, replace=True, random_state=42)
                    print('sampling surface')
                elif len(surf_pc) < 2000:
                    print('uh oh, small surface')
                    
            if not stl_only:  
                with h5py.File(f'{directory}Compressed/{s}/{MTname}.h5', 'w') as hf:
                    hf.create_dataset('Pointcloud', data=reor_pc)
                    hf.create_dataset('Surface', data=surf_pc)
                    hf.create_dataset('MTno', data=MTno)
                    hf.create_dataset('Side', data=MTside)
            
            if args.stl:
                dir_ = f'{volumes}{MTside}/MT{MTno}/{sample_dir}'
                os.makedirs(dir_, exist_ok=True)
                create_stl(surf_pc, f'{dir_}/{s}_{MTside}_MT{MTno}.stl')
            
            print(f'{s}/{MTname} complete.')

