# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:25:45 2024
handler? But I don't even know her!
@author: Andrew
"""
#%% Initialize
import argparse
import os, re
import open3d as o3d
import pandas as pd
import numpy as np
import h5py
from sklearn.decomposition import IncrementalPCA
from alphashape import alphashape
from scipy.spatial import Delaunay
import meshio
import pyvista as pv
import gmsh

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

def create_stl(points, filename, depth=16, view=False):
    
    # Ensure filename has no extension
    filename = os.path.splitext(filename)[0]
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    base = os.path.basename(filename)
    base_name = re.sub(r'_\d+$', '', base)

    existing = [f for f in os.listdir(folder) if f.startswith(base_name + '_')]
    nums = []
    for f in existing:
        match = re.match(rf'{re.escape(base_name)}_(\d+)\.stl$', f)
        if match:
            nums.append(int(match.group(1)))

    n = 1 + max(nums) if nums else 0
    filename = os.path.join(folder, f"{base_name}_{n}")
    
    # Step 1: Point cloud to surface mesh (Poisson reconstruction)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh = mesh.remove_non_manifold_edges()
    
    mesh.compute_vertex_normals()
    if not (mesh.is_edge_manifold() and mesh.is_watertight()):
        print("Mesh is not watertight or edge-manifold. Exiting.")
        return -1
    
    # Optional visualization of the surface mesh
    if view:
        o3d.visualization.draw_geometries([mesh], window_name="Mesh Preview")
        confirm = input("Save the mesh as STL? (y/n): ")
        if confirm.lower() != 'y':
            print("Mesh not saved.")
            return 0
    
    # Save the surface mesh as STL
    o3d.io.write_triangle_mesh(filename + '.stl', mesh, write_ascii=False)
    
    # Step 2: Convert Open3D mesh to GMSH mesh format
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.4)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    gmsh.option.setNumber("Mesh.RefineSteps", 10)
    
    gmsh.model.add("TetrahedralMesh")
    
    # Add points to GMSH
    gmsh_points = []
    for i, (x, y, z) in enumerate(vertices):
        gmsh_points.append(gmsh.model.geo.addPoint(x, y, z))
    
    # Add surface (triangular) faces to GMSH
    gmsh_surfaces = []
    for face in faces:
        p1, p2, p3 = [gmsh_points[i] for i in face]
        line1 = gmsh.model.geo.addLine(p1, p2)
        line2 = gmsh.model.geo.addLine(p2, p3)
        line3 = gmsh.model.geo.addLine(p3, p1)
        wire = gmsh.model.geo.addCurveLoop([line1, line2, line3])
        surface = gmsh.model.geo.addPlaneSurface([wire])
        gmsh_surfaces.append(surface)
    
    gmsh.model.geo.synchronize()  # Sync all points, lines, and surfaces
    
    surface_loop = gmsh.model.geo.addSurfaceLoop(gmsh_surfaces)
    volume = gmsh.model.geo.addVolume([surface_loop])
    gmsh.model.geo.synchronize()  # Sync everything again
    
    gmsh.model.mesh.setTransfiniteAutomatic()
    
    # Generate the mesh
    gmsh.model.mesh.generate(3)
    
    
    gmsh.write(filename + ".vtk")
    
    # Optionally view the mesh using GMSH's viewer
    if view:
        gmsh.fltk.run()
    
    gmsh.finalize()
    
    print(f"Mesh saved as {filename}.stl and {filename}.vtk")
    return 1
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

