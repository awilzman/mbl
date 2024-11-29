import os
import pyvista as pv
import numpy as np
import argparse
import vtk

class TetMeshProcessor:
    def __init__(self, source_folder, destination_folder, file_extension="obj"):
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.file_extension = file_extension
        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)

    def read_surface_mesh(self, file_path):
        """
        Read the surface mesh from an OBJ file using PyVista.
        """
        print(f"Reading surface mesh from {file_path}")
        return pv.read(file_path)

    def refine_mesh(self, surface_mesh, max_edge_length):
        """
        Refine the surface mesh by subdividing edges longer than the given length.
        """
        print(f"Refining surface mesh to max edge length: {max_edge_length}")
        refined_mesh = surface_mesh.subdivide_adaptive(max_edge_length)
        return refined_mesh

    def create_volume_mesh(self, surface_mesh):
        """
        Generate a volumetric mesh from the surface.
        """
        print("Generating volume mesh...")
        del_filter = vtk.vtkDelaunay3D()
        del_filter.SetInputData(surface_mesh)
        del_filter.Update()
        volume_mesh = pv.wrap(del_filter.GetOutput())
        
        return volume_mesh
    
    def refine_surface_mesh(self, surface_mesh, iterations=20, relaxation=0.5):
        """
        Smooth the surface mesh to improve aspect ratio before volumetric meshing.
        """
        print("Refining surface mesh...")
        if not surface_mesh.is_all_triangles():
            surface_mesh = surface_mesh.triangulate()
        
        surface_mesh = surface_mesh.subdivide_adaptive(max_edge_length=self.max_edge_length)
        surface_mesh = surface_mesh.clean()
        
        smooth_filter = vtk.vtkSmoothPolyDataFilter()
        smooth_filter.SetInputData(surface_mesh)
        smooth_filter.SetNumberOfIterations(iterations)
        smooth_filter.SetRelaxationFactor(relaxation)
        smooth_filter.FeatureEdgeSmoothingOff()
        smooth_filter.BoundarySmoothingOn()
        smooth_filter.Update()

        return pv.wrap(smooth_filter.GetOutput())

    def fix_degenerate_triangles(self, surface_mesh):
        """
        Fix degenerate triangles by subdividing long edges and cleaning non-manifold edges.
        """
        print("Fixing degenerate triangles...")
        # Subdivide to ensure consistent edge lengths
        subdivided = surface_mesh.subdivide_adaptive(2.0)
        cleaned = subdivided.clean()
        return cleaned
    
    def enforce_aspect_ratio(self, volume_mesh, target_ratio=1.0, max_iterations=20, tolerance=0.1):
        import vtk
    
        for _ in range(max_iterations):
            quality_filter = vtk.vtkMeshQuality()
            quality_filter.SetInputData(volume_mesh)
            quality_filter.SetTriangleQualityMeasureToAspectFrobenius()
            quality_filter.Update()
            
            output = quality_filter.GetOutput()
            quality_array = output.GetCellData().GetArray("Quality")
    
            aspect_ratios = np.array([quality_array.GetValue(i) for i in range(
                quality_array.GetNumberOfValues())])
            mean_ratio = np.mean(aspect_ratios)
            if abs(mean_ratio - target_ratio) < tolerance:
                break
            
            # Convert UnstructuredGrid to PolyData for smoothing
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(volume_mesh.GetPoints())  # Set the points
    
            # Manually set the cells (connectivity) for the vtkPolyData
            cell_array = vtk.vtkCellArray()
            for i in range(volume_mesh.GetNumberOfCells()):
                cell = volume_mesh.GetCell(i)
                num_points = cell.GetNumberOfPoints()
                cell_array.InsertNextCell(num_points)
                for j in range(num_points):
                    point_id = cell.GetPoints()[j]  # Get point ID instead of using subscript
                    cell_array.InsertCellPoint(point_id)  # Insert point into the cell
    
            poly_data.SetCells(volume_mesh.GetCellTypes(), cell_array)
    
            # Apply the smoothing filter
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputData(poly_data)
            smoother.SetNumberOfIterations(5)
            smoother.Update()
    
            # Get the smoothed mesh as vtkUnstructuredGrid if needed
            smoothed_polydata = smoother.GetOutput()
    
            # Convert back to UnstructuredGrid (if needed)
            volume_mesh = pv.wrap(smoothed_polydata)
    
        return volume_mesh

    def process_file(self, file_path, max_edge_length, target_ratio, max_volume):
        """
        Process a single OBJ file: read, refine, generate volumetric mesh, and save TET10.
        """
        # Read surface mesh
        surface_mesh = self.read_surface_mesh(file_path)

        # Refine surface mesh
        refined_surface = self.refine_mesh(surface_mesh, max_edge_length)

        # Create volumetric mesh
        volume_mesh = self.create_volume_mesh(refined_surface)

        # Re-enforce aspect ratio after coarsening
        #volume_mesh = self.enforce_aspect_ratio(volume_mesh, target_ratio=target_ratio)

        # Extract TET4 elements
        tetrahedra, points = self.extract_tet4(volume_mesh)

        # Convert TET4 to TET10
        refined_points, tet10_elements = self.generate_tet10(points, tetrahedra)

        # Save TET10 to file
        output_file = os.path.join(
            self.destination_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_tet10.inp"
        )
        self.save_to_inp(refined_points, tet10_elements, output_file)

    def process_files(self, max_edge_length=2, target_ratio=1.0, max_volume=1):
        """
        Process all OBJ files in the source folder with the specified refinement parameters.
        """
        obj_files = [f for f in os.listdir(self.source_folder) if f.endswith(f'.{self.file_extension}')]
        if not obj_files:
            print("No OBJ files found to process.")
            return

        for obj_file in obj_files:
            file_path = os.path.join(self.source_folder, obj_file)
            print(f"Processing file: {file_path}")
            self.process_file(file_path, max_edge_length, target_ratio, max_volume)

    def extract_tet4(self, volume_mesh):
        """
        Extract TET4 connectivity and points from the volumetric mesh.
        """
        print("Extracting TET4 elements...")
        tetrahedra = volume_mesh.cells_dict[10]  # VTK_TETRA = 10
        return tetrahedra.reshape(-1, 4)+1, volume_mesh.points

    def generate_tet10(self, points, tetrahedra):
        """
        Generate mid-side nodes for TET10 elements based on TET4 elements.
        """
        print("Generating TET10 elements...")
        mid_side_nodes = {}  # Map of edge to mid-side node index
        updated_points = points.tolist()  # List of all points, initially the same as original points
        tet10_elements = []  # Connectivity for TET10 elements

        for tet in tetrahedra:
            tet10 = list(tet)  # Start with the four corner nodes
            edges = [
                (tet[0], tet[1]), (tet[1], tet[2]), (tet[2], tet[0]),  # Base triangle
                (tet[0], tet[3]), (tet[1], tet[3]), (tet[2], tet[3])   # Edges to the apex
            ]

            for edge in edges:
                edge = tuple(sorted(edge))  # Sort edge to ensure consistent key
                if edge not in mid_side_nodes:
                    # Calculate midpoint
                    mid_x = (points[edge[0] - 1][0] + points[edge[1] - 1][0]) / 2
                    mid_y = (points[edge[0] - 1][1] + points[edge[1] - 1][1]) / 2
                    mid_z = (points[edge[0] - 1][2] + points[edge[1] - 1][2]) / 2
                    mid_node_id = len(updated_points)+1  # Index of the new mid-side node
                    updated_points.append([mid_x, mid_y, mid_z])
                    mid_side_nodes[edge] = mid_node_id
                tet10.append(mid_side_nodes[edge])

            tet10_elements.append(tet10)

        return np.array(updated_points), tet10_elements

    def save_to_inp(self, points, tetrahedra, output_file):
        """
        Save refined mesh (nodes and elements) to an Abaqus .inp file.
        """
        print(f"Saving TET10 mesh to {output_file}")
        with open(output_file, 'w') as f_inp:
            f_inp.write("** File generated by TetMeshProcessor\n")
            f_inp.write("*Node\n")
            for idx, point in enumerate(points, start=1):
                f_inp.write(f"{idx}, {point[0]:.6e}, {point[1]:.6e}, {point[2]:.6e}\n")

            f_inp.write("*Element, type=C3D10\n")
            for elem_id, elem in enumerate(tetrahedra, start=1):
                f_inp.write(f"{elem_id}, " + ", ".join(map(str, [v + 1 for v in elem])) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='diff_med_fold_512_128_128')
    parser.add_argument('-e', '--extension', type=str, default='obj')
    
    args = parser.parse_args()

    source_folder = f"Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Generated/tet4/{args.model}"
    destination_folder = f"Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Generated/tet10/{args.model}"

    processor = TetMeshProcessor(source_folder, destination_folder, file_extension="obj")
    processor.process_files(max_edge_length=2, target_ratio=1.0, max_volume=1)
