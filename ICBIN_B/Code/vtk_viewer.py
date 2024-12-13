import vtk
import argparse
import math


# Initialize the reader for VTK files based on dataset type
def load_model(file_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

# Function to compute the aspect ratio of a tetrahedron
def compute_aspect_ratio(cell, points):
    # Get the points of the tetrahedron
    p0 = points.GetPoint(cell.GetPointId(0))
    p1 = points.GetPoint(cell.GetPointId(1))
    p2 = points.GetPoint(cell.GetPointId(2))
    p3 = points.GetPoint(cell.GetPointId(3))
    
    # Compute the lengths of the edges
    edges = [
        vtk.vtkMath.Distance2BetweenPoints(p0, p1),
        vtk.vtkMath.Distance2BetweenPoints(p0, p2),
        vtk.vtkMath.Distance2BetweenPoints(p0, p3),
        vtk.vtkMath.Distance2BetweenPoints(p1, p2),
        vtk.vtkMath.Distance2BetweenPoints(p1, p3),
        vtk.vtkMath.Distance2BetweenPoints(p2, p3)
    ]
    
    # Aspect ratio is the ratio of the longest edge to the shortest edge
    max_edge = math.sqrt(max(edges))
    min_edge = math.sqrt(min(edges))
    
    return max_edge / min_edge


# Function to calculate the volume of a tetrahedron
def compute_tetrahedron_volume(cell, points):
    # Get the points of the tetrahedron (the cell contains point ids)
    p0 = points.GetPoint(cell.GetPointId(0))
    p1 = points.GetPoint(cell.GetPointId(1))
    p2 = points.GetPoint(cell.GetPointId(2))
    p3 = points.GetPoint(cell.GetPointId(3))
    
    # Volume formula for a tetrahedron
    matrix = [
        [p0[0], p0[1], p0[2], 1],
        [p1[0], p1[1], p1[2], 1],
        [p2[0], p2[1], p2[2], 1],
        [p3[0], p3[1], p3[2], 1]
    ]
    
    # Using numpy for determinant calculation
    import numpy as np
    det = np.linalg.det(np.array(matrix))
    volume = abs(det) / 6.0  # Volume of tetrahedron is determinant / 6
    
    return volume


# Function to compute the condition number of a tetrahedron
def compute_condition_number(cell, points):
    # Get the points of the tetrahedron (the cell contains point ids)
    p0 = points.GetPoint(cell.GetPointId(0))
    p1 = points.GetPoint(cell.GetPointId(1))
    p2 = points.GetPoint(cell.GetPointId(2))
    p3 = points.GetPoint(cell.GetPointId(3))
    
    # Create vectors from the tetrahedron points
    v0 = [p1[i] - p0[i] for i in range(3)]
    v1 = [p2[i] - p0[i] for i in range(3)]
    v2 = [p3[i] - p0[i] for i in range(3)]
    
    # Compute the determinant of the 3x3 matrix formed by the vectors
    matrix = [v0, v1, v2]
    
    # Condition number calculation assumes the matrix is non-singular
    # Compute the inverse and condition number
    import numpy as np
    try:
        inv_matrix = np.linalg.inv(matrix)
        condition_number = np.max(np.abs(inv_matrix))  # Maximum singular value approach
    except np.linalg.LinAlgError:
        condition_number = float('inf')  # Degenerate case
    
    return condition_number

# Function to calculate and print cell quality metrics (averages, min, max)
def compute_cell_quality(unstructured_grid):
    cells = unstructured_grid.GetCells()
    points = unstructured_grid.GetPoints()
    num_cells = cells.GetNumberOfCells()
    
    print(f"Total number of cells: {num_cells}")
    
    # Initialize variables for accumulating metrics and tracking min/max values
    total_volume = 0.0
    total_aspect_ratio = 0.0
    total_condition_number = 0.0
    min_volume = float('inf')
    max_volume = float('-inf')
    min_aspect_ratio = float('inf')
    max_aspect_ratio = float('-inf')
    min_condition_number = float('inf')
    max_condition_number = float('-inf')
    num_tetrahedrons = 0
    
    # Loop over each cell and check if it's a tetrahedron (VTK_TETRA)
    for i in range(num_cells):
        cell = unstructured_grid.GetCell(i)  # Get cell by index
        if cell.GetCellType() == vtk.VTK_TETRA:
            num_tetrahedrons += 1
            volume = compute_tetrahedron_volume(cell, points)
            aspect_ratio = compute_aspect_ratio(cell, points)
            condition_number = compute_condition_number(cell, points)
            
            # Accumulate values
            total_volume += volume
            total_aspect_ratio += aspect_ratio
            total_condition_number += condition_number
            
            # Track min/max values
            min_volume = min(min_volume, volume)
            max_volume = max(max_volume, volume)
            min_aspect_ratio = min(min_aspect_ratio, aspect_ratio)
            max_aspect_ratio = max(max_aspect_ratio, aspect_ratio)
            min_condition_number = min(min_condition_number, condition_number)
            max_condition_number = max(max_condition_number, condition_number)
    
    if num_tetrahedrons > 0:
        # Calculate and print the averages
        avg_volume = total_volume / num_tetrahedrons
        avg_aspect_ratio = total_aspect_ratio / num_tetrahedrons
        avg_condition_number = total_condition_number / num_tetrahedrons
        
        print(f"\nSummary of Cell Quality Metrics (for {num_tetrahedrons} tetrahedral cells):")
        print(f"  Volume - Min: {min_volume:.4f}, Max: {max_volume:.4f}, Average: {avg_volume:.4f}")
        print(f"  Aspect Ratio - Min: {min_aspect_ratio:.4f}, Max: {max_aspect_ratio:.4f}, Average: {avg_aspect_ratio:.4f}")
        print(f"  Condition Number - Min: {min_condition_number:.4f}, Max: {max_condition_number:.4f}, Average: {avg_condition_number:.4f}")
    else:
        print("No tetrahedral cells found.")


# Set up the rendering pipeline
def create_renderer(model):
    mapper = vtk.vtkDataSetMapper()  # Generalized mapper for all dataset types
    mapper.SetInputData(model)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Set background color to dark

    return render_window, render_window_interactor

# Main function to load and display
def main(file_path):
    model = load_model(file_path)
    
    # Compute and print cell quality metrics
    compute_cell_quality(model)
    
    # Create and start the rendering loop
    render_window, render_window_interactor = create_renderer(model)
    render_window.Render()
    render_window_interactor.Start()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="VTK Viewer for .vtk files with cell quality metrics.")
    parser.add_argument("--file", default='1')
    parser.add_argument("--model", default='diff_med_fold_512_128_128')
    
    # Parse the arguments
    args = parser.parse_args()
    direct = 'Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Generated/unprocessed/'
    file = direct + args.model + '/' + args.model + '_' + args.file + '.vtk'
    
    # Call the main function with the provided file path
    main(file)
