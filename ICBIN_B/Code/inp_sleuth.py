# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 00:49:30 2024

@author: arwilzman
"""

import numpy as np
import pandas as pd
import os
import argparse

class AbaqusInpParser:
    def __init__(self, input_file):
        self.input_file = input_file
        self.nodes = {}
        self.elements = {}
        self.materials = {}
        self.element_material_map = {}  # Maps element IDs to materials
        self.elsets_to_elements = {}  # Maps elsets to element IDs
        self.elsets_to_materials = {}  # Maps elsets to material names
        self.surface_elements = set() # Surface element marker

    def read_inp(self):
        with open(self.input_file, 'r') as inp_file:
            return [line.strip() for line in inp_file]
        
    def extract_nodes_elements(self, inp_data):
        """
        First pass: Extracts nodes, elements, and element set (elset) relationships.
        """
        node_section = False
        element_section = False
        elset_name = None

        for line in inp_data:
            if line.lower().startswith('*node'):
                node_section = True
                element_section = False
            elif line.lower().startswith('*element'):
                node_section = False
                element_section = True
            elif line.lower().startswith('*elset') and 'es_volume' in line.lower():
                node_section = False
                element_section = False
                elset_name = line.split('=')[-1].strip()  # Extract elset name
                if 'generate' in elset_name.lower():
                    elset_name = elset_name.split(',')[0]
                # Collect the elements belonging to this elset
                self.elsets_to_elements[elset_name] = []
            elif elset_name:
                start, stop, step = map(int, line.split(','))
                self.elsets_to_elements[elset_name].extend(range(start, stop + 1, step))
                elset_name = None
            elif line.startswith('*'):
                node_section = False
                element_section = False
            # Parse node data
            if node_section and not line.startswith('*') and line != '':
                try:
                    parts = list(map(float, line.split(',')))
                    node_id, coords = int(parts[0]), parts[1:]
                    self.nodes[node_id] = coords
                except:
                    node_section = False
                    
            # Parse element data
            elif element_section and not line.startswith('*') and line != '':
                try:
                    parts = list(map(int, line.split(',')))
                    elem_id, node_ids = parts[0], parts[1:]
                    self.elements[elem_id] = node_ids
                except:
                    element_section = False

    def extract_materials(self, inp_data):
        """
        Second pass: Extracts material definitions and assigns them to the appropriate elset.
        """
        material_name = None

        for idx, line in enumerate(inp_data):
            if line.lower().startswith('*material'):
                material_name = self.extract_material_name(line)
                self.materials[material_name] = {}
            elif line.lower().startswith('*density'):
                density_value = float(inp_data[idx + 1].strip())
                self.materials[material_name]['density'] = density_value
            elif line.lower().startswith('*elastic'):
                elastic_data = inp_data[idx + 1].split(',')[0]  # Only the first elastic value (e11)
                self.materials[material_name]['elastic'] = float(elastic_data)
            elif line.lower().startswith('*solid section') and 'material=' in line.lower():
                # This will map the elset to the material in SOLID SECTION
                elset_name = self.extract_elset_name(line)
                material_name = self.extract_material_name(line)
                self.elsets_to_materials[elset_name] = material_name
                
    def extract_surface_elements(self, inp_data):
        """
        Third pass: Extracts surface elements, marking the elements on the surface.
        """
        surface_section = False

        for line in inp_data:
            if line.lower().startswith('*surface'):
                surface_section = True
            elif surface_section and not line.startswith('*'):
                parts = line.split(',')
                elem_id = int(parts[0])  # Extract element ID
                self.surface_elements.add(elem_id)  # Mark as surface element
            elif line.startswith('*'):
                surface_section = False
        
        face_to_elements = {}
        
        for elem_id, node_ids in self.elements.items():
            # Get all faces of this element
            faces = [
                [node_ids[0], node_ids[1], node_ids[2]], 
                [node_ids[0], node_ids[1], node_ids[3]], 
                [node_ids[1], node_ids[2], node_ids[3]], 
                [node_ids[0], node_ids[2], node_ids[3]]
            ]
    
            # For each face, we update the `face_to_elements` map
            for face in faces:
                face_tuple = tuple(sorted(face))  # Sort to ensure the face is represented consistently
                if face_tuple not in face_to_elements:
                    face_to_elements[face_tuple] = []
                face_to_elements[face_tuple].append(elem_id)
    
        # Now, for each surface element, find its neighbors by looking up the shared faces
        old_surf = self.surface_elements.copy()
        for elem_id in old_surf:
            # Check for neighboring elements that share a face
            for face in faces:
                face_tuple = tuple(sorted(face))
                neighbors = face_to_elements.get(face_tuple, [])
                for neighbor in neighbors:
                    if neighbor != elem_id:
                        self.surface_elements.add(neighbor)
                
    def extract_elset_name(self, line):
        """
        Extracts elset name from *SOLID SECTION line.
        """
        parts = line.split(',')
        for part in parts:
            if 'elset=' in part.lower():
                return part.split('=')[-1].strip()
        return None

    def extract_material_name(self, line):
        """
        Extracts material name from *MATERIAL line.
        """
        parts = line.split(',')
        for part in parts:
            if 'material=' in part.lower():
                return part.split('=')[-1].strip()
            if 'name=' in part.lower():
                return part.split('=')[-1].strip()
        return None

    def create_element_data(self):
        """
        Creates a 2D array where each row corresponds to an element.
        Each row contains the x, y, z coordinates of the element's nodes, 
        a binary feature denoting surface or not, and the material's first property (e11).
        """
        data = []
        for elem_id, node_ids in self.elements.items():
            row = []
            for node_id in node_ids:
                row.extend(self.nodes.get(node_id, [0, 0, 0]))  # Fill missing nodes with zeros if not found
    
            # Determine if element is a surface element
            is_surface = 1 if elem_id in self.surface_elements else 0
            row.append(is_surface)  # Append binary surface feature
            # Find the elset to which the element belongs
            elset_name = None
            if len(self.elsets_to_elements.items()) > 0:
                for es_name, es_elements in self.elsets_to_elements.items():
                    if elem_id in es_elements:
                        elset_name = es_name.split(',')[0]
                        break
            material_name = self.elsets_to_materials.get(elset_name, None)
            material_props = self.materials.get(material_name, {}).get('elastic', 0)  # First elastic property (e11)
            row.append(material_props)  # Append elastic property
            data.append(row)
            
        return np.array(data)
    
    def calculate_average_material(self, element_ids):
        """
        Calculates the average material property (e11) for a set of elements.
        """
        material_values = []
        for elem_id in element_ids:
            # Get the elset to which the element belongs
            elset_name = None
            for es_name, es_elements in self.elsets_to_elements.items():
                if elem_id in es_elements:
                    elset_name = es_name.split(',')[0]
                    break
    
            # Get the material name assigned to the elset
            material_name = self.elsets_to_materials.get(elset_name, None)
            if material_name:
                # Get the elastic property (e11) from the material properties
                elastic = self.materials.get(material_name, {}).get('elastic', 0)
                material_values.append(elastic)
    
        if material_values:
            return sum(material_values) / len(material_values)
        return 0

    def process_inp_file(self):
        inp_data = self.read_inp()
        self.extract_nodes_elements(inp_data)  # First pass to gather nodes, elements, and elsets
        
        x_values = [coords[0] for coords in self.nodes.values()]
        y_values = [coords[1] for coords in self.nodes.values()]
        z_values = [coords[2] for coords in self.nodes.values()]
        ranges = {
            "x": (max(x_values)-min(x_values)),
            "y": (max(y_values)-min(y_values)),
            "z": (max(z_values)-min(z_values))
        }
        max_axis, max_range = max(ranges.items(), key=lambda item: item[1])
        second_largest_axis, second_largest_range = sorted(ranges.items(), key=lambda item: item[1], reverse=True)[1]
        
        # we need to make sure the inp wasn't flipped earlier by 
        # Z:\_Current IRB Approved Studies\Karens_Metatarsal_Stress_Fractures\inp_from_mimics.py
        if second_largest_axis == 'x':
            centroid_x = sum(coords[0] for coords in self.nodes.values()) / len(self.nodes)
            centroid_y = sum(coords[1] for coords in self.nodes.values()) / len(self.nodes)
            
            # Step 2: Rotate nodes in-place
            for node, coords in self.nodes.items():
                x, y, z = coords
                # Translate to centroid, rotate, and translate back in one step
                x_rotated = -(y - centroid_y) + centroid_x
                y_rotated = (x - centroid_x) + centroid_y
                self.nodes[node] = [x_rotated, y_rotated, z]
            x_values = [coords[0] for coords in self.nodes.values()]
            y_values = [coords[1] for coords in self.nodes.values()]
            z_values = [coords[2] for coords in self.nodes.values()]
            ranges = {
                "x": (max(x_values)-min(x_values)),
                "y": (max(y_values)-min(y_values)),
                "z": (max(z_values)-min(z_values))
            }
            max_axis, max_range = max(ranges.items(), key=lambda item: item[1])
            second_largest_axis, second_largest_range = sorted(ranges.items(), key=lambda item: item[1], reverse=True)[1]
        
        print(f"{max_axis}: {max_range:.3f}; {second_largest_axis}: {second_largest_range:.3f}")
            
        self.extract_materials(inp_data)  # Second pass to gather material definitions
        self.extract_surface_elements(inp_data)  # Third pass to gather surface elements
        element_data = self.create_element_data()  # Finally create the element data array
        
        surface_avg_material = self.calculate_average_material(self.surface_elements)
        non_avg_material = self.calculate_average_material(set(self.elements.keys()) - self.surface_elements)
        
        # Compare averages and print flag if cortical bone is less stiff
        if surface_avg_material < non_avg_material:
            print(f"Low cortical density in {self.input_file}")

        return element_data
    
def save_data(item,df,abaqus_dir,inp_file,encoding_folder,aug):
    if not os.path.exists(encoding_folder):
        os.makedirs(encoding_folder)
        
    inp_path = os.path.join(abaqus_dir, inp_file)
    
    parser = AbaqusInpParser(inp_path)
    element_data = parser.process_inp_file()
    
    #readodb.py required
    fe_data_file = os.path.join(abaqus_dir,inp_file.replace('.inp','_data.csv'))
    
    df_fe = pd.read_csv(fe_data_file)
    
    df_fe['elem'] = [i.split('_')[-1] for i in df_fe['step']]
    df_fe['step'] = df_fe['step'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    
    for key in df_fe['step'].unique():
        angle = 0 if 'axial' in key else int(key.split('_')[-1].replace('deg',''))
        
        load = key.split('_')[2]
        if 'N' in load:
            load = load.replace('N','')
        load = int(load)
            
        final_data = df_fe[df_fe['step']==key].drop(columns='step')
        final_data['elem'] = final_data['elem'].astype(int)
        final_data = final_data.sort_values(by='elem')
        
        zeros_ = np.zeros((element_data.shape[0], final_data.shape[1]-1))
        
        if aug:
            e_data = np.hstack((element_data[:,-1:], zeros_))
        else:
            e_data = np.hstack((element_data, zeros_))
        
        elems = final_data['elem'].values - 1
        vals = final_data.iloc[:, :-1].values 
        valid = (elems >= 0) & (elems < e_data.shape[0])
        e_data[elems[valid], -vals.shape[1]:] = vals[valid]
        
        meta_cols = 7  # number of metadata fields you're writing to head
        data_cols = e_data.shape[1]
        total_cols = max(meta_cols, data_cols)
        
        if data_cols < total_cols:
            pad_width = total_cols - data_cols
            e_data = np.pad(e_data, ((0,0),(0,pad_width)), mode='constant')
        
        head = np.zeros((1, total_cols), dtype=np.float32)
        head[0, 0] = load
        head[0, 1] = angle
                
        try:
            
            head[0, 2] = df['Age']
            head[0, 3] = df['Height_cm']
            head[0, 4] = df['Weight_kg']
            head[0, 5] = -1 if df['Sex'] == 1 else 1
            head[0, 6] = -1
        except:
            meta = df.iloc[0] # it's in a dataframe if it's runner data
            head[0, 2] = meta['Age']
            head[0, 3] = meta['Height_cm']
            head[0, 4] = meta['BW_kg']
            head[0, 5] = -1 if meta['Sex'] == 1 else 1
            head[0, 6] = 1 #runner tag
        
        final_data = np.concatenate((head, e_data), axis=0)
        
        mtno = None
        for i in range(2, 5):
            if f'MT{i}' in inp_file:
                mtno = i
                break
        
        # Default to 'R' if '_L_' not in file
        side = 'L' if '_L_' in inp_file else 'R'
        
        # Strip underscores from item
        item = item.replace('_', '')
        
        # Base file name
        file_name = f'{item}_{side}_MT{mtno}_{angle}deg_{load}N_'
        
        # Augmentation suffix
        if aug:
            aug_keys = ['n1', 'n2', 'n3', 'n4', 'p1', 'p2', 'p3', 'p4']
            suffix = next((f'{k}_raw.npy' for k in aug_keys if f'_{k}_' in inp_file), None)
            file_name += suffix if suffix else 'aug_raw.npy'
        else:
            file_name += 'raw.npy'
        
        # Save file
        np.save(os.path.join(encoding_folder, file_name), final_data)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ABAQUS model generator')
    parser.add_argument('--directory', type=str, default='Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/')
    parser.add_argument('--save_dir', type=str, default='Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/Fatigue/')
    args = parser.parse_args()
#%%
    item_dir = os.path.join(args.directory, 'Cadaver_Data/')
    items = []
    for d in os.listdir(item_dir):
        if os.path.isdir(os.path.join(item_dir,d)) and ('220' in d or '2108482' in d):
            items.append(d)
    data = {
        "2108482": {"Age": 93, "Height_cm": 65 * 2.54, "Weight_kg": 178 / 2.2, "Sex": 0},
        "2202457M": {"Age": 97, "Height_cm": 63 * 2.54, "Weight_kg": 82 / 2.2, "Sex": 0},
        "2202474M": {"Age": 83, "Height_cm": 63 * 2.54, "Weight_kg": 117 / 2.2, "Sex": 0},
        "2202556M": {"Age": 65, "Height_cm": 63 * 2.54, "Weight_kg": 169 / 2.2, "Sex": 0},
        "2203581M": {"Age": 24, "Height_cm": 64 * 2.54, "Weight_kg": 136 / 2.2, "Sex": 0},
        "2204751M": {"Age": 40, "Height_cm": 71 * 2.54, "Weight_kg": 189 / 2.2, "Sex": 1},
        "2204828M": {"Age": 60, "Height_cm": 69 * 2.54, "Weight_kg": 140 / 2.2, "Sex": 1},
        "2204869M": {"Age": 90, "Height_cm": 70 * 2.54, "Weight_kg": 195 / 2.2, "Sex": 1},
        "2205030M": {"Age": 74, "Height_cm": 61 * 2.54, "Weight_kg": 95 / 2.2, "Sex": 0},
        "2205033M": {"Age": 82, "Height_cm": 72 * 2.54, "Weight_kg": 157 / 2.2, "Sex": 1},
        "2205041M": {"Age": 69, "Height_cm": 66 * 2.54, "Weight_kg": 153 / 2.2, "Sex": 0},
        "2205048M": {"Age": 81, "Height_cm": 71 * 2.54, "Weight_kg": 163 / 2.2, "Sex": 1},
        "2205976M": {"Age": 62, "Height_cm": 66 * 2.54, "Weight_kg": 191 / 2.2, "Sex": 1},
        "2206149M": {"Age": 65, "Height_cm": 64 * 2.54, "Weight_kg": 151 / 2.2, "Sex": 0},
    }
    encoding_folder = f'{args.save_dir}FE_Cadaver/'
    
    for item in items:
        df = data[item]
        for side in ['L', 'R']:
            print(f"Processing item {item} ({side})...")
            abaqus_dir = os.path.join(item_dir, item, side, 'abaqus_files')
            
            if os.path.isdir(abaqus_dir):
                
                pert_dir = os.path.join(abaqus_dir,'perturbed')
                if os.path.isdir(pert_dir):
                    pert_files = [f for f in os.listdir(pert_dir) if f.endswith('_new.inp')]
                    for inp_file in pert_files:
                        save_data(item,df,pert_dir,inp_file,encoding_folder,True)
                        
                inp_files = [f for f in os.listdir(abaqus_dir) if f.endswith('_new.inp')]
                for inp_file in inp_files:
                    save_data(item,df,abaqus_dir,inp_file,encoding_folder,False)
                
                

    #%%
    ###
    # Subject Info
    # ID | Sex | Race | Age | Height cm | Weight kg | Weight lbs | MT inj
    # read from "MTSFX Subject Demographics.xlsx"
    sub_dir = os.path.join(args.directory, 'Subject Data/')
    subs = []
    for d in os.listdir(sub_dir):
        if os.path.isdir(os.path.join(sub_dir, d)) and ('R15BSI' in d or 'MTSFX' in d):
            subs.append(d)
            
    df_subs = pd.read_excel(args.directory+'Subject Data/MTSFX Subject Demographics.xlsx',
                            sheet_name='Demographics', header=0)
    
    encoding_folder = f'{args.save_dir}FE_Runner/'
            
    #Process all subjects
        
    for sub in subs:
        print(f"Processing subject {sub}...")
        abaqus_dir = os.path.join(sub_dir, sub, 'abaqus_files')
        if os.path.isdir(abaqus_dir):
            df = df_subs[df_subs['Participant ID']==sub]
            
            if os.path.isdir(abaqus_dir):
                
                
                inp_files = [f for f in os.listdir(abaqus_dir) if f.endswith('_new.inp')]
                for inp_file in inp_files:
                    save_data(sub,df,abaqus_dir,inp_file,encoding_folder,False)
                pert_dir = os.path.join(abaqus_dir,'perturbed')
                if os.path.isdir(pert_dir):
                    pert_files = [f for f in os.listdir(pert_dir) if f.endswith('_new.inp')]
                    for inp_file in pert_files:
                        save_data(sub,df,pert_dir,inp_file,encoding_folder,True)
                
                
