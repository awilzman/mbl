# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 00:49:30 2024

@author: arwilzman
"""

import numpy as np
import os
import argparse
import h5py
from sklearn.model_selection import train_test_split 

class AbaqusInpParser:
    def __init__(self, input_file):
        self.input_file = input_file
        self.nodes = {}
        self.elements = {}
        self.materials = {}
        self.element_material_map = {}  # Maps element IDs to materials
        self.elsets_to_elements = {}  # Maps elsets to element IDs
        self.elsets_to_materials = {}  # Maps elsets to material names

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
            if line.startswith('*NODE'):
                node_section = True
                element_section = False
            elif line.startswith('*ELEMENT'):
                node_section = False
                element_section = True
            elif line.startswith('*ELSET') and 'ES_Volume' in line:
                node_section = False
                element_section = False
                elset_name = line.split('=')[-1].strip()  # Extract elset name
                # Collect the elements belonging to this elset
                self.elsets_to_elements[elset_name] = []
            elif elset_name and not line.startswith('*'):
                start, stop, step = map(int, line.split(','))
                self.elsets_to_elements[elset_name].extend(range(start, stop + 1, step))
            elif line.startswith('*'):
                elset_name = None
                node_section = False
                element_section = False
            # Parse node data
            if node_section and not line.startswith('*'):
                parts = list(map(float, line.split(',')))
                node_id, coords = int(parts[0]), parts[1:]
                self.nodes[node_id] = coords

            # Parse element data
            elif element_section and not line.startswith('*'):
                parts = list(map(int, line.split(',')))
                elem_id, node_ids = parts[0], parts[1:]
                self.elements[elem_id] = node_ids

    def extract_materials(self, inp_data):
        """
        Second pass: Extracts material definitions and assigns them to the appropriate elset.
        """
        material_name = None

        for idx, line in enumerate(inp_data):
            if line.startswith('*MATERIAL'):
                material_name = self.extract_material_name(line)
                self.materials[material_name] = {}
            elif line.startswith('*DENSITY'):
                density_value = float(inp_data[idx + 1].strip())
                self.materials[material_name]['density'] = density_value
            elif line.startswith('*ELASTIC'):
                elastic_data = inp_data[idx + 1].split(',')[0]  # Only the first elastic value (e11)
                self.materials[material_name]['elastic'] = float(elastic_data)
            elif line.startswith('*SOLID SECTION') and 'MATERIAL=' in line:
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
            if line.startswith('*SURFACE'):
                surface_section = True
            elif surface_section and not line.startswith('*'):
                parts = line.split(',')
                elem_id = int(parts[0])  # Extract element ID
                self.surface_elements.add(elem_id)  # Mark as surface element
            elif line.startswith('*'):
                surface_section = False  # End of surface section
                
    def extract_elset_name(self, line):
        """
        Extracts elset name from *SOLID SECTION line.
        """
        parts = line.split(',')
        for part in parts:
            if 'ELSET=' in part:
                return part.split('=')[-1].strip()
        return None

    def extract_material_name(self, line):
        """
        Extracts material name from *MATERIAL line.
        """
        parts = line.split(',')
        for part in parts:
            if 'MATERIAL=' in part:
                return part.split('=')[-1].strip()
            if 'NAME=' in part:
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ABAQUS model generator')
    parser.add_argument('--directory', type=str, default='Z:/_Current IRB Approved Studies/Karens_Metatarsal_Stress_Fractures/')
    args = parser.parse_args()

    sub_dir = os.path.join(args.directory, 'Subject Data/')
    item_dir = os.path.join(args.directory, 'Cadaver_Data/')
    save_dir = 'Z:/_PROJECTS/Deep_Learning_HRpQCT/ICBIN_B/Data/inps/Labeled/'

    subs = [d for d in os.listdir(sub_dir)
            if os.path.isdir(os.path.join(sub_dir, d))
            and ('MTSFX' in d or 'R15BSI' in d)]

    items = [d for d in os.listdir(item_dir) if os.path.isdir(
        os.path.join(item_dir, d)) and ('220' in d or '2108482' in d)]

    all_data = {}

    # Process all subjects
    for sub in subs:
        print(f"Processing subject {sub}...")
        abaqus_dir = os.path.join(sub_dir, sub, 'abaqus_files')
        if os.path.isdir(abaqus_dir):
            
            if sub not in all_data:
                all_data[sub] = []
                
            inp_files = [f for f in os.listdir(abaqus_dir) if f.endswith('_new.inp')]
            
            for inp_file in inp_files:
                inp_path = os.path.join(abaqus_dir, inp_file)
                parser = AbaqusInpParser(inp_path)
                element_data = parser.process_inp_file()
                all_data[sub].append(element_data)

    # Process all items
    for item in items:
        for side in ['L', 'R']:
            print(f"Processing item {item} ({side})...")
            abaqus_dir = os.path.join(item_dir, item, side, 'abaqus_files')
            if os.path.isdir(abaqus_dir):
                
                key = f'{item}_{side}'
                if key not in all_data:
                    all_data[key] = []
                    
                inp_files = [f for f in os.listdir(abaqus_dir) if f.endswith('_new.inp')]
                
                for inp_file in inp_files:
                    inp_path = os.path.join(abaqus_dir, inp_file)
                    
                    parser = AbaqusInpParser(inp_path)
                    element_data = parser.process_inp_file()
                    all_data[key].append(element_data)
                    
# Prepare and split data into train and test sets, then save to HDF5 files.

    train_path = os.path.join(save_dir, 'train.h5')
    test_path = os.path.join(save_dir, 'test.h5')
    
    data_list = []
    labels = []
    real_labels = []
    
    for subject, data in all_data.items():
        double = False
        if '_R' in subject or '_L' in subject:
            side = '_'
        else:
            double = True
            side = '_L_'
        
        for i in [2, 3, 4]:
            if len(data) > (i - 2):
                data_list.append(data[i - 2])
                labels.append(f'MT{i}_R' if '_R' in subject else f'MT{i}_L')
                real_labels.append(f'{subject}{side}MT{i}')
    
        if double:
            side = '_R_'
            for i in [2, 3, 4]:
                if len(data) > (i + 1):
                    data_list.append(data[i + 1])
                    labels.append(f'MT{i}_R')
                    real_labels.append(f'{subject}{side}MT{i}')
    
    data_array = np.array(data_list, dtype=object)
    labels_array = np.array(labels)
    real_labels_arr = np.array(real_labels)
    
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_array, real_labels_arr, test_size=0.2, stratify=labels_array, random_state=42
    )
    
    with h5py.File(train_path, 'w') as train_h5, h5py.File(test_path, 'w') as test_h5:
        for i, label in enumerate(train_labels):
            train_h5.create_dataset(f"{label}_{i}", data=train_data[i])
        
        for i, label in enumerate(test_labels):
            test_h5.create_dataset(f"{label}_{i}", data=test_data[i])