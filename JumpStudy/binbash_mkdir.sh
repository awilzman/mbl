#!/bin/bash

study_folders=("JS_S3" "JS_S4" "JS_S5" "JS_S6" "JS_S7" "JS_S8" "JS_S9" "JS_S10" "JS_F11" "JS_F12" "JS_F13" "JS_F14" "JS_F15" "JS_F16" "JS_F17" "JS_S18" "JS_F19" "JS_S20")

base_path="Z:/_Current IRB Approved Studies/Jumping_Study/"

for folder in "${study_folders[@]}"; do
    full_path="$base_path$folder"
    
    # Perform actions within each directory
    # For example, create an empty directory
    mkdir "$full_path/new_empty_directory"
done