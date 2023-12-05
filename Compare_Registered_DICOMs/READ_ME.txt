The three folders contain DICOM stacks that could be output by using
	functions laid out in iplfe_register3d.txt
	I could not get this text file to copy over correctly to the 
	Scanco software, but if you run the 'iplfe' command (without 
	quotes) in the Scanco software terminal, you should be able 
	to run the commands on two scans of your choice. This code
	was built to register two scans of the same bone, imaged
	at different times. 
	
The MATLAB code will chew through these DICOM stacks and ask for 
	two key pieces of information: the AP axis and medial side. 
	For the AP axis, please choose the anterior side first.
	
The scan 1, 2, and difference figures can be navigated by slice #

The BV, BMC, BMD variables are tables that report TOTAL bone volume,
	bone mineral content, and bone mineral density in the anterior
	medial, anterior lateral, posterior medial, and posterior lateral 
	quadrants, respectively. The units are dependent on the units in 
	the DICOMS, which are HU. These need to be converted, 
	but the area and volume units end up based in centimeters. 
	
Voxel resolution should be changed in the code if necessary, 
and the calibration slope and intercept should be machine-specific.


-Andrew Wilzman: arwilzman@wpi.edu	