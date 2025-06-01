// Proprietary Files (ND2, CZI, OIF etc to TIF)

// Author: 	Luke Hammond
// Cellular Imaging | Zuckerman Institute, Columbia University
// Date:	November 3, 2021
// Updated: August 6, 2024 - added rescaling and jpgMIPs
// Updated: September 25, 2024 - fixed bug for composite if single channel

// Converts other formats to tif z-stacks and/or max ip images 
// background subtraction optional
// updated to create jpeg folder, and also gracefully handle multichannel 2D images


// Initialization
requires("1.53c");
run("Options...", "iterations=3 count=1 black edm=Overwrite");
run("Colors...", "foreground=white background=black selection=yellow");
run("Clear Results"); 
run("Close All");

// Parameters
#@ File[] listOfPaths(label="select files or folders", style="both")
#@ Integer(label="Background Subtraction (rolling ball radius in px, 0 if none):", value = 0, style="spinner") BGSub
#@ boolean(label="Export MIPs:", description=".") MIPon
#@ boolean(label="Export autoscaled jpeg MIPs:", description=".") JMIPon


run("Input/Output...", "jpeg=100")
TIFon = true
start = getTime();
setBatchMode(true);

print("\\Clear");
print("\\Update0:File Conversion running...");
print("\\Update1: "+listOfPaths.length+" folders selected for processing.");


for (FolderNum=0; FolderNum<listOfPaths.length; FolderNum++) {

	input=listOfPaths[FolderNum];
	if (File.exists(input)) {
    	if (File.isDirectory(input) == 0) {
        	print(input + "Is a file, please select only directories containing brain datasets");
        } else {
        	
	        print("\\Update2:  Processing folder "+FolderNum+1+": " + input + " ");
	
	
			//process folder
			input = input +"/";
			files = getFileList(input);	
			files = ImageFilesOnlyArray( files );
			
			
			
			
				
			if (JMIPon == true) {
				//make output directory
				File.mkdir(input + "JPEG_MIP");
				JMIPoutput = input + "JPEG_MIP/";
				}
			
			if (TIFon == true) {
				File.mkdir(input + "TIF");
				TIFoutput = input + "TIF/";
				}
			
			//process files
			
			for(i=0; i<files.length; i++) {	
				print("\\Update3:   Processing Image " + (i+1) +" of " + files.length +".");
				// open image
				run("Bio-Formats", "open=["+input + files[i]+"] color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
				outputname = short_title(files[i]);
				getDimensions(width, height, channels, slices, frames);
				
				//make output directory
				if (MIPon == true && slices > 1 && i ==0) {
					//make output directory
					File.mkdir(input + "MIP");
					MIPoutput = input + "MIP/";
				}
				
				//bg subtract
				if (BGSub > 0) { 
					run("Subtract Background...", "rolling="+BGSub+" stack");
				}
				
				// save tifs
				if (channels > 1){
						Stack.setDisplayMode("composite");
					}
				for (j = 1; j <= channels; j++) {
    				Stack.setChannel(j);
    				run("Enhance Contrast", "saturated=0.35");
				}
				
				save(TIFoutput + outputname+".tif");
				
				
				if (MIPon == true && slices > 1){
					// create max ip
					run("Z Project...", "projection=[Max Intensity]");
					if (channels > 1){
						Stack.setDisplayMode("composite");
					}
					for (j = 1; j <= channels; j++) {
	    			Stack.setChannel(j);
	    			run("Enhance Contrast", "saturated=0.35");
					}	
					// save image
					save(MIPoutput + outputname +".tif");
					
					if (JMIPon == true) {
						saveAs("Jpeg", JMIPoutput + outputname+".jpg");
					}
					close();
				}
						
			
				if (MIPon == false && JMIPon == true && slices > 1){
					// create max ip
					run("Z Project...", "projection=[Max Intensity]");
					if (channels > 1){
						Stack.setDisplayMode("composite");
					}
					for (j = 1; j <= channels; j++) {
	    			Stack.setChannel(i);
	    			run("Enhance Contrast", "saturated=0.35");
					}	
					saveAs("Jpeg", JMIPoutput + outputname+".jpg");
					close();
				}
				
				if (JMIPon == true && slices == 1){
					// create max ip
					
					if (channels > 1){
						Stack.setDisplayMode("composite");
					}
					for (j = 1; j <= channels; j++) {
	    			Stack.setChannel(j);
	    			run("Enhance Contrast", "saturated=0.35");
					}	
					saveAs("Jpeg", JMIPoutput + outputname+".jpg");
					close();
				}
				
				close("*");
				run("Collect Garbage");
				
			}
        }
	}
}

end = getTime();
time = (end-start)/1000/60;
print("Processing time =", time, "minutes");			


function ImageFilesOnlyArray (arr) {
	//pass array from getFileList through this e.g. NEWARRAY = ImageFilesOnlyArray(NEWARRAY);
	setOption("ExpandableArrays", true);
	f=0;
	files = newArray;
	for (i = 0; i < arr.length; i++) {
		if(endsWith(arr[i], ".tif") || endsWith(arr[i], ".tiff") || endsWith(arr[i], ".oir") ||endsWith(arr[i], ".nd2") || endsWith(arr[i], ".LSM") || endsWith(arr[i], ".czi") || endsWith(arr[i], ".jpg") || endsWith(arr[i], ".ets")) {   //if it's a tiff image add it to the new array
			files[f] = arr[i];
			f = f+1;
		}
	}
	arr = files;
	arr = Array.sort(arr);
	return arr;
}

function short_title(imagename){
	nl=lengthOf(imagename);
	nl2=nl-4;
	Sub_Title=substring(imagename,0,nl2);
	Sub_Title = replace(Sub_Title, "(", "_");
	Sub_Title = replace(Sub_Title, ")", "_");
	Sub_Title = replace(Sub_Title, "-", "_");
	Sub_Title = replace(Sub_Title, "+", "_");
	Sub_Title = replace(Sub_Title, " ", "_");
	//Sub_Title=Sub_Title+".tif";
	return Sub_Title;
}
