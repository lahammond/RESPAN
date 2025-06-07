// Channels to Labels for Training
// Author: 	Luke Hammond
// Department of Neurology, The Ohio State University
// Date:	March 31, 2024

// Initialization
requires("1.53c");
run("Options...", "iterations=3 count=1 black edm=Overwrite");
run("Colors...", "foreground=white background=black selection=yellow");
run("Clear Results"); 
run("Close All");

// Parameters
#@ File rawpath(label="select folder containing imagesTr and labelsTr subfolders", style="directory")
#@ Integer(label="Maximum label intensity:", value = 6, style="spinner") MaxLabel
//#@ boolean(label="Export MIPs:", description=".") MIPon
//#@ boolean(label="Export Tif images :", description=".") TIFon

start = getTime();
//setBatchMode(true);

print("\\Clear");
print(rawpath)
getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);

channels_dir = rawpath +"/labels_as_channels/";
new_labels_dir = rawpath +"/labelsTr_"+year+"_"+month+"_"+dayOfMonth+"/";

filelist = getFileList(channels_dir);

File.makeDirectory(new_labels_dir);

for (i = 0; i < lengthOf(filelist); i++) {
    if (endsWith(filelist[i], ".tif")) { 
    		
	    print("\\Update1:  Processing file " + filelist[i] +". File " + (i+1) +" of " + filelist.length +".");

		// open image
		open(channels_dir + filelist[i]);
		rename("Channels");
		run("Split Channels");
		
		close("C1-Channels");
		
		if (MaxLabel >= 2) {
			imageCalculator("Subtract create stack", "C2-Channels","C3-Channels");
			close("C2-Channels");
			selectImage("Result of C2-Channels");
			rename("C2-Channels");
			run("Subtract...", "value=254 stack");
			
		}
		if (MaxLabel >= 3) {
			imageCalculator("Subtract create stack", "C3-Channels","C4-Channels");
			close("C3-Channels");
			selectImage("Result of C3-Channels");
			rename("C3-Channels");
			run("Subtract...", "value=253 stack");
		}
		if (MaxLabel >= 4) {
			imageCalculator("Subtract create stack", "C4-Channels","C5-Channels");
			close("C4-Channels");
			selectImage("Result of C4-Channels");
			rename("C4-Channels");
			run("Subtract...", "value=252 stack");
		}
		if (MaxLabel >= 5) {
			imageCalculator("Subtract create stack", "C5-Channels","C6-Channels");
			close("C5-Channels");
			selectImage("Result of C5-Channels");
			rename("C5-Channels");
			run("Subtract...", "value=251 stack");
		}
		
		if (MaxLabel >= 6) {
			imageCalculator("Subtract create stack", "C6-Channels","C7-Channels");
			close("C6-Channels");
			selectImage("Result of C6-Channels");
			rename("C6-Channels");
			run("Subtract...", "value=250 stack");
			selectImage("C7-Channels");
			run("Subtract...", "value=249 stack");
		}
	
		
		
		if (MaxLabel >= 2) {
			imageCalculator("Add create stack", "C2-Channels","C3-Channels");
			selectImage("Result of C2-Channels");
			rename("labels");
		}
		if (MaxLabel >= 3) {
			imageCalculator("Add create stack", "labels","C4-Channels");
			close("labels");
			selectImage("Result of labels");
			rename("labels");
		}
		if (MaxLabel >= 4) {
			imageCalculator("Add create stack", "labels","C5-Channels");
			close("labels");
			selectImage("Result of labels");
			rename("labels");
		}
		if (MaxLabel >= 5) {
			imageCalculator("Add create stack", "labels","C6-Channels");
			close("labels");
			selectImage("Result of labels");
			rename("labels");
		}
		if (MaxLabel >= 6) {
			imageCalculator("Add create stack", "labels","C7-Channels");
			close("labels");
			selectImage("Result of labels");
			rename("labels");
		}

		run("glasbey_on_dark");
		save(new_labels_dir + filelist[i]);
		
		
		close("*");
		run("Collect Garbage");
		
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
		if(endsWith(arr[i], ".tif") || endsWith(arr[i], ".nd2") || endsWith(arr[i], ".LSM") || endsWith(arr[i], ".czi") || endsWith(arr[i], ".jpg") ) {   //if it's a tiff image add it to the new array
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
	Sub_Title=Sub_Title+".tif";
	return Sub_Title;
}
