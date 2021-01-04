package uk.ac.soton.ecs.dm4g17.run1;

import org.apache.commons.vfs2.FileObject;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.springframework.util.ResourceUtils;

import afu.org.checkerframework.checker.units.qual.K;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;


public class Run1 {

    /*
    Run #1: You should develop a simple k-nearest-neighbour classifier using the “tiny image” feature.

    The “tiny image” feature is one of the simplest possible image representations.
    One simply crops each image to a square about the centre,
      and then resizes it to a small, fixed resolution (we recommend 16x16).
    The pixel values can be packed into a vector by concatenating each image row.
    It tends to work slightly better if the tiny image is made to have zero mean and unit length.
    You can choose the optimal k-value for the classifier.
    */

    public static void main( String[] args ) throws IOException {
    	
    	Run1 run1 = new Run1();
    	run1.run();
    }
        
    Run1() {
    }
    
    public void run() throws IOException {
    System.out.println("Run 1 Initiated...");

        // Retrieving files
        File trainingFile = ResourceUtils.getFile("classpath:training");
        File testingFile = ResourceUtils.getFile("classpath:testing");

        // Files are found
        System.out.println("File Found : " + trainingFile.exists());
        System.out.println("File Found : " + testingFile.exists());

        // Adding files to VFSDatasets
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingFile.getPath(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingFile.getPath(), ImageUtilities.FIMAGE_READER);
        
        // Used to ensure that there are 1500 images in the training set & 2985 testing images
        //System.out.println(trainingData.numInstances());
        //System.out.println(testingData.numInstances());

        // Needed otherwise training set appears to be only 15 due to scene sub-folders
//       ArrayList<FImage> trainingImages = new ArrayList<FImage>();
//        ArrayList<String> trainingScenes = new ArrayList<String>();
        
        // Loop through images/scenes within sub directories and add them to an ArrayList
//       for(String scenes : trainingData.getGroups()) {
//        	
//        	    trainingScenes.add(scenes);
//
//       // Adds every image found in each sub directory within the training folder to an ArrayList
//       	    for(FImage image : trainingData.getInstances(scenes)) {
//        		trainingImages.add(image);
//       	}
//        }
         
        //System.out.println(trainingData.getGroups().size());
        //System.out.println(testingData.size());
        //System.out.println(trainingImages.size());
        //System.out.println(trainingScenes.size());

		
        // Split the training & testing datasets into training, validation and testing subsets 
        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String,FImage>(trainingData, 10, 0, 10);
        
        // Creating an instance of the TinyImage class
        // Allows us to extract the centre of the images for resizing to 16x16
        FeatureExtractor<DoubleFV, FImage> tinyImage = new TinyImage();

        //Number of k neighbours , a parameter of KNNAnnotator
        int knnNumber = 15;
        
        // Performs the knn search using the chosen k number
        // Trains the KNNAnnotator on the datsets with the chosen subset splits
        KNNAnnotator<FImage, String, DoubleFV> knn = new KNNAnnotator<FImage, String, DoubleFV>(tinyImage, DoubleFVComparison.EUCLIDEAN, knnNumber);
        knn.train(splitter.getTrainingDataset());
        
               
        // Creates a file named run1.txt where we write our predictions for each run
        File run1File = new File ("/COMP3204_CW3/run1.txt");
        PrintWriter txtPrinter = new PrintWriter("run1.txt");

        // Allows us to iterate through the testing dataset classes
        String result;
        FileObject[] files = testingData.getFileObjects();

        // Iterate through the training images and run our classifier on them
        // Print our results to the created run1.txt file
        for(int i=0; i < testingData.size(); i++) {
        	result = files[i].getName().getBaseName()+" "+knn.classify(testingData.get(i)).getPredictedClasses().iterator().next();
        
			txtPrinter.println(result);
        }
        // Close the stream
        txtPrinter.close();
       }   
    
}
    
    
    
