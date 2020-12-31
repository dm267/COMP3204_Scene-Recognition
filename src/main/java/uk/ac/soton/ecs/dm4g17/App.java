package uk.ac.soton.ecs.dm4g17;

import org.apache.commons.vfs2.FileSystemException;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class App {

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

        System.out.println("Run 1 Initiated...");

        //Retrieving files
        File trainingFile = ResourceUtils.getFile("classpath:training.zip");
        File testingFile = ResourceUtils.getFile("classpath:testing.zip");

        //Files are found
        System.out.println("File Found : " + trainingFile.exists());
        System.out.println("File Found : " + testingFile.exists());

        //Adding files to VFSDatasets
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingFile.getPath(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingFile.getPath(), ImageUtilities.FIMAGE_READER);
		
        //Not sure what parameters to pass this so Ijust put 0,0,0 for now
        //GroupedRandomSplitter<String, FImage> spliter = new GroupedRandomSplitter<String,FImage>(trainingData, 0, 0, 0);


		// Was thinking of creating an instance of a new TinyImage class but maybe the method at the bottom is calm?
        //FeatureExtractor<FloatFV, FImage> tinyImage = new TinyImage();

        //Number of different scene classes to decide from
        int knnNumber = 15;
        //Size of tinyImage is 16*16, a 16 bit representation of original image
        int tinyImageSize = 16;
        
        //KNNAnnotator<FImage, String, FloatFV> knn = new KNNAnnotator<FImage, String, FloatFV>(tinyImage, FloatFVComparison.EUCLIDEAN, knnNumber);
		//knn.train(trainingData);

        FImage oldImage = null;
        FImage newImage = null;
        newImage = createTinyImage(oldImage, tinyImageSize);







    }

    //Auxiliary function to convert image to tinyImage version for further processing
    public static FImage createTinyImage(FImage image, int newSize) {
        int startingX = (image.width);
        int startingY = (image.height);
        int smallestDimension = Math.min(startingX, startingY);

        //This will extract a square version of the image that extends width/2 and height/2 from the centre point
        //Uses smallest dimension of image as we are cropping the image and cant add any information to enlarge it
        FImage squareVersionOfImage = image.extractCenter(smallestDimension,smallestDimension);

        //Resizes to the passed required size
        return squareVersionOfImage.process(new ResizeProcessor(newSize, newSize));
    }
}
