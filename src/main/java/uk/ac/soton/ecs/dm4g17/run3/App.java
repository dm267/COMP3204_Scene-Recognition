package uk.ac.soton.ecs.dm4g17.run3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.FileNotFoundException;

    /*
    Run #3: You should try to develop the best classifier you can!
        You can choose whatever feature, encoding and classifier you like.

    Potential features: the GIST feature; Dense SIFT; Dense SIFT in a Gaussian Pyramid;
        Dense SIFT with spatial pooling (i.e. PHOW as in the OpenIMAJ tutorial),
        etc. Potential classifiers: Naive bayes; non-linear SVM (perhaps using a linear classifier with a Homogeneous Kernel Map),

     … Note: you don’t have to use OpenIMAJ for this run if you don’t want to
        (for example you might want to try and use a deep learning framework instead).
    */



public class App {

    public static void main( String[] args ) throws FileNotFoundException, FileSystemException {


        System.out.println("Run 3 Initiated...");

        //Retrieving files
        File trainingFile = ResourceUtils.getFile("classpath:training.zip");
        File testingFile = ResourceUtils.getFile("classpath:testing.zip");

        //Files are found
        System.out.println("File Found : " + trainingFile.exists());
        System.out.println("File Found : " + testingFile.exists());

        //Adding files to VFSDatasets
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingFile.getPath(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingFile.getPath(), ImageUtilities.FIMAGE_READER);


    }
}
