package uk.ac.soton.ecs.dm4g17.run2;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;

public class App {
    public static void main( String[] args ) throws IOException {

        //Retrieving files
        File trainFile = ResourceUtils.getFile("classpath:training.zip");
        File testFile = ResourceUtils.getFile("classpath:testing.zip");

        //Files are found
        System.out.println("File Found : " + trainFile.exists());
        System.out.println("File Found : " + testFile.exists());

        //Adding files to VFSDatasets
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainFile.getPath(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testFile.getPath(), ImageUtilities.FIMAGE_READER);

    }
}
