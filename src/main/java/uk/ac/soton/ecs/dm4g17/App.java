package uk.ac.soton.ecs.dm4g17;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class App {
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



    }
}
