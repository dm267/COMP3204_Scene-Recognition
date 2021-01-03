package uk.ac.soton.ecs.dm4g17.run2;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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

    private static class ExtractPatches {
        // size of each patch
        int patchSize;
        // space/distance between each patch
        int patchDistance;

        //constructor of the ExtractPatches class
        public ExtractPatches (int patchSize, int patchDistance)
        {
            this.patchSize = patchSize;
            this.patchDistance = patchDistance;
        }

        /*
        Split FImage into patches.
        returns a List of normalised patches from the input image.
         */
        public List<FImage> getExtractedPatches(FImage image)
        {
            List<FImage> patches = new ArrayList<>();
            //height of the image
            int imageHeight = image.getHeight();
            //width of the image
            int imageWidth = image.getWidth();

            for (int row = 0; row < imageHeight; row += patchDistance)
            {
                for (int col = 0; col < imageWidth; col += patchDistance)
                {
                    FImage patch = image.extractROI(row, col, patchSize, patchSize);
                    //normalise
                    patch = patch.normalise();
                    patches.add(patch);
                }
            }
            return patches;
        }
    }
}
