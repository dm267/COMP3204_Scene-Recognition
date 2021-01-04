package uk.ac.soton.ecs.dm4g17.run2;

import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;

import java.util.ArrayList;
import java.util.List;

public class ExtractPatches {
    // size of each patch
    int patchSize;
    // space/distance between each patch
    int patchDistance;

    //constructor of the ExtractPatches class
    public ExtractPatches(int patchSize, int patchDistance) {
        this.patchSize = patchSize;
        this.patchDistance = patchDistance;
    }

    /*
    Split FImage into patches.
    returns a List of normalised patches from the input image.
     */
    public List<FImage> getExtractedPatches(FImage image) {
        List<FImage> patches = new ArrayList<>();
        //height of the image
        int imageHeight = image.getHeight();
        //width of the image
        int imageWidth = image.getWidth();

        for (int row = 0; row < imageHeight; row += patchDistance) {
            for (int col = 0; col < imageWidth; col += patchDistance) {
                FImage patch = image.extractROI(row, col, patchSize, patchSize);
                //normalise
                patch = patch.normalise();
                patches.add(patch);
            }
        }
        return patches;
    }

}
