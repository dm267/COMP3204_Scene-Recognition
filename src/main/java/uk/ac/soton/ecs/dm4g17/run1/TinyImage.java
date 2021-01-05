package uk.ac.soton.ecs.dm4g17.run1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class TinyImage implements FeatureExtractor<DoubleFV, FImage>{

    public DoubleFV extractFeature(FImage image) {

        // Size of tinyImage is 16*16, a 16 bit representation of original image
        int tinyImageSize = 16*16;
        double[] vector;

        // Crop the image to a square about the centre
        int startingX = (image.width);
        int startingY = (image.height);
        int smallestDimension = Math.min(startingX, startingY);

        // Extract the centre of the image  crop the image to a square about the centre
        image = image.extractCenter(smallestDimension, smallestDimension);
        // Resize the image to a 16x16 image
        image = ResizeProcessor.resample(image, 16, 16);



        // Find the mean
        float mean = image.sum()/tinyImageSize;

        // Make the tiny image have zero mean and unit length for a potential performance improvement (as per Spec)
        image = image.subtract(mean);
        image = image.normalise();

        double[] pixelVector = image.getDoublePixelVector();

        // Concatenates each row of pixel values into a single vector
        DoubleFV featureVector = new DoubleFV(pixelVector);
        return featureVector;
    }

}