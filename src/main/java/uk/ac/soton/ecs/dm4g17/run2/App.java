package uk.ac.soton.ecs.dm4g17.run2;

import afu.org.checkerframework.checker.igj.qual.I;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.FloatArrayBackedDataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
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
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>("C:\\Users\\Test\\Desktop\\training", ImageUtilities.FIMAGE_READER);
        System.out.println("Training data file added to VFSDatasets");
        VFSListDataset<FImage> testingData = new VFSListDataset<>("C:\\Users\\Test\\Desktop\\testing", ImageUtilities.FIMAGE_READER);
        System.out.println("Testing data file added to VFSDatasets");

        //making a patch extractor
        ExtractPatches ep = new ExtractPatches(8,4);
        System.out.println("Patch extractor object made");
        Identifier idnt = new Identifier(trainingData,ep);
        System.out.println("Identifier object made");
        //making an assigner
        HardAssigner<float[], float[], IntFloatPair> assigner =
                idnt.trainQuantiser();

        System.out.println("Hard assigner object made");
        // construct an instance of our BoVWExtractor
        FeatureExtractor<SparseIntFV, FImage> extractor = new BoVWExtractor(ep, assigner);
        System.out.println("Instance of BoVWExtractor made");
        //construct and train a linear classifier
        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        System.out.println("LibLinear classifier constructed");
        ann.train(trainingData);
        System.out.println("LibLinear classifier trained");
        //Needs code for the mf comparison

    }

    //FeatureExtractor implementation with which we train the classifier
    static class BoVWExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        ExtractPatches ep;
        HardAssigner<float[], float[], IntFloatPair> assigner;
        LocalFeatureList<FloatKeypoint> keypointList;
        List<FImage> patchesOfImage;

        //Bag-of-Visual-Words extractor

        public BoVWExtractor(ExtractPatches ep, HardAssigner<float[], float[], IntFloatPair> assigner)
        {
            this.ep = ep;
            this.assigner = assigner;
            System.out.println("BoVWExtractor constructor made");
        }

        //Extracts features from each image and builds Bag-Of-Visual-Words around it
        public SparseIntFV extractFeature(FImage image) {
        System.out.println("extractFeatures() method called");
            SparseIntFV sparseIntFV;
            //Bag-Of-Visual-Words

            BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<>(assigner);
            System.out.println("Bag-of-Visual-Words made");
            this.keypointList = new MemoryLocalFeatureList<>();
            System.out.println("Keypoint list made");
            this.patchesOfImage = ep.getExtractedPatches(image);
            System.out.println("List of patches of images made");
            for (FImage patch : patchesOfImage)
            {
                //not sure if I got the right float vector of the image
                FloatKeypoint floatKeypoint =
                        new FloatKeypoint(0, 0, 0, 0, patch.getFloatPixelVector());
                keypointList.add(floatKeypoint);
            }

            sparseIntFV = bagOfVisualWords.aggregate(keypointList);
            System.out.println("extractFeature method returns SpareIntFv object");
            return sparseIntFV;
        }
    }
}

