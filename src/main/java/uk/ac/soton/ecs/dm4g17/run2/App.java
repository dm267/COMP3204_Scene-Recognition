package uk.ac.soton.ecs.dm4g17.run2;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileObject;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

public class App {
    public static void main( String[] args ) throws IOException {
        System.out.println("Run 2 Initiated...");

        /// Retrieving files
        File trainingFile = ResourceUtils.getFile("classpath:training");
        File testingFile = ResourceUtils.getFile("classpath:testing");

        // Files are found
        System.out.println("File Found : " + trainingFile.exists());
        System.out.println("File Found : " + testingFile.exists());

        // Adding files to VFSDatasets
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainingFile.getPath(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testingFile.getPath(), ImageUtilities.FIMAGE_READER);

        GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset = GroupSampler.sample(trainingData, 15, false);
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(groupedDataset, 80, 0, 20);

        //making a patch extractor
        ExtractPatches ep = new ExtractPatches(8,4);
        System.out.println("Patch extractor object made");
        Identifier idnt = new Identifier(splits,ep);
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
        ann.train(splits.getTrainingDataset());
        System.out.println("LibLinear classifier trained");
        //Needs code for the mf comparison

        System.out.println("Starting classification");
        ClassificationEvaluator<CMResult<String>, String, FImage> HKMeval = new ClassificationEvaluator<CMResult<String>, String, FImage>(
                ann, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        System.out.println("Starting evaluation");
        Map<FImage, ClassificationResult<String>> HKMguesses = HKMeval.evaluate();

        System.out.println("Analysing results");
        CMResult<String> HKMresult = HKMeval.analyse(HKMguesses);
        System.out.println(HKMresult.getDetailReport());

        // Creates a file named run2.txt where we write our predictions for each run
        File run2File = new File ("/COMP3204_CW3/run2.txt");
        PrintWriter txtPrinter = new PrintWriter("run2.txt");

        // Allows us to iterate through the testing dataset classes
        String result;
        FileObject[] files = testingData.getFileObjects();

        // Iterate through the training images and run our classifier on them
        // Print our results to the created run2.txt file
        for(int i=0; i < testingData.size(); i++) {
            result = files[i].getName().getBaseName()+" "+ann.classify(testingData.get(i)).getPredictedClasses().iterator().next();

            txtPrinter.println(result);
        }
        // Close the stream
        txtPrinter.close();

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

