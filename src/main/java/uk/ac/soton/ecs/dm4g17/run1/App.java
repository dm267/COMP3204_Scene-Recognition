package uk.ac.soton.ecs.dm4g17.run1;

import org.apache.commons.vfs2.FileObject;
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
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Map;


public class App {

    public static void main( String[] args ) throws IOException {

        App run1 = new App();
        run1.run();
    }

    App() {
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

        // Split the training & testing datasets into training, validation and testing subsets
        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String,FImage>(trainingData, 80, 0, 20);
        GroupedDataset<String, ListDataset<FImage>, FImage> testSet  = splitter.getTestDataset();

        // Creating an instance of the TinyImage class
        // Allows us to extract the centre of the images for resizing to 16x16
        FeatureExtractor<DoubleFV, FImage> tinyImage = new TinyImage();

        //Number of k neighbours , a parameter of KNNAnnotator
        int knnNumber = 15;

        // Performs the knn search using the chosen k number
        // Trains the KNNAnnotator on the datsets with the chosen subset splits
        KNNAnnotator<FImage, String, DoubleFV> knn = new KNNAnnotator<FImage, String, DoubleFV>(tinyImage, DoubleFVComparison.EUCLIDEAN, knnNumber);
        knn.train(splitter.getTrainingDataset());


        // Find and print the accuracy of our KNN classifier
        // Help from chapter 12 of OpenIMAJ tutorial
        ClassificationEvaluator<CMResult<String>, String, FImage> knnResult = new ClassificationEvaluator<CMResult<String>, String, FImage>(
                knn, testSet, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        System.out.println("Starting evaluation");
        Map<FImage, ClassificationResult<String>> knnGuesses = knnResult.evaluate();

        System.out.println("Analysing results");
        CMResult<String> knnScore = knnResult.analyse(knnGuesses);
        System.out.println(knnScore.getDetailReport());


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



