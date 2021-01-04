package uk.ac.soton.ecs.dm4g17.run3;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LinearSVMAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;
import org.springframework.util.ResourceUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

    /*
    Run #3: You should try to develop the best classifier you can!
        You can choose whatever feature, encoding and classifier you like.

    Potential features: the GIST feature; Dense SIFT; Dense SIFT in a Gaussian Pyramid;
        Dense SIFT with spatial pooling (i.e. PHOW as in the OpenIMAJ tutorial),
        etc. Potential classifiers: Naive bayes; non-linear SVM (perhaps using a linear classifier with a Homogeneous Kernel Map),

     … Note: you don’t have to use OpenIMAJ for this run if you don’t want to
        (for example you might want to try and use a deep learning framework instead).
    */

    //Features:
    //      = Dense SIFT;
    //      = Dense SIFT in a Gaussian Pyramid;
    //      = Dense SIFT with spatial pooling (i.e. PHOW as in the OpenIMAJ tutorial)
    //With classifier:
    //      = Linear SVM with a Homogeneous Kernel Map


public class App {

    public static void main( String[] args ) throws FileNotFoundException, FileSystemException {
        setupClass();
    }



    public static void setupClass() throws FileSystemException {
        System.out.println("Run 3 Initiated...");

        //Adding files to VFSDatasets from their respective URL.
        System.out.println("Downloading datasets..");
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<FImage>("C:\\Users\\Test\\Desktop\\training", ImageUtilities.FIMAGE_READER);
        //VFSGroupDataset includes each instance with a training label and also its expected name therefore we need to remove all training instances
        trainingData.remove("training");

        VFSListDataset<FImage> testingData   = new VFSListDataset<FImage>("C:\\Users\\Test\\Desktop\\testing", ImageUtilities.FIMAGE_READER);
        System.out.println("Datasets downloaded.");

        System.out.println("Size of training class: " + trainingData.numInstances());
        System.out.println("Size of testing class: " + testingData.numInstances());

        GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset = GroupSampler.sample(trainingData, 15, false);
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(groupedDataset, 80, 0, 20);

        GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet     = splits.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> testSet         = splits.getTestDataset();

        //Try step=4 & binsize=8
        DenseSIFT denseSIFT = new DenseSIFT(5, 7);
        PyramidDenseSIFT<FImage> pyramidDenseSIFT = new PyramidDenseSIFT<FImage>(denseSIFT, 6f, 2,4,6,8);

        File fileDirectory = new File("C:\\CW3 - Group Project\\Run1\\src\\main\\cache");

        //Creating/Accessing an already created assigner that can assign features to identifiers.
        HardAssigner<byte[], float[], IntFloatPair> hardAssigner;
        try {
            //Illegal reflective access operation has occurred - This occurs when trying to access a file which is not already created.
            hardAssigner = IOUtils.readFromFile(fileDirectory);
            System.out.println("Assigner read from: " +fileDirectory.toString());
        } catch (IOException e1) {
            System.out.println("Assigner not read from: " +fileDirectory.toString());
            hardAssigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingSet, 30), pyramidDenseSIFT);
            try {
                IOUtils.writeToFile(hardAssigner,fileDirectory);
                System.out.println("Assigner saved to: " +fileDirectory.toString());
            } catch (IOException e2) {
                System.out.println("Assigner not saved to: " +fileDirectory.toString());
            }
        }

        //Pyramid Histogram Of Words - Currently not working
        FeatureExtractor<DoubleFV, FImage> PHOWExtractor = new PHOWExtractor(pyramidDenseSIFT, hardAssigner);
        //LiblinearAnnotator<FImage, String> annotatorPHOW = new LiblinearAnnotator<FImage, String>(PHOWExtractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        //long totalTimePHOWExtractor = trainTimeEstimator(annotatorPHOW, trainingData);
        //ClassificationEvaluator<CMResult<String>, String, FImage> PHOWeval = new ClassificationEvaluator<CMResult<String>, String, FImage>(annotatorPHOW, trainingData, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        //Map<FImage, ClassificationResult<String>> PHOWguesses = PHOWeval.evaluate();
        //CMResult<String> PHOWresult = PHOWeval.analyse(PHOWguesses);

        //Homogeneous Kernel Map
        //A Homogeneous Kernel Map transforms data into a compact linear representation such that applying a linear classifier approximates,
        // to a high degree of accuracy, the application of a non-linear classifier over the original data.
        HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        FeatureExtractor<DoubleFV, FImage> HKMExtractor = homogeneousKernelMap.createWrappedExtractor(PHOWExtractor);
        LiblinearAnnotator<FImage, String> annotatorHKM = new LiblinearAnnotator<FImage, String>(HKMExtractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        long totalTimeHKMExtractor = trainTimeEstimator(annotatorHKM, splits);
        System.out.println("Training time for HKMExtractor: " +totalTimeHKMExtractor +"s");

        System.out.println("Starting classification");
        ClassificationEvaluator<CMResult<String>, String, FImage> HKMeval = new ClassificationEvaluator<CMResult<String>, String, FImage>(annotatorHKM, testSet, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        System.out.println("Starting evaluation");
        Map<FImage, ClassificationResult<String>> HKMguesses = HKMeval.evaluate();

        System.out.println("Analysing results");
        CMResult<String> HKMresult = HKMeval.analyse(HKMguesses);

        //System.out.println("PHOW" +PHOWresult);
        System.out.println("HKM" +HKMresult);
        System.out.println(HKMresult.getDetailReport());

        //System.out.println("Training time for PHOWExtractor: " +totalTimePHOWExtractor +"s");


        testClassifier(annotatorHKM,splits,testingData);
    }

    //Performs K-Means Clustering on a sample of SIFT features in order to build a HardAssigner that can assign features to identifiers.
    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pyramidDenseSIFT)
    {
        List<LocalFeatureList<ByteDSIFTKeypoint>> featureLists = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        //Iterates through the sample dataset, analysing and adding the dense SIFT features to an arraylist
        for (FImage image : sample) {
            pyramidDenseSIFT.analyseImage(image);
            featureLists.add(pyramidDenseSIFT.getByteKeypoints(0.005f));
        }

        //Extracts the first 10000 dense SIFT features from the provided image dataset/
        if (featureLists.size() > 10000)
            featureLists = featureLists.subList(0, 10000);

        //Clusters the features into 500 separate classes/
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(500);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(featureLists);
        ByteCentroidsResult result = km.cluster(datasource);

        //Returns a HardAssigner which is then used to assign SIFT features to identifiers.
        return result.defaultHardAssigner();
    }

    //FeatureExtractor implementation with which we train the classifier
    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pyramidDenseSIFT;
        HardAssigner<byte[], float[], IntFloatPair> hardAssigner;

        //Pyramid Histogram Of Words Extractor
        public PHOWExtractor(PyramidDenseSIFT<FImage> pyramidDenseSIFT, HardAssigner<byte[], float[], IntFloatPair> hardAssigner)
        {
            this.pyramidDenseSIFT = pyramidDenseSIFT;
            this.hardAssigner = hardAssigner;
        }

        //Extracts features from each image and builds Bag-Of-Visual-Words around it
        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pyramidDenseSIFT.analyseImage(image);

            //Bag-Of-Visual-Words //Change Y-Block to 4 and re-test performance
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(hardAssigner);
            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);
            return spatial.aggregate(pyramidDenseSIFT.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }

    //Times how long it takes to train passed classifier
    public static long trainTimeEstimator(LiblinearAnnotator<FImage, String> annotator, GroupedRandomSplitter<String, FImage> splits)
    {
        long startTime = System.nanoTime();
        System.out.println("Training Model for Labelled Test Set...");
        annotator.train(splits.getTrainingDataset());
        System.out.println("Model Trained Successfully.");
        long endTime = System.nanoTime();
        return (endTime - startTime) / 100000000;
    }

    //Tests classifier on test dataset
    public static void testClassifier(LiblinearAnnotator<FImage, String> annotator, GroupedRandomSplitter<String, FImage> splits, VFSListDataset<FImage> unlabeledTestSet)
    {

        System.out.println("Training Model for Unlabeled Test Set...");
        annotator.train(splits.getTrainingDataset());
        System.out.println("Model Trained Successfully.");

        int sizeOfTestSet = unlabeledTestSet.numInstances();
        System.out.println("Cannot Write Run3 Results to File!");

        //Produce output of tests into a .txt file
        String nameOfResultsFile = "run3.txt";

        //Strings
        String classification;
        String fileName;

        //File to write
        File outputFile = new File(nameOfResultsFile);

        try {
            System.out.println("Writing Run3 Classification Results to " +nameOfResultsFile +" located at " +outputFile);
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
            for (int i = 0; i < sizeOfTestSet; i++)
            {
                FImage testImage = unlabeledTestSet.getInstance(i);
                fileName = unlabeledTestSet.getID(i);
                classification = String.valueOf(annotator.classify(testImage));
                writer.write(fileName +" " +classification +"\n");
            }
            writer.close();
            System.out.println("Successfully written Run3 Classification Results to " +nameOfResultsFile +" located at " +outputFile.getPath());
        } catch (IOException e) {
            System.out.println("Cannot Write Run3 Results to File!");
        }
    }

}