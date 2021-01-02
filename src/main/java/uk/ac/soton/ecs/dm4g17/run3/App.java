package uk.ac.soton.ecs.dm4g17.run3;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
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
import org.openimaj.util.pair.IntFloatPair;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
        setupClass("classpath:training.zip","classpath:testing.zip");


    }



    public static void setupClass(String trainingFileLocation, String testFileLocation) throws FileSystemException, FileNotFoundException {
        System.out.println("Run 3 Initiated...");

        //Retrieving files
        File trainingFile = ResourceUtils.getFile(trainingFileLocation);
        File testingFile  = ResourceUtils.getFile(testFileLocation);

        //Files are found
        System.out.println("File Found : " + trainingFile.exists());
        System.out.println("File Found : " + testingFile.exists());

        String test = trainingFile.getPath();
        String test2 = testingFile.getPath();

        //Adding files to VFSDatasets
        //Path version
        //VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>("C:\\Users\\Test\\Desktop\\11.12.20 Memory Pen Backup\\Uni Work\\Computer Vision Assignments\\CW3 - Group Project\\Run1\\src\\main\\resources\\training", ImageUtilities.FIMAGE_READER);
        //VFSListDataset<FImage> testingData   = new VFSListDataset<>("C:\\Users\\Test\\Desktop\\11.12.20 Memory Pen Backup\\Uni Work\\Computer Vision Assignments\\CW3 - Group Project\\Run1\\src\\main\\resources\\testing", ImageUtilities.FIMAGE_READER);
        //Specified Path Version
        //VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(test, ImageUtilities.FIMAGE_READER);
        //VFSListDataset<FImage> testingData   = new VFSListDataset<>(test2, ImageUtilities.FIMAGE_READER);
        //Url Version
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData    = new VFSListDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip", ImageUtilities.FIMAGE_READER);
        System.out.println("Size of training set: " + trainingData.size());
        System.out.println("Size of training set: " + testingData.size());

        //If we want a validation set we can use method below
        //Splits passed dataset into trainingData, validationData, testData
        //Training Dataset contains 15*100 = 1500 images
        double trainingDataSize   = 1500*0.8;
        double validationDataSize = 1500*0.2;
        double testingDataSize    = 2985;
        //GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String,FImage>(trainingData, 0, 0, 0);

        //Linear SVM Annotator
        //LinearSVMAnnotator<FImage, String> annotatorLinearSVM = new LinearSVMAnnotator<FImage, String>()

        DenseSIFT dsift = new DenseSIFT(5, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);

        //annotatorLinearSVM.train(trainingData);

        File fileDirectory = new File("C:\\Users\\Test\\Desktop\\11.12.20 Memory Pen Backup\\Uni Work\\Computer Vision Assignments\\CW3 - Group Project\\Run1\\src\\main\\cache");

        HardAssigner<byte[], float[], IntFloatPair> assigner;
        try {
            //Illegal reflective access operation has occurred - I think because I am trying to access a file here thats not already created?
            assigner = IOUtils.readFromFile(fileDirectory);
            System.out.println("Assigner read from: " +fileDirectory.toString());
        } catch (IOException e1) {
            System.out.println("Assigner not read from: " +fileDirectory.toString());
            assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingData, 30), pdsift);
            try {
                IOUtils.writeToFile(assigner,fileDirectory);
                System.out.println("Assigner saved to: " +fileDirectory.toString());
            } catch (IOException e2)
            {
                System.out.println("Assigner not saved to: " +fileDirectory.toString());
            }
        }


        FeatureExtractor<DoubleFV, FImage> PHOWExtractor = new PHOWExtractor(pdsift, assigner);
        LiblinearAnnotator<FImage, String> annotatorPHOW = new LiblinearAnnotator<FImage, String>(PHOWExtractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        long totalTimePHOWExtractor = trainTimeEstimator(annotatorPHOW, trainingData);
        ClassificationEvaluator<CMResult<String>, String, FImage> PHOWeval = new ClassificationEvaluator<CMResult<String>, String, FImage>(annotatorPHOW, trainingData, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        Map<FImage, ClassificationResult<String>> PHOWguesses = PHOWeval.evaluate();
        CMResult<String> PHOWresult = PHOWeval.analyse(PHOWguesses);
        System.out.println("PHOW" +PHOWresult);
        System.out.println("Training time for PHOWExtractor: " +totalTimePHOWExtractor +"s");

    }

    //Method to perform K-Means Clustering on a sample of SIFT features in order to build a HardAssigner that can assign features to identifiers
    //Takes as input a dataset and a PyramidDenseSIFT object
    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift)
    {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        for (FImage image : sample) {
            pdsift.analyseImage(image);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    //FeatureExtractor implementation with which we train the classifier
    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }

    public static void setupClassifier()
    {

        //Linear SVM Annotator
        LinearSVMAnnotator<FImage, String> annotatorLinearSVM;

    }

    public static void trainClassifier()
    {
        DenseSIFT dsift = new DenseSIFT(5, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);


    }

    public static void testClassifier()
    {

    }

    public static long trainTimeEstimator(LiblinearAnnotator<FImage, String> ann, VFSGroupDataset<FImage> trainingData)
    {
        long startTime = System.nanoTime();
        ann.train(trainingData);
        long endTime = System.nanoTime();
        return (endTime - startTime) / 100000000;
    }
}
