package uk.ac.soton.ecs.dm4g17.run2;

import org.openimaj.data.DataSource;
import org.openimaj.data.FloatArrayBackedDataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
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
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<>(trainFile.getPath(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>(testFile.getPath(), ImageUtilities.FIMAGE_READER);

    }



        /*
        perform K-Means clustering in order to build
        a HardAssigner that can assign features to identifiers.
        Note: Some of the code is reused from tutorial 12
         */
        private static HardAssigner<float[], float[], IntFloatPair> trainQuantiser
                (VFSGroupDataset<FImage> dataset, ExtractPatches extractPatches)
        {
            LocalFeatureList<FloatKeypoint> allKeys =
                     new MemoryLocalFeatureList<>();
            for(FImage img: dataset)
            {
                LocalFeatureList<FloatKeypoint> keypointList = new MemoryLocalFeatureList<>();
                List<FImage> patchesOfImage = extractPatches.getExtractedPatches(img);
                for (FImage patch : patchesOfImage)
                {
                    //not sure if I got the right float vector of the image
                    FloatKeypoint floatKeypoint =
                            new FloatKeypoint(0, 0, 0, 0, patch.getFloatPixelVector());
                    keypointList.add(floatKeypoint);
                }
                allKeys.addAll(keypointList);
            }
            if (allKeys.size() > 10000)
            {
                allKeys = allKeys.subList(0, 10000);
            }

            //clusters the features into 500 separate classes
            FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
            DataSource<float[]> datasource = new LocalFeatureListDataSource<>(allKeys);
            FloatCentroidsResult result = km.cluster(datasource);
            return result.defaultHardAssigner();
        }
    }

