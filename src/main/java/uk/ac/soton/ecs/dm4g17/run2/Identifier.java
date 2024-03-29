package uk.ac.soton.ecs.dm4g17.run2;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.List;

public class Identifier {

    private GroupedRandomSplitter<String, FImage> splits;
    private ExtractPatches extractPatches;

    public Identifier(GroupedRandomSplitter<String, FImage> splits, ExtractPatches extractPatches)
    {
        this.splits = splits;
        this.extractPatches = extractPatches;
        System.out.println("Identifier constructor made");
    }
    /*
        perform K-Means clustering in order to build
        a HardAssigner that can assign features to identifiers.
        Note: Some of the code is reused from tutorial 12
         */
    public HardAssigner<float[], float[], IntFloatPair> trainQuantiser()
    {
        LocalFeatureList<FloatKeypoint> allKeys =
                new MemoryLocalFeatureList<>();
        //Iterates through the sample dataset and extracting the patches and adding them
        for(FImage img: splits.getTrainingDataset())
        {
            LocalFeatureList<FloatKeypoint> keypointList;
            List<FImage> patchesOfImage;

            keypointList = new MemoryLocalFeatureList<>();
            patchesOfImage = extractPatches.getExtractedPatches(img);

            for (FImage patch : patchesOfImage)
            {
                //not sure if I got the right float vector of the image
                FloatKeypoint floatKeypoint =
                        new FloatKeypoint(0, 0, 0, 0, patch.getFloatPixelVector());
                keypointList.add(floatKeypoint);
            }
            allKeys.addAll(keypointList);
        }
        System.out.println("Features extracted");
        //Extracts the first 10000 keypoints
        if (allKeys.size() > 10000)
        {
            allKeys = allKeys.subList(0, 10000);
        }

        //clusters the features into 500 separate classes
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
        System.out.println("Features clustered into 500 separate classes");
        DataSource<float[]> datasource = new LocalFeatureListDataSource<>(allKeys);
        FloatCentroidsResult result = km.cluster(datasource);
        System.out.println("Returning HardAssigner");
        return result.defaultHardAssigner();
    }
}
