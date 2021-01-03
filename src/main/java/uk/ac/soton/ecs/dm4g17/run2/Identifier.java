package uk.ac.soton.ecs.dm4g17.run2;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
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

    private VFSGroupDataset<FImage> dataset;
    private ExtractPatches extractPatches;

    public Identifier(VFSGroupDataset<FImage> dataset, ExtractPatches extractPatches)
    {
        this.dataset = dataset;
        this.extractPatches = extractPatches;
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
        for(FImage img: dataset)
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
