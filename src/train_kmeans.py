import pandas as pd 
import numpy as np
import utils
import config
from sklearn.cluster import KMeans 

def train():
    """
    Clustering train set using Kmeans clustering and storing the train file
    with the newly created cluster centers.
    """
    
    cluster_data = []
    day_names = config.DAY_NAMES
    bins = config.BIN_LABELS

    # Load the train and test datasets
    data = utils.load_dataset(config.TRAIN_PATH, ['datetime'])
    test = utils.load_dataset(config.TEST_PATH, ['date'])

    # Update the datasets with "time_bins" and "day" columns
    data = utils.create_bins_and_day_column(data, 'data')
    test = utils.create_bins_and_day_column(test, 'test')

    for idx, day in enumerate(day_names):
        for idy, time_bin in enumerate(bins):
            data_selected = data[
                (data['day'] == day_names[idx]) &
                (data['time_bin'] == bins[idy]) &
                (data['latitude'] >= -2.0 ) & 
                (data['longitude'] <= 37.4)
            ]
            
            kmeans = KMeans(
                n_clusters=config.KMEANS_N_CLUSTERS, 
                init=config.KMEANS_INIT,
                max_iter=config.KMEANS_MAX_ITER,
                random_state=config.RANDOM_STATE,
                algorithm=config.KMEANS_ALGORITHM

            )
            kmeans.fit(data_selected[['latitude', 'longitude']])
            cluster_data.append([day, time_bin, *(np.concatenate(kmeans.cluster_centers_).flatten())])
    
    # Creating the file with cluster centers and saving the file.

    cluster_df = pd.DataFrame(cluster_data)
    cluster_df.columns = ['Day', 'time_bin',
                        'A0_Latitude', 'A0_Longitude', 
                        'A1_Latitude', 'A1_Longitude', 
                        'A2_Latitude', 'A2_Longitude', 
                        'A3_Latitude', 'A3_Longitude', 
                        'A4_Latitude', 'A4_Longitude', 
                        'A5_Latitude', 'A5_Longitude' ]

    cluster_df.to_csv(config.INPUT_PATH + config.CLUSTER_DF, index=False)
    test.to_csv(config.INPUT_PATH + config.UPDATED_TEST_SET, index=False)

    print("Cluster_df.csv and test.csv created successfully.")