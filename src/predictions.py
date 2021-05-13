import pandas as pd
import numpy as np
import config
import utils
import os

# Load cluster dataframe.

def create_prediction_file():
    """
    Creating and saving the the predictions file.
    """
    submission_file_name = utils.next_output_file_name(config.OUTPUT_PATH)

    # importing the dataframe made after kmeans clustering.
    # importing the test file.

    cluster_df = utils.load_dataset(config.INPUT_PATH + config.CLUSTER_DF)
    test = utils.load_dataset(config.INPUT_PATH + config.UPDATED_TEST_SET)

    submission_df = pd.merge(
        cluster_df,
        test,
        how='left',
        left_on=['Day', 'time_bin'],
        right_on=['day', 'time_bin']
    )

    # creating the final submission file.

    submission_df = submission_df.drop(
        columns=['time_bin', 'day', 'Day']
    )

    submission_df.to_csv(config.OUTPUT_PATH + submission_file_name)
    
    # Verifying that the submission file was created successfully.
    
    if os.path.exists(config.OUTPUT_PATH + submission_file_name):
        print(f"File: {submission_file_name} created successfully.")
    else:
        print("Error creating submission file.")