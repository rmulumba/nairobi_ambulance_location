import train_kmeans
import predictions

"""
Training the ML model and making predictions.
"""

if __name__ == "__main__":
    train_kmeans.train()
    predictions.create_prediction_file()