import tensorflow_datasets as tfds
import pandas as pd

def download_and_save_ag_news():
    """
    Downloads the AG News dataset and saves it as train.csv and test.csv.
    """
    # Download the dataset
    ds, info = tfds.load('ag_news_subset', with_info=True, as_supervised=True)
    train_ds, test_ds = ds['train'], ds['test']

    # Convert to pandas DataFrame
    def to_dataframe(dataset):
        texts, labels = [], []
        for text, label in tfds.as_numpy(dataset):
            texts.append(text.decode('utf-8'))
            labels.append(label)
        return pd.DataFrame({'text': texts, 'label': labels})

    train_df = to_dataframe(train_ds)
    test_df = to_dataframe(test_ds)

    # Save to CSV
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print("AG News dataset downloaded and saved to data/train.csv and data/test.csv")
    print("\nTrain data sample:")
    print(train_df.head())
    print(f"\nTrain data shape: {train_df.shape}")
    print(f"\nTest data shape: {test_df.shape}")


if __name__ == '__main__':
    download_and_save_ag_news()
