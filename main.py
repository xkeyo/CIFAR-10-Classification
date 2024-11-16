from load_data import get_test_features_pca, get_train_features_pca

def main():
    return 0

if __name__ == "__main__":

    train_features_pca = get_train_features_pca()
    test_features_pca = get_test_features_pca()

    # Print the shape of the transformed features
    # Should be (5000, 50) and (1000, 50) because we have 
    # 10 classes with 500 training images each and 10 classes with 100 testing images each
    print(train_features_pca.shape)
    print(test_features_pca.shape)

    exit(0)
