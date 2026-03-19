import math


def euclidean_distance(row1, row2, feature_set):
    # Compute distance using only the selected features
    distance_sum = 0.0

    for feature in feature_set:
        diff = row1[feature] - row2[feature]
        distance_sum += diff ** 2

    return math.sqrt(distance_sum)


def nearest_neighbor_predict(data, test_index, feature_set):
    # Find the closest training instance to the test instance
    best_distance = float("inf")
    predicted_label = None
    test_row = data[test_index]

    for i in range(len(data)):
        if i == test_index:
            continue  # skip comparing the row to itself

        train_row = data[i]
        dist = euclidean_distance(test_row, train_row, feature_set)

        if dist < best_distance:
            best_distance = dist
            predicted_label = train_row[0]  # use the nearest neighbor's class

    return predicted_label


def leave_one_out_accuracy(data, feature_set):
    # Test each instance once while using all other instances as training data
    correct = 0

    for i in range(len(data)):
        predicted_label = nearest_neighbor_predict(data, i, feature_set)
        actual_label = data[i][0]

        if predicted_label == actual_label:
            correct += 1

    # Return the percentage of correctly classified instances
    return (correct / len(data)) * 100.0