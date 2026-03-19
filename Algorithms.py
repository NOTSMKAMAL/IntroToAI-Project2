from nearestNeighbor import leave_one_out_accuracy


def format_feature_set(feature_set):
    # Turn a set like {3,1,5} into "{1,3,5}" for cleaner output
    if not feature_set:
        return "{}"
    return "{" + ",".join(str(f) for f in sorted(feature_set)) + "}"


def forward_selection(data):
    num_features = len(data[0]) - 1
    current_set = set()
    best_overall_set = set()
    best_overall_accuracy = 0.0

    print("Starting search.\n")

    # Build the feature set one feature at a time
    for _ in range(num_features):
        best_feature_this_round = None
        best_accuracy_this_round = 0.0

        # Try adding each feature that is not already selected
        for feature in range(1, num_features + 1):
            if feature in current_set:
                continue

            candidate_set = set(current_set)
            candidate_set.add(feature)

            # Evaluate this candidate subset using leave-one-out accuracy
            accuracy = leave_one_out_accuracy(data, candidate_set)

            print(
                f"Using feature(s) {format_feature_set(candidate_set)} "
                f"accuracy is {accuracy:.1f}%"
            )

            # Keep the best feature choice for this round
            if accuracy > best_accuracy_this_round:
                best_accuracy_this_round = accuracy
                best_feature_this_round = feature

        # After testing all candidates, permanently add the best one
        if best_feature_this_round is not None:
            current_set.add(best_feature_this_round)

            print(
                f"\nFeature set {format_feature_set(current_set)} was best, "
                f"accuracy is {best_accuracy_this_round:.1f}%\n"
            )

            # Track the best subset seen during the whole search
            if best_accuracy_this_round > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_this_round
                best_overall_set = set(current_set)

    return best_overall_set, best_overall_accuracy


def backward_elimination(data):
    num_features = len(data[0]) - 1
    # Start with all features selected
    current_set = set(range(1, num_features + 1))
    best_overall_set = set(current_set)
    best_overall_accuracy = leave_one_out_accuracy(data, current_set)

    print("Starting search.\n")

    # Remove one feature at a time until only one is left
    while len(current_set) > 1:
        best_feature_to_remove = None
        best_accuracy_this_round = 0.0

        # Try removing each feature currently in the set
        for feature in current_set.copy():
            candidate_set = set(current_set)
            candidate_set.remove(feature)

            # Evaluate the subset after removing that feature
            accuracy = leave_one_out_accuracy(data, candidate_set)

            print(
                f"Using feature(s) {format_feature_set(candidate_set)} "
                f"accuracy is {accuracy:.1f}%"
            )

            # Keep the removal that gives the highest accuracy this round
            if accuracy > best_accuracy_this_round:
                best_accuracy_this_round = accuracy
                best_feature_to_remove = feature

        # Permanently remove the best feature choice for this round
        if best_feature_to_remove is not None:
            current_set.remove(best_feature_to_remove)

            print(
                f"\nFeature set {format_feature_set(current_set)} was best, "
                f"accuracy is {best_accuracy_this_round:.1f}%\n"
            )

            # Track the best subset found during the full search
            if best_accuracy_this_round > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_this_round
                best_overall_set = set(current_set)

    return best_overall_set, best_overall_accuracy