import numpy as np

from nearestNeighbor import leave_one_out_accuracy
from Algorithms import forward_selection, backward_elimination


def load_data(filename: str):
    return np.loadtxt(filename)


def print_dataset_info(data) -> None:
    num_instances, num_columns = data.shape
    num_features = num_columns - 1

    print(
        f"This dataset has {num_features} features "
        f" with {num_instances} instances."
    )


def format_feature_set(feature_set: set[int]) -> str:
    if not feature_set:
        return "{}"
    return "{" + ",".join(str(f) for f in sorted(feature_set)) + "}"


def main():
    print("Welcome to SM's Feature Selection Algorithm.\n")

    filename = input("Type in the name of the file to test: ").strip()
    data = load_data(filename)

    print()
    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    choice = input().strip()

    print()
    print_dataset_info(data)

    all_features = set(range(1, data.shape[1]))
    all_features_accuracy = leave_one_out_accuracy(data, all_features)

    print()
    print(
        f"Running nearest neighbor with all {len(all_features)} features, "
        f'using "leaving-one-out" evaluation, I get an accuracy of '
        f"{all_features_accuracy:.1f}%\n"
    )

    if choice == "1":
        best_set, best_accuracy = forward_selection(data)
    elif choice == "2":
        best_set, best_accuracy = backward_elimination(data)
    else:
        print("Invalid choice.")
        return

    print(
        f"Finished search. The best feature subset is "
        f"{format_feature_set(best_set)}, which has an accuracy of "
        f"{best_accuracy:.1f}%"
    )


if __name__ == "__main__":
    main()