import csv
import sys


def compare_with_counts_simple(file1, file2):
    """
    Simple comparison showing all mismatches with value counts.
    """
    # Read first file
    data1 = {}
    zeros1 = ones1 = 0
    with open(file1, 'r') as f1:
        reader = csv.reader(f1)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                id_val = row[0].strip()
                pred_val = row[1].strip()
                data1[id_val] = pred_val
                if pred_val == '0':
                    zeros1 += 1
                elif pred_val == '1':
                    ones1 += 1

    # Compare and count
    mismatches = []
    zeros2 = ones2 = 0
    matches = 0
    total_compared = 0

    print(f"\n{'ID':<20} | {'File 1':<8} | {'File 2':<8}")
    print("-" * 45)

    with open(file2, 'r') as f2:
        reader = csv.reader(f2)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                id_val = row[0].strip()
                pred_val = row[1].strip()

                # Count values in file2
                if pred_val == '0':
                    zeros2 += 1
                elif pred_val == '1':
                    ones2 += 1

                if id_val in data1:
                    total_compared += 1
                    if data1[id_val] == pred_val:
                        matches += 1
                    else:
                        print(f"{id_val:<20} | {data1[id_val]:<8} | {pred_val:<8}")
                        mismatches.append((id_val, data1[id_val], pred_val))

    # Calculate percentages
    total1 = zeros1 + ones1
    total2 = zeros2 + ones2
    match_percentage = (matches / total_compared * 100) if total_compared > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS WITH VALUE COUNTS")
    print("=" * 60)
    print(f"Total compared: {total_compared}")
    print(f"Matches: {matches} ({match_percentage:.2f}%)")
    print(f"Mismatches: {len(mismatches)}")

    print(f"\n🔢 VALUE COUNTS:")
    print(f"File 1: {zeros1} zeros ({zeros1 / total1 * 100:.1f}%), {ones1} ones ({ones1 / total1 * 100:.1f}%)")
    print(f"File 2: {zeros2} zeros ({zeros2 / total2 * 100:.1f}%), {ones2} ones ({ones2 / total2 * 100:.1f}%)")

    # Distribution comparison
    print(f"\n📊 DISTRIBUTION:")
    bar_length = 20

    zeros_bar1 = "█" * int((zeros1 / total1) * bar_length)
    ones_bar1 = "█" * int((ones1 / total1) * bar_length)
    zeros_bar2 = "█" * int((zeros2 / total2) * bar_length)
    ones_bar2 = "█" * int((ones2 / total2) * bar_length)

    print(f"File 1: 0s: [{zeros_bar1:<{bar_length}}] 1s: [{ones_bar1:<{bar_length}}]")
    print(f"File 2: 0s: [{zeros_bar2:<{bar_length}}] 1s: [{ones_bar2:<{bar_length}}]")

    return match_percentage


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_simple.py <file1.csv> <file2.csv>")
        sys.exit(1)

    compare_with_counts_simple(sys.argv[1], sys.argv[2])