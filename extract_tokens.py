# 已经提取到tokens.txt了

import argparse
import csv


def parse_symbols(truth):
    unique_symbols = set()
    i = 0
    while i < len(truth):
        char = truth[i]
        i += 1
        if char.isspace():
            continue
        elif char == "\\":
            if i < len(truth) and (truth[i] == "{" or truth[i] == "}"):
                unique_symbols.add(char + truth[i])
                i += 1
                continue
            escape_seq = char
            while i < len(truth) and truth[i].isalpha():
                escape_seq += truth[i]
                i += 1
            unique_symbols.add(escape_seq)
        else:
            unique_symbols.add(char)
    return unique_symbols


def create_tokens(groundtruth, output="tokens.txt"):
    with open(groundtruth, "r") as fd:
        unique_symbols = set()
        reader = csv.DictReader(fd)
        for row in reader:
            truth = row["formula"]  # 替换为实际的列名
            truth_symbols = parse_symbols(truth)
            unique_symbols = unique_symbols.union(truth_symbols)

        # unique_symbols.remove("\\ltN")
        symbols = list(unique_symbols)
        symbols.sort()
        with open(output, "w") as output_fd:
            writer = csv.writer(output_fd, delimiter="\t")
            writer.writerow(symbols)


if __name__ == "__main__":
    """
    extract_tokens path/to/groundtruth.csv [-o OUTPUT]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="tokens.txt",
        help="Output path of the tokens text file",
    )
    parser.add_argument("groundtruth", nargs=1, help="Ground truth CSV file")
    args = parser.parse_args()
    create_tokens(args.groundtruth[0], args.output)