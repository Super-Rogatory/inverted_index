import os
import shutil
from pyspark import SparkContext
import re

target_bigrams = {"computer science", "information retrieval", "power politics", "los angeles", "bruce willis"} # set of target bigrams

def sanitize_text(line):
    # split into docID and content
    parts = line.split("\t", 1)
    if len(parts) < 2:
        return []

    docID, text = parts
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    words = text.split()

    # create bigrams
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return [(f"{w1} {w2}", f"{docID}:1") for w1, w2 in bigrams if f"{w1} {w2}" in target_bigrams]

def merge_counts(values):
    count_map = {}
    for val in values:
        docID, count = val.split(":")
        count_map[docID] = count_map.get(docID, 0) + int(count)
    return [f"{doc}:{cnt}" for doc, cnt in count_map.items()]

if __name__ == "__main__":
    sc = SparkContext("local", "BigramIndex")

    lines = sc.textFile("devdata/")  # folder with input files
    bigram_doc_pairs = lines.flatMap(sanitize_text)
    # create pairs of (bigram, docID:1)
    grouped = bigram_doc_pairs.groupByKey()
    result = grouped.mapValues(merge_counts).mapValues(lambda lst: ",".join(sorted(lst)))

    # merge part files into a single output file
    output_dir = "selected_bigram_index"
    output_file = "selected_bigram_index.txt"

    result.saveAsTextFile("selected_bigram_index") # output file
    with open(output_file, "w") as outfile:
        for fname in sorted(os.listdir(output_dir)):
            if fname.startswith("part-"):
                with open(os.path.join(output_dir, fname)) as f:
                    outfile.write(f.read())
    # cleanup
    sc.stop()
    # remove intermediate files
    shutil.rmtree("selected_bigram_index")
