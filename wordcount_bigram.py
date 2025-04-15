import os
from pyspark import SparkContext
import re

target_bigrams = {"computer science", "information retrieval", "power politics", "los angeles", "bruce willis"} # set of target bigrams

def extract_bigrams(line):
    # split into docID and content
    parts = line.split("\t", 1)
    if len(parts) < 2:
        return []

    docID, text = parts
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    words = text.split()

    # create bigrams
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return [(f"{w1} {w2}", docID) for w1, w2 in bigrams if f"{w1} {w2}" in target_bigrams]

if __name__ == "__main__":
    sc = SparkContext("local", "BigramIndex")

    lines = sc.textFile("devdata/")  # folder with input files
    bigram_doc_pairs = lines.flatMap(extract_bigrams)

    # group by bigram and deduplicate docIDs
    grouped = bigram_doc_pairs.distinct().groupByKey()
    result = grouped.mapValues(lambda docIDs: ",".join(sorted(set(docIDs))))

    result.saveAsTextFile("selected_bigram_index") # output file
    # merge part files into a single output file
    output_dir = "selected_bigram_index"
    output_file = "selected_bigram_index.txt"

    with open(output_file, "w") as outfile:
        for fname in sorted(os.listdir(output_dir)):
            if fname.startswith("part-"):
                with open(os.path.join(output_dir, fname)) as f:
                    outfile.write(f.read())
