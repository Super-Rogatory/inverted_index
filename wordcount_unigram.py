import os
import shutil
from pyspark import SparkContext
import re
# tokenizer_mapper = sanitize_text + merge_counts
def sanitize_text(line):
    # split into docID and content
    parts = line.split("\t", 1)
    if len(parts) < 2:
        return []

    docID, text = parts
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    words = text.split()

    return [(word, f"{docID}:1") for word in words]

def merge_counts(values):
    # combine docID:count strings, e.g., "doc1:1,doc2:1", using hashmap
    count_map = {}
    for val in values:
        docID, count = val.split(":")
        count_map[docID] = count_map.get(docID, 0) + int(count)
    return [f"{doc}:{cnt}" for doc, cnt in count_map.items()]

if __name__ == "__main__":
    sc = SparkContext("local", "UnigramIndex")

    lines = sc.textFile("fulldata/")  # input folder
    word_doc_pairs = lines.flatMap(sanitize_text)

    grouped = word_doc_pairs.groupByKey()
    result = grouped.mapValues(merge_counts).mapValues(lambda lst: ",".join(lst))

    result.saveAsTextFile("unigram_index") # output file
    # merge part files into a single output file
    output_dir = "unigram_index"
    output_file = "unigram_index.txt"

    with open(output_file, "w") as outfile:
        for fname in sorted(os.listdir(output_dir)):
            if fname.startswith("part-"):
                with open(os.path.join(output_dir, fname)) as f:
                    outfile.write(f.read())
    # cleanup
    sc.stop()
    # remove intermediate files
    shutil.rmtree("unigram_index")
