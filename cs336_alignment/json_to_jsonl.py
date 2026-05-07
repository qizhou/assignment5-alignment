import json

inputfile = "./data/math/sft.json"
outputfile = "./data/math/sft.jsonl"

with open(inputfile) as f:
    j = json.load(f)

with open(outputfile, "w") as f:
    for jl in j:
        json.dump(jl, f)
        f.write("\n")