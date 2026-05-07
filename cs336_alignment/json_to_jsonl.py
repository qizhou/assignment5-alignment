import json

inputfile = "./data/math/val.json"
outputfile = "./data/math/val.jsonl"

with open(inputfile) as f:
    j = json.load(f)

with open(outputfile, "w") as f:
    for jl in j:
        json.dump(jl, f)
        f.write("\n")