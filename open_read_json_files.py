#code to open and read source files as json files
#created new json files and opened in notepad++

import json

splits = ["train", "valid", "test"]

for split in splits:
    source_file = f"{split}.wp_source"
    target_file = f"{split}.wp_target"
    output_file = f"{split}.json"

    data = []

    with open(source_file, "r", encoding="utf-8") as sf, \
         open(target_file, "r", encoding="utf-8") as tf:

        source_lines = sf.readlines()
        target_lines = tf.readlines()

        if len(source_lines) != len(target_lines):
            print(f"WARNING: {split} source/target lengths differ!")

        for src, tgt in zip(source_lines, target_lines):
            src = src.strip()
            tgt = tgt.strip()

            # ---- SAFE TAG REMOVAL ----
            if src.startswith("[") and "] " in src:
                src = src.split("] ", 1)[1]

            data.append({
                "prompt": src,
                "story": tgt
            })

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(data, out, indent=2, ensure_ascii=False)

    print(f"Created {output_file} with {len(data)} entries.")