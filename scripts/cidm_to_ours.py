"""Ich results/<method>/<replace_prompt>/{samples,prompts.json} -> nasz after_task{k}/task{j}_{cid}/."""
import json, os, shutil, sys
src_root, dst_root, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
ORDER = ["cifc_dog","cifc_duck_toy","cifc_cat","cifc_backpack","cifc_teddybear",
         "cifc_painting","cifc_dog2","cifc_drawing","cifc_cat2","cifc_ink_painting"]
maps = json.load(open("datasets/data_cfgs/task10.json"))
done = 0
for j, (cid, m) in enumerate(zip(ORDER, maps)):
    s = os.path.join(src_root, m["replace_mapping"])
    if not os.path.isdir(s):
        continue
    d = os.path.join(dst_root, f"after_task{k:02d}", f"task{j:02d}_{cid}")
    os.makedirs(d, exist_ok=True)
    if os.path.isdir(os.path.join(s, "samples")):
        shutil.copytree(os.path.join(s, "samples"), os.path.join(d, "samples"), dirs_exist_ok=True)
    shutil.copy(os.path.join(s, "prompts.json"), os.path.join(d, "prompts.json"))
    done += 1
print(f"przeniesiono {done} konceptow -> after_task{k:02d}")
