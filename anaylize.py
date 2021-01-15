import matplotlib.pyplot as plt
import numpy as np

label_names = ["Oval_Face", "Attractive", "Bald", "Eyeglasses", "Male", "Mouth_Slightly_Open", "Smiling", "Young"]
label_path = "data/list_attr_celeba.txt"
with open(label_path) as f:
    lines = f.readlines()
    label_names_id = []
    all_labels = lines[1].strip().split(" ")
    for label_name in label_names:
        label_names_id.append(all_labels.index(label_name))
    data = []
    for line in lines[2:]:
        img_name, img_labels = line.split(" ", 1)
        label_str = img_labels.replace("  ", " ").split(" ")
        labels = [0 if label_str[i] == "-1" else 1 for i in label_names_id]
        data.append(labels)
    data = np.array(data)

for i, label_name in enumerate(label_names):
    label1 = np.count_nonzero(data[:, i])
    label0 = len(data) - label1
    plt.bar([0, 1], [label0, label1])
    plt.title(label_name)
    plt.show()