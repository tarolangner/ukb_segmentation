import numpy as np

###### 
# Generate a k-fold cross-validation split from a list of ids.
# Any ids that should be added only for training can be listed 
# in a separate file at a later stage.

split_name = "kidney_64_8fold"
split_path = "../splits/" + split_name + "/"

id_list = split_path + "id_list.txt"

# Number of cross-validations sets
K = 8

# Read names of available images
with open(id_list) as f: img_names = f.read().splitlines() 

# Split on subject level into random sets
rand_perm = np.random.permutation(len(img_names))
sets = np.array_split(rand_perm, K)

# Write image names by set
for k in range(K):

    sets[k] = np.sort(sets[k])
    file_k = open(split_path + "images_set_{}.txt".format(k),"w") 

    for i in range(len(sets[k])):
        file_k.write("{}\n".format(img_names[sets[k][i]]))

    file_k.close() 
