# Importing Libraries
import shutil
import pathlib
import os

type1 = "chihuahua"
type2 = "muffin"

def categorize_folder(path_name, naming_start, pic_extension):
    path = r'./'+ path_name+ '/'
    counter = 1
    for file in os.listdir(path):
        old_full_path = path + file
        new_full_path = path + naming_start + str(counter) + pic_extension

        if not os.path.isfile(new_full_path):
            os.rename(old_full_path, new_full_path)
            counter += 1       
    print(os.listdir(path))

def segregate_dataset(type, dataset_type, indx_first, indx_last):
    old_path = pathlib.Path(type)
    new_path = pathlib.Path(type1 + '_' + type2)

    path = new_path / dataset_type / type
    if not os.path.isdir(path):
        os.makedirs(path)
    all_path = []
    for i in range(indx_first, indx_last):
        all_path.append(type + '_' + str(i) + '.jpg')
    for paths in all_path:
        shutil.copyfile(src=old_path / paths, dst=path / paths)

#categorize_folder(type1, type1 + '_', ".JPG")
#categorize_folder(type2, type2 + '_', ".JPG")

segregate_dataset(type1, "train", 1, 251)
segregate_dataset(type1, "validation", 251, 401)
segregate_dataset(type1, "test", 401, 501)
segregate_dataset(type2, "train", 1, 251)
segregate_dataset(type2, "validation", 251, 401)
segregate_dataset(type2, "test", 401, 501)
