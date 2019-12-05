#Date:      4dec-2019
#Developer: Dennis Croon

import jsonlines
import numpy as np
from tqdm import tqdm
from pathlib import Path
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


def main():
    model = VGG16(weights='imagenet', include_top=False)
    image_paths = [x for x in Path("data/").glob('**/*') if x.is_file() and ".jpeg" in x.name]
    output = []
    for fpath in tqdm(image_paths, desc="Extracting Features", total=len(image_paths)):
        split_path = str(fpath).split("/")  
        img = image.load_img(fpath, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        output.append({
            "file_path": str(fpath),
            "dataset": split_path[1],
            "label": split_path[2],
            "file_name": split_path[3],
            "features": features.tolist()
        })
    with jsonlines.open("features.jsonl", "w") as outfile:
        outfile.write_all(output)

if __name__ == "__main__":
    main()