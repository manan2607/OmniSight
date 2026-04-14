from stage_1 import meta_data_pipeline
from stage_2 import load_data, process_batch,npz_file_making
from pathlib import Path




if __name__ == '__main__':

    # STAGE_1 - Extracting Metadata
    # File Paths
    folder = Path("/Users/manan/Desktop/my_own_codes/AK-pic/photos/")
    image_paths = list(folder.glob("*"))

    #Getting all the MetaData from the image
    meta_data_pipeline(image_paths)
 



    # STAGE_2 - Using Metadata and Images for Vector Embeddings
    total = 0
    while(1):
        # Loading Data
        data = load_data()
        total+=len(data)
        if not data:
            break
        # Processing
        vectors = process_batch(data)

        # Storing
        npz_file_making(vectors)

    print(f"✅ Saved {total} embeddings")