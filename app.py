from stage_1 import meta_data_pipeline
from stage_2 import load_data, process_batch,update_faiss,mark_processed
from pathlib import Path




if __name__ == '__main__':

    # STAGE_1 - Extracting Metadata
    # File Paths
    folder = Path("/Users/manan/Desktop/my_own_codes/AK-Gift/photos/")
    image_paths = list(folder.glob("*"))

    #Getting all the MetaData from the image
    meta_data_pipeline(image_paths)
 
 


    # STAGE_2 - Using Metadata and Images for Vector Embeddings
    while(1):
        data = load_data()

        if not data:
            break

        file_names, metadata_list, embeddings = process_batch(data)

        if len(embeddings) > 0:
            update_faiss((file_names, metadata_list, embeddings))
            mark_processed(file_names)

    print("🚀 Pipeline completed")