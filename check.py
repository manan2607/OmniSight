from pathlib import Path
import shutil

def move_images_safe(image_paths, destination_folder):
    destination = Path(destination_folder)
    destination.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        src = Path(img_path)

        if not src.exists():
            print(f"❌ Not found: {src}")
            continue

        dest = destination / src.name

        # Handle duplicate names
        counter = 1
        while dest.exists():
            dest = destination / f"{src.stem}_{counter}{src.suffix}"
            counter += 1

        shutil.move(str(src), str(dest))
        print(f"✅ Moved: {src.name} → {dest.name}")


move_images_safe()