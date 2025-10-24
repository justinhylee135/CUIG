from constants.const import theme_available, class_available
import os
import shutil

if __name__ == "__main__":

    # For style unlearning
    original_data_dir = "/fs/scratch/PAS2099/unlearn_canvas/"
    new_dir = '/fs/scratch/PAS2099/lee.10369/ediff-sample'
    prompts_and_path_dir = '/users/PAS2099/justinhylee135/Research/UnlearningDM/CUIG/UnlearningMethods/EraseDiff/data'
    num_indices = 20
    print(f"Collecting theme datasets...")
    for theme in theme_available:
        theme_dir = os.path.join(new_dir, theme)
        os.makedirs(theme_dir, exist_ok=True)
        prompt_list = []
        path_list = []
        for class_ in class_available:
            for idx in range(1, num_indices + 1):
                # Build source and destination paths
                source_path = os.path.join(original_data_dir, theme, class_, f"{idx}.jpg")
                dest_filename = f"{class_}_{idx}.jpg"
                dest_path = os.path.join(theme_dir, dest_filename)
                # Prepare prompt and record destination path
                prompt_list.append(f"A {class_} image in {theme.replace('_', ' ')} style.")
                path_list.append(dest_path)
                # Copy the image from the source to the destination
                shutil.copy(source_path, dest_path)

        # Write the prompts and destination paths to text files
        with open(os.path.join(prompts_and_path_dir, theme, 'prompts.txt'), 'w') as f:
            f.write('\n'.join(prompt_list))
        with open(os.path.join(prompts_and_path_dir, theme, 'images.txt'), 'w') as f:
            f.write('\n'.join(path_list))

    # For Seed_Images style
    seed_theme = "Seed_Images"
    seed_theme_dir = os.path.join(new_dir, seed_theme)
    os.makedirs(seed_theme_dir, exist_ok=True)
    prompt_list = []
    path_list = []
    print(f"Collecting Seed_Images datasets...")
    for class_ in class_available:
        for idx in range(1, num_indices + 1):
            source_path = os.path.join(original_data_dir, seed_theme, class_, f"{idx}.jpg")
            dest_filename = f"{class_}_{idx}.jpg"
            dest_path = os.path.join(seed_theme_dir, dest_filename)
            prompt_list.append(f"A {class_} image in Photo style.")
            path_list.append(dest_path)
            shutil.copy(source_path, dest_path)

    with open(os.path.join(prompts_and_path_dir, 'Seed_Images', 'prompts.txt'), 'w') as f:
        f.write('\n'.join(prompt_list))
    with open(os.path.join(prompts_and_path_dir, 'Seed_Images', 'images.txt'), 'w') as f:
        f.write('\n'.join(path_list))

    # For class unlearning
    print(f"Collecting object datasets...")
    for object_class in class_available:
        class_dir = os.path.join(new_dir, object_class)
        os.makedirs(class_dir, exist_ok=True)
        prompt_list = []
        path_list = []
        for theme in theme_available:
            for idx in range(1, num_indices + 1):
                source_path = os.path.join(original_data_dir, theme, object_class, f"{idx}.jpg")
                dest_filename = f"{theme}_{idx}.jpg"
                dest_path = os.path.join(class_dir, dest_filename)
                prompt_list.append(f"A {object_class} image in {theme.replace('_', ' ')} style.")
                path_list.append(dest_path)
                shutil.copy(source_path, dest_path)

        with open(os.path.join(prompts_and_path_dir, object_class,'prompts.txt'), 'w') as f:
            f.write('\n'.join(prompt_list))
        with open(os.path.join(prompts_and_path_dir, object_class, 'images.txt'), 'w') as f:
            f.write('\n'.join(path_list))
