import os
import shutil


def setup_resnet_data():
    original_data = r"C:\Users\Surface\PycharmProjects\PythonProject2\imageData"
    new_data = r"C:\Users\Surface\PycharmProjects\PythonProject2\RESNET_CARTOONIST_DATA"

    if not os.path.exists(new_data):
        os.makedirs(new_data)
        for artist in ['Pendleton', 'Tartakovsky', 'Timm']:
            src = os.path.join(original_data, artist)
            if os.path.exists(src):
                shutil.copytree(src, os.path.join(new_data, artist))
                print(f"Copied {artist}")
    return new_data


if __name__ == "__main__":
    setup_resnet_data()