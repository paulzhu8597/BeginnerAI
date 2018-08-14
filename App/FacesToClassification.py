'''
将名人人脸数据集按照不同的特征进行分类
5_o_Clock_Shadow
Arched_Eyebrows
Attractive
Bags_Under_Eyes
Bald
Bangs
Big_Lips
Big_Nose
Black_Hair
Blond_Hair
Blurry
Brown _Hair
Bushy_Eyebrows
Chubby
Double_Chin
Eyeglasses
Goatee
Gray_Hair
Heavy_Makeup
High_Cheekbones
Male
Mouth_Slightly_Open
Mustache
Narrow_Eyes	No_Beard	Oval_Face	Pale_Skin	Pointy_Nose	Receding_Hairline	Rosy_Cheeks	Sideburns
Smiling	Straight_Hair	Wavy_Hair	Wearing_Earrings	Wearing_Hat	Wearing_Lipstick
Wearing_Necklace
Wearing_Necktie
Young
'''
import os
import shutil
# import face_recognition
import pandas as pd

# from PIL import Image
from tqdm import tqdm

FACE_ROOT_PATH = os.path.join("/input", "Faces")
SOURCE_IMAGE_PATH = os.path.join(FACE_ROOT_PATH, "Images", "celeba")
TARGET_IMAGE_PATH = os.path.join(FACE_ROOT_PATH, "SquareImages", "celeba")

FACE_ATTRIBUTE_FILE_PATH = os.path.join(FACE_ROOT_PATH, "face_attr.csv")

# def ToSquareImages():
#     filelist = os.listdir(SOURCE_IMAGE_PATH)
#
#     for file in tqdm(filelist):
#         imgName = os.path.basename(file)
#         if (os.path.splitext(imgName)[1] != ".jpg"):
#             continue
#
#         image = face_recognition.load_image_file(os.path.join(SOURCE_IMAGE_PATH, imgName))
#         face_locations = face_recognition.face_locations(image)
#         for face_location in face_locations:
#             top, right, bottom, left = face_location
#             width = right - left
#             height = bottom - top
#             if (width > height):
#                 right -= (width - height)
#             elif (height > width):
#                 bottom -= (height - width)
#             face_image = image[top:bottom, left:right]
#             pil_image = Image.fromarray(face_image)
#             pil_image.save(os.path.join(TARGET_IMAGE_PATH, imgName))

def ToClassification(feature_name):
    DATA_PATH = os.path.join(FACE_ROOT_PATH, feature_name)
    os.makedirs(DATA_PATH)
    os.makedirs(os.path.join(DATA_PATH, "0"))
    os.makedirs(os.path.join(DATA_PATH, "1"))

    ATTR_FILE_PATH = FACE_ATTRIBUTE_FILE_PATH
    data = pd.read_csv(ATTR_FILE_PATH, usecols=("File_Names", feature_name)).as_matrix()

    pbar = tqdm(enumerate(data))

    for index, record in pbar:
        pbar.set_description("loading images : %s" % (len(data) - index - 1))
        filename = record[0]
        label = record[1]
        label = 0 if label == -1 else 1
        if os.path.exists(os.path.join(TARGET_IMAGE_PATH, filename)) == False:
            continue
        shutil.copy(os.path.join(TARGET_IMAGE_PATH, filename),
                    os.path.join(DATA_PATH, "%s" % label))

if __name__ == '__main__':
    # ToSquareImages()
    ToClassification("Eyeglasses")







