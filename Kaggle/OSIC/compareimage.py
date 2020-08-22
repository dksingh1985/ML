import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
import re
import pydicom


sns.set(color_codes=True)

custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']

train = pd.read_csv("D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\train.csv")

train = pd.DataFrame(train['Patient'].unique())

train.columns = ['Patient']

# Create base director for Train .dcm files
director = "D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\train"

# Create path column with the path to each patient's CT
train["Path"] = director + "\\" + train["Patient"]

for k, path in enumerate(train["Path"]):
    train.iloc[k,1] = train.iloc[k,1] + "\\" + os.listdir(train.iloc[k,1])[0]
    print("Path : " + train.iloc[k,1])

from PIL import Image
from IPython.display import Image as show_gif
import scipy.misc
import matplotlib


def create_gif_common():

    if os.path.isdir(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_All") == False:
        os.mkdir(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_All")
        os.mkdir(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_Al2")

    #iterate all patients DCM file
    for k, path in enumerate(train["Path"]):
        print(k, " --------> ", path)
        try:
            img1 = pydicom.dcmread(path).pixel_array
            img1 = Image.fromarray(img1).convert("L")
            img1.save(f'D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_Al2\\img_{k}.png')
            img2 = img1.resize((512, 512),Image.ANTIALIAS)
            print(k,img1.size, " ----> ", img2.size)
            img2.save(f'D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_All\\img_{k}.png')

            '''
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(img1)
            ax.set_title('Before')
            plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
            ax = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(img2)
            imgplot.set_clim(0.0, 0.7)
            ax.set_title('After')
            plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
            '''
            
        except:
            print("Error : ###############################################")

    '''
    for i in range(len(datasets)):
        try:
            img = datasets[i].pixel_array
            img = Image.fromarray(img).convert("L")
            img.resize((512, 512),Image.ANTIALIAS)
            print(i, "----> ", img.size)
            img.save(f'D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_All\\img_{i}.png')
        except:
            print("Error : ", i, "----> ",train.iloc[i,1])
            
    for png in list(os.listdir(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_All")):
        img = Image.open("D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_All\\" + png)#.convert('LA')
        #img.thumbnail((64, 64), Image.ANTIALIAS)  # resizes image in-place
        img = img.resize((512, 512), Image.ANTIALIAS)
        #imgplot = plt.imshow(img)
        img.save("D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_All\\" + png)
    '''
    
    files = []
    # === CREATE GIF ===
    # First Order the files in the dataset (again)
    for png in list(os.listdir(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_All")):
        files.append(png) 
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Create the frames
    frames = []

    # Create frames
    for file in files:
    #     print("../working/png_images/" + name)
        new_frame = Image.open(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_All\\" + file)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(f'D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\gif_All.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=200, loop=0)
        
create_gif_common()
