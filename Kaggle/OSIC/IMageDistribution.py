import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import pydicom

sns.set(color_codes=True)

custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']

train = pd.read_csv("D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\train.csv")


# Create base director for Train .dcm files
director = "D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\train"

# Create path column with the path to each patient's CT
train["Path"] = director + "/" + train["Patient"]

# Create variable that shows how many CT scans each patient has
train["CT_number"] = 0

for k, path in enumerate(train["Path"]):
    print("--------> ", path)
    train["CT_number"][k] = len(os.listdir(path))

print("Minimum number of CT scans: {}".format(train["CT_number"].min()), "[" , train[train["CT_number"]==12].Patient.unique()[0] , "]" , "\n" + "Maximum number of CT scans: {:,}".format(train["CT_number"].max()), train[train["CT_number"]==1018].Patient.unique()[0] , "]" )

# Scans per Patient
data = train.groupby(by="Patient")["CT_number"].first().reset_index(drop=False)
# Sort by Weeks
data = data.sort_values(['CT_number']).reset_index(drop=True)

# Plot
plt.figure(figsize = (16, 6))
p = sns.barplot(data["Patient"], data["CT_number"], color=custom_colors[5])
plt.axvline(x=85, color=custom_colors[2], linestyle='--', lw=3)

plt.title("Number of CT Scans per Patient", fontsize = 17)
plt.xlabel('Patient', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.text(86, 850, "Median=94", fontsize=13)

p.axes.get_xaxis().set_visible(False);

plt.show()

#######################################################################################
#GIF from Images

from PIL import Image
from IPython.display import Image as show_gif
import scipy.misc
import matplotlib


def create_gif(number_of_CT = 87):
    """Picks a patient at random and creates a GIF with their CT scans."""
    
    # Select one of the patients
    # patient = "ID00007637202177411956430"
    patient = train[train["CT_number"] == number_of_CT].sample(random_state=1)["Patient"].values[0]

    
    # === READ IN .dcm FILES ===
    patient_dir = director + "\\" + patient
    datasets = []

    # First Order the files in the dataset
    files = []
    for dcm in list(os.listdir(patient_dir)):
        files.append(dcm) 
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Read in the Dataset from the Patient path
    for dcm in files:
        path = patient_dir + "\\" + dcm
        datasets.append(pydicom.dcmread(path))
        
        
    # === SAVE AS .png ===
    # Create directory to save the png files
    if os.path.isdir(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_{patient}") == False:
        os.mkdir(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_{patient}")

    # Save images to PNG
    for i in range(len(datasets)):
        img = datasets[i].pixel_array
        matplotlib.image.imsave(f'D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_{patient}\\img_{i}.png', img)
        
      
    # === CREATE GIF ===
    # First Order the files in the dataset (again)
    files = []
    for png in list(os.listdir(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_{patient}")):
        files.append(png) 
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Create the frames
    frames = []

    # Create frames
    for file in files:
    #     print("../working/png_images/" + name)
        new_frame = Image.open(f"D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\png_{patient}\\" + file)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(f'D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\gif\\gif_{patient}.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=200, loop=0)

    ##########################################Create and compare GIFs

# create_gif(number_of_CT=12)
# create_gif(number_of_CT=30)
# create_gif(number_of_CT=87)
# create_gif(number_of_CT=1018)

# print("First file len:", len(os.listdir("../working/png_ID00165637202237320314458")), "\n" +
#       "Second file len:", len(os.listdir("../working/png_ID00199637202248141386743")), "\n" +
#       "Third file len:", len(os.listdir("../working/png_ID00340637202287399835821")))

#show_gif(filename=".\\gif_ID00165637202237320314458.gif", format='png', width=400, height=400)
