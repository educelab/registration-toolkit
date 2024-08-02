Aligning images with registration-toolkit
=========================================

This guide provides an introduction to image alignment in 
`registration-toolkit`. To follow this tutorial, please download the 
[fixed](fixed.jpg) and [moving](moving.jpg) example images.

## Prerequisites
### Install registration-toolkit

```shell
# get source and dependencies
git clone https://gitlab.com/educelab/registration-toolkit.git
cd registration-toolkit
brew bundle

# build the software
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build/

# add the build/bin directory to PATH
export PATH=${PATH}:${PWD}/build/bin
```

## Image alignment with automatic landmark selection
```shell
# saves the auto-generated landmarks to landmarks.ldm
rt_register -f fixed.jpg -m moving.jpg -o registered.jpg

# generate a diff image with ImageMagick
magick fixed.jpg registered.jpg -compose difference -composite -auto-level registered-diff.jpg 
```

## Image alignment with manual landmark selection

### Install Landmarks Picker
```shell
brew install --no-quarantine educelab/casks/landmarks-picker
```

### Manually select matching landmarks
1. Open `Landmarks Picker` 
   * On macOS, you may need to right-click the program in the Applications 
     directory and select `Open` the first time you run the program.
2. File → Open fixed... and select `fixed.jpg`.
3. File → Open moving... and select `moving.jpg`.
4. In the image viewers, scroll to zoom in/out and 
   click + drag to pan.
5. In the bottom of the Landmarks panel, click the + sign 
   to add a new landmark to the list.
6. Move the images until you find a matching feature in 
   both images. Double-click the matching feature in both 
   images. The coordinates of the landmark should update in 
   the Landmarks panel.
7. Repeat steps 5 and 6 until you have at least 4 landmarks.
8. In the bottom of the Landmarks panel, click the save 
   button (looks like a downward arrow) and save your file 
   as `landmarks.ldm` next to the images.
9. In the Terminal, run the following command in the same
   directory as the images (make sure you've run the export
   command from the installation instsructions):

### Run registration with manual landmarks
```shell
# register the images
rt_register -f fixed.jpg -m moving.jpg -o registered.jpg -l landmarks.ldm

# generate a diff image with ImageMagick
magick fixed.jpg registered.jpg -compose difference -composite -auto-level registered-diff.jpg 
```

### Viewing/editing automatic landmarks
When running `rt_register`, add `--output-ldm {file}.ldm` to save the 
automatically generated landmarks to disk. This file can then be opened in 
`Landmarks Picker` using the folder icon in the bottom of the Landmarks panel. 
When you select a landmark pair in the landmarks list, the image viewers will
auto-navigate to make the landmarks visible (make sure to zoom in first.)
