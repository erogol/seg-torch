# Dataset loader description
Data loader reads raw image files with multi-threading support. It only requires
a certain folder structure for the provided dataset. For the first run of the code,
it parses file paths, computes mean, std and class weights statistics and save the all
in a file.

Sample working data-loader is given for [CamVid](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)
which uses special format then the original dataset.

## Folder/file structure for each dataset:
    ```
    CamVid/
    ├── train
    ├── trainannot
    ├── val
    └── valannot
    ```

# Creating your own data loader
- Create dataset-gen.lua for parsing dataset paths and sacve to a file. It also
computes mean, std and class weights.
- Create dataset.lua for real-time image and mask pair loading.
In general you need to modify followings.
    - "get()" loads image/label pairs
    - "preprocess()" applies preprocessing like augmentation, scaling etc.
    
