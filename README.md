# Microinfarcts
* Microinfarcts is a project for loacting the real location of beads inside the brain and use ANTs to align the brain into Allen atlas. 
* First human labeled position of micro infarcts inside the brain images will be clustered and organized, after some calculation they will be transformed into a list of masks containing the real location of the micro infarcts. After that the transform matrix achieved in aligning the brain into atlas will be applied on the masks. Then the masks as well as the Allen annotaion files will be involved in coming processes and registered into the standard Allen atlas. Finally you will have two optional ways to view your result.
* 1. A csv file indicating the number of micro infarcts located in different brain regions.
* 2. A opencv window showing the standard atlas, symmerically normalized brain images and the location of the micro infarcts. 
* ![opencv window](https://github.com/SilasiLab/microinfarcts/blob/master/pics/show.png)
* The reference atlas comes from Allen Atlas organization. You can find reference data on google drive link attached here:(https://drive.google.com/drive/folders/10MqL8BXkfRsjLgWRuoxJUuZzH9AuWIOe?usp=sharing)
* After downloading the reference file, you need to copy it into `atlas_reference` folder.
* So the whole structure of the project should be:
    * `microinfarcts/src`
    * `microinfarcts/atlas_reference`
    * `microinfarcts/atlas_reference/annotation.mhd`
    * `microinfarcts/atlas_reference/annotation.raw`
    * `microinfarcts/atlas_reference/atlasVolume.mhd`
    * `microinfarcts/atlas_reference/atlasVolume.raw`

## 1. Install dependencies
 * 1. `conda install pandas`
 * 2. `conda install shlex`
 * 3. `conda install subprocess`
 * 4. `conda install -c conda-forge ffmpeg`
 * 5. `conda install -c conda-forge opencv`
 * 6. `conda install matplotlib`
 * 7. `conda install pickle`
 * 8. `conda install tqdm`
 * 9. `conda install skimage`
 * 10. `pip install nipype`
 * 11. Download and compile ANTs from (https://brianavants.wordpress.com/2012/04/13/updated-ants-compile-instructions-april-12-2012/)
 * 12. `git clone https://github.com/SilasiLab/microinfarcts.git`

## 2. Preparatory phase
  * Microinfarcts is based on the result given by imageJ process. The input data should be the result of imageJ process.
  * For the input raw data, the whole directory structure should be:
  * 1. `root directory/[brain id](individual brain)/3 - Processed Images/7 - Counted Reoriented Stacks Renamed/*imgs`
  * 2. `root directory/[brain id](individual brain)/5 - Data/[brain id] - Manual Bead Location Data v0.1.4 - Dilation Factor 0.csv`
  * Note: The first directory should contain the brain images aligned by imageJ. And under the second one there should be a csv containing the human labeled micro infarcts loaction.
  * After downloading as well as compiling ANTs, you should find the dirctory of `antsRegistrationSyNQuick.sh` under ANTs `Scripts` folder. Take this PC as an example, it is `/home/silasi/ANTs/Scripts/antsRegistrationSyNQuick.sh`. Then the folder containing `antsRegistrationSyNQuick.sh`, that is `/home/silasi/ANTs/Scripts/` which will be used as the parameter `--ant` of the whole project. Here we leave it as [Script folder] for short and for future use.

## 3. User guide
  * 1. Simple guide.
      * Write a summary:
        * 1. [Input directory]: The folder holds individual brains folders.
        * 2. [Output directory]: A empty folder you would like to save the result.
        * 3. `cd microinfarcts/src`
        * 4. `python main.py --r [Input directory] --s [Output directory] --ant [Script folder]`
        * 5. Microinfarcts script will run through brains. It will take a while to finish the whole process. After running, there will be a csv file named as `summary.csv` under the `[output directory]/[brain id]`.
      * Show in image:
        * 1. First three steps are the same as `write a summary`.
        * 2. If you have already gone through the previous step then the command should be `python main.py --r [Input directory] --s [Output directory] --ant [Script folder] --p False --re False --a True --w False --sh True` 
        * 3. Then it will show up a window presenting the result.
  * 2. Detailed guide.
      * `--r`: Root directory or input directory, indicates the root directory mentioned in `2.Preparatory phase`, which holds individual brains.
      * `--s`: Save directory or output directory, an empty folder to save your middle results and final results.
      * `--ant`: The directory to script folder under ANTs directory.
      * `--p`: Read the atlas file and annotation file and transfer them into .pickle data. Default is `True`, if you have already finshed one whole process before or have those pickle file saved under `atlas_reference` folder, then you may set this parameter as `False` to skip it and save time.
      * `--re`: Use ANTs as backend to align the brain into standard Allen atlas. Default is `True`, Similar to the previous parameter, if you already have the middle result saved in `[Output directory]/output` folder, then you may set it as `False `to skip it and save time.
      * `--a`: Apply the trasform matrix calculated in ANTs registration on mask of micro infarcts as well as the tissue image. Default is `True`. Please always set it as `True` to avoide hindering the next process.
      * `--w`: Write the registraion result into a csv file. This function is not compatible with `show` function. So the parameter `--w` cannot be set as `True` while `--sh` is already set to `True`. 
      * `--sh`: Show the alignment and registration result in a opencv window. This function is not compatible with `write_summary` function. So the parameter `--sh` cannot be set as `True` while `--w` is already set to `True`. 
