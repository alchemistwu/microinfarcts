# microinfarcts
* Microinfarcts is a project for loacting the real location of beads inside the brain and use ANTs to align the brain into Allen atlas.
* The reference atlas is taken from Allen Atlas organization. You can find reference data on google drive link attached here:(pass)
* After downloading the reference file, you need to copy it into `atlas_reference` folder.
* So the whole structure of the project should be:
    * `microinfarcts/src`
    * `microinfarcts/atlas_reference`
    * `microinfarcts/atlas_reference/annotation.mhd`
    * `microinfarcts/atlas_reference/annotation.raw`
    * `microinfarcts/atlas_reference/atlasVolume.mhd`
    * `microinfarcts/atlas_reference/atlasVolume.raw`

## 1. Install dependencies
 * a. `conda install pandas`
 * b. `conda install shlex`
 * c. `conda install subprocess`
 * d. `conda install -c conda-forge ffmpeg`
 * e. `conda install -c conda-forge opencv`
 * f. `conda install matplotlib`
 * g. `conda install pickle`
 * h. `conda install tqdm`
 * i. `conda install skimage`
 * k. `pip install nipype`

## 2. Preparatory phase
  * Microinfarcts is based on the result given by imageJ process.
