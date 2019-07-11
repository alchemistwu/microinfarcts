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
 * 11. Download and compile ANTs from `https://brianavants.wordpress.com/2012/04/13/updated-ants-compile-instructions-april-12-2012/`

## 2. Preparatory phase
  * Microinfarcts is based on the result given by imageJ process.
