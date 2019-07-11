'''
author: Junzheng Wu
Email: jwu220@uottawa.ca
github: alchemistWu0521@gmail.com
Organization: Silasi Lab
'''
from nipype.interfaces.ants import RegistrationSynQuick
from nipype.interfaces.ants.resampling import ApplyTransforms, ApplyTransformsToPoints
import os
import shlex
import subprocess

def quick(dir_fix_image='/home/silasi/ants_data/nrrd/0.tif', dir_moving_img='/home/silasi/ants_data/tissue/0.tif', dir_output='/home/silasi/ants_data/output_'):
    """
    Ants registration function
    :param dir_fix_image:
    :param dir_moving_img:
    :param dir_output:
    :return:
    """
    reg = RegistrationSynQuick()
    reg.inputs.dimension = 2
    reg.inputs.fixed_image = dir_fix_image
    reg.inputs.moving_image = dir_moving_img
    reg.inputs.output_prefix = dir_output
    reg.inputs.transform_type = 's'
    reg.inputs.num_threads = 16
    command = os.path.join("/home/silasi/ANTs/Scripts/", reg.cmdline)
    args = shlex.split(command)
    p = subprocess.Popen(args)
    p.wait()

def apply_transform(input_img, reference_img, transforms, output_img):
    """

    :param input_img:
    :param reference_img:
    :param transforms: should be a list of .mat and warp.nii.gz
    :param output_img:
    :return:
    """
    at1 = ApplyTransforms()
    at1.inputs.dimension = 2
    at1.inputs.input_image = input_img
    at1.inputs.reference_image = reference_img
    at1.inputs.output_image = output_img
    # at1.inputs.interpolation = 'BSpline'
    # at1.inputs.interpolation_parameters = (5,)
    at1.inputs.default_value = 0
    at1.inputs.transforms = transforms
    at1.inputs.invert_transform_flags = [False, False]
    args = shlex.split(at1.cmdline)
    p = subprocess.Popen(args)
    p.wait()

def apply_transform_2_points(input_csv, transforms, output_csv):

    at = ApplyTransformsToPoints()
    at.inputs.dimension = 2
    at.inputs.input_file = input_csv
    at.inputs.transforms = transforms
    at.inputs.invert_transform_flags = [False, False]
    at.inputs.output_file = output_csv
    args = shlex.split(at.cmdline)
    p = subprocess.Popen(args)
    p.wait()