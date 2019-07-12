from image_processing_utils import *
import argparse

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--r', help='The root directory of data given by imageJ process.', dest='root_dir', type=str)
parser.add_argument('--s', help='The directory for saving result', dest='save_dir', type=str)
parser.add_argument('--ant', help='The directory for Ants scrips', dest='Ants_script', default="/home/silasi/ANTs/Scripts", type=str)
parser.add_argument('--p', help='Whether to prepare the atlas files and tissue images or not', dest='prepare_atlas_tissue', default=True, type=boolean_string)
parser.add_argument('--re', help='Whether to use ANTs to register or not', dest='registration', default=True, type=boolean_string)
parser.add_argument('--a', help='Apply transform on micro infarcts mask and tissue images', dest='app_tran', default=True, type=boolean_string)
parser.add_argument('--w', help='Write a csv summary, cannot be used alongside show', dest='write_summary', default=True, type=boolean_string)
parser.add_argument('--sh', help='Show result in a preview window, cannot be used alongside write csv', dest='show', default=False, type=boolean_string)

args = parser.parse_args()

root_dir = args.root_dir
save_dir = args.save_dir
prepare_atlas_tissue = args.prepare_atlas_tissue
registration = args.registration
app_tran = args.app_tran
write_summary = args.write_summary
show = args.show
Ants_script = args.Ants_script


main(root_dir=root_dir, save_dir=save_dir, prepare_atlas_tissue=prepare_atlas_tissue, registration=registration, Ants_script=Ants_script, app_tran=app_tran, write_summary=write_summary, show=show)
