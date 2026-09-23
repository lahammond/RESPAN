# -*- coding: utf-8 -*-
"""
Main functions
==========


"""
__title__     = 'RESPAN'
__version__   = '1.5.01'
__date__      = "20 September, 2026"
__author__    = 'Luke Hammond <lh2881@columbia.edu>'
__license__   = 'MIT License (see LICENSE)'
__copyright__ = 'Copyright © 2022 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/RESPAN'


import os
import sys
import yaml
import ast

##############################################################################
# Main Functions
##############################################################################
Locations = None
Settings = None

#create dir   
def create_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

#count dirs
def count_dirs(path):
    count = 0
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            count += 1

    return count

#count files
def count_files(path):
    count = 0
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            count += 1

    return count

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def check_gpu():
  """
    if tf.test.gpu_device_name()=='':
    print('You do not have GPU access.') 

  else:
    print('You have GPU access')
    #!nvidia-smi
"""

#place holder for creating pipeline dirs
def create_dirs(Settings, Locations):

    create_dir(Locations.tables)

    #create_dir(Locations.plots)
    create_dir(Locations.validation_dir)
    create_dir(Locations.labels)
    create_dir(Locations.restored)
    create_dir(Locations.nnUnet_input)
    create_dir(Locations.arrays)
    create_dir(Locations.Meshes)
    #create_dir(Locations.nnUnet_2nd_pass)

    create_dir(Locations.MIPs)
    create_dir(Locations.Vols)

    
    
def initialize_RESPAN(data_dir):
    
    Settings = Create_Settings(data_dir)
    Locations =  Create_Locations(data_dir)
    create_dirs(Settings, Locations)
    
    return Settings, Locations

def initialize_RESPAN_validation(data_dir):
    
    Settings = Create_Settings(data_dir)
    Locations =  data_dir
    
    return Settings, Locations
    
class ConfigObject:
    def __init__(self, data):
        self.__dict__.update(data)


 
##############################################################################
# Main Classes
##############################################################################
class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Create_Locations():
    def __init__(self, data_dir):
        # Normalize data_dir to ensure consistent separators
        data_dir = os.path.normpath(str(data_dir)) + os.sep
        self.input_dir = data_dir
        self.validation_dir = os.path.join(data_dir, "Validation_Data") + os.sep
        self.labels = os.path.join(data_dir, "Validation_Data", "Segmentation_Labels") + os.sep
        self.restored = os.path.join(data_dir, "Validation_Data", "Restored_Images") + os.sep
        self.MIPs = os.path.join(data_dir, "Validation_Data", "Validation_MIPs") + os.sep
        self.Vols = os.path.join(data_dir, "Validation_Data", "Validation_Vols") + os.sep
        self.Meshes = os.path.join(data_dir, "Validation_Data", "Spine_Meshes") + os.sep
        self.tables = os.path.join(data_dir, "Tables") + os.sep
        self.plots = os.path.join(data_dir, "Plots") + os.sep
        self.arrays = os.path.join(data_dir, "Spine_Arrays") + os.sep
        self.nnUnet_input = os.path.join(data_dir, "nnUnet_input") + os.sep
        self.nnUnet_2nd_pass = os.path.join(data_dir, "nnUnet_2nd_pass") + os.sep
        self.swcs = os.path.join(data_dir, "SWC_files") + os.sep
      
    def inspect(self):
        for attr_name in dir(self):
            if not callable(getattr(self, attr_name)) and not attr_name.startswith("__"):
                value = getattr(self, attr_name)
                print(f"{attr_name}: {value}")


class Create_Settings():

    def __init__(self, data_dir):
        settings_file = os.path.join(str(data_dir), "Analysis_Settings.yaml")
        if not os.path.exists(settings_file):
            # Auto-create from template with default settings
            template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Templates")
            template_file = os.path.join(template_dir, "Analysis_Settings.yaml")
            if os.path.exists(template_file):
                import shutil
                shutil.copy2(template_file, settings_file)
                print(f"  Created Analysis_Settings.yaml from template in '{data_dir}'.\n"
                      f"  Review and update resolution parameters before running analysis.")
            else:
                raise FileNotFoundError(
                    f"Analysis_Settings.yaml not found in '{data_dir}' "
                    f"and template not found at '{template_file}'."
                )
        with open(settings_file, 'r') as ymlfile:
            #setting = yaml.safe_load(ymlfile)
            setting = yaml.load(ymlfile, Loader = PrettySafeLoader)
            
            self.input_resXY = setting["Parameters"]["input_resXY"]
            self.input_resZ = setting["Parameters"]["input_resZ"]
            self.model_resXY = setting["Parameters"]["model_resXY"]
            self.model_resZ = setting["Parameters"]["model_resZ"]
            
            self.tiles_for_prediction = list(map(int, (ast.literal_eval(setting["Analysis"]["tiles_for_prediction"]))))
            self.roi_volume_size = setting["Analysis"]["roi_volume_size"]            
            self.GPU_block_size = list(map(int, (ast.literal_eval(setting["Analysis"]["GPU_block_size"]))))
            self.erode_shape = list(map(int, (ast.literal_eval(setting["Analysis"]["erode_shape"]))))
            self.remove_touching_boarders = setting["Analysis"]["remove_touching_boarders"]
            
            self.validation_format = setting["Analysis"]["validation_format"]
            self.validation_scale = list(map(int, (ast.literal_eval(setting["Analysis"]["validation_scale"]))))
            self.validation_jpeg_comp = setting["Analysis"]["validation_jpeg_comp"]
            try:
                self.additional_logging = setting["Analysis"]["additional_logging"]
            except (KeyError, TypeError):
                self.additional_logging = False
  
            self.c1_restore = setting["Channel1"]["restore"]
            self.c1_rest_model_path = setting["Channel1"]["rest_model_path"]
            self.c1_rest_type = ast.literal_eval(setting["Channel1"]["rest_type"])
            
            self.c2_restore = setting["Channel2"]["restore"]
            self.c2_rest_model_path = setting["Channel2"]["rest_model_path"]
            self.c2_rest_type = ast.literal_eval(setting["Channel2"]["rest_type"])
            
            self.c3_restore = setting["Channel3"]["restore"]
            self.c3_rest_model_path = setting["Channel3"]["rest_model_path"]
            self.c3_rest_type = ast.literal_eval(setting["Channel3"]["rest_type"])
            
            self.c4_restore = setting["Channel4"]["restore"]
            self.c4_rest_model_path = setting["Channel4"]["rest_model_path"]
            self.c4_rest_type = ast.literal_eval(setting["Channel4"]["rest_type"])

            #skip if not found in the file in case older setting file
            try:
                self.selfnet_path = setting["SelfNet"]["model_path"]
            except (KeyError, TypeError):
                self.selfnet_path = None

            # skip if not found in the file in case older setting file
            try:
                self.refinement_model_path = setting["Refinement"]["model_path"]
                self.refinement_Z = setting["Refinement"]["model_resZ"]
                self.refinement_XY = setting["Refinement"]["model_resXY"]
            except (KeyError, TypeError):
                self.refinement_model_path = None
                self.refinement_Z = None
                self.refinement_XY = None

            # Spurious-neck filter — drops the NECK when the AND-gate fires
            # (long path AND low nnU-Net support), but by default retains the
            # spine head. Default `neck_only` avoids creating long fake necks
            # in the CSV while still letting the head be evaluated by the
            # partial-spine classifier (head with no reachable neck → kept
            # with zeroed neck metrics, or dropped via keep_partial_spines).
            # Previous default was `both` (drop head+neck) but that was
            # asymmetric with orphan-head handling: a real spine whose neck
            # was pathfinder-synthesised lost its head entirely, while an
            # nnU-Net orphan head with no neck at all was silently kept.
            #
            # YAML section "SpuriousNeck" (optional — defaults below):
            #   exclude_mode: neck_only  # "both" | "neck_only" | "flag_only"
            #   nnunet_support_min: 0.2  # min fraction of neck voxels overlapping nnU-Net
            #   intensity_ratio_min: 0.4 # recorded in CSV but not part of the AND-gate
            #   path_length_min_um: 1.0  # only flag necks longer than this
            try:
                _sn = setting.get("SpuriousNeck", {}) or {}
            except AttributeError:
                _sn = {}
            # Backward compat: older configs used `exclude: bool`. Translate.
            if "exclude_mode" in _sn:
                _mode = str(_sn["exclude_mode"]).lower().strip()
            elif "exclude" in _sn:
                _mode = "neck_only" if _sn["exclude"] else "flag_only"
            else:
                _mode = "neck_only"
            if _mode not in {"both", "neck_only", "flag_only"}:
                _mode = "neck_only"
            self.spurious_exclude_mode = _mode
            self.spurious_nnunet_support_min = float(_sn.get("nnunet_support_min", 0.2))
            self.spurious_intensity_ratio_min = float(_sn.get("intensity_ratio_min", 0.4))
            self.spurious_path_length_min_um = float(_sn.get("path_length_min_um", 1.0))
            # Keep the spine head (in spines_filtered and detected_spines.csv)
            # even when its neck is flagged spurious. Default False → head is
            # removed along with neck (matches `exclude_mode=both` semantics).
            # When True, flagged spines have their neck voxels nulled but the
            # head + head-only measurements are retained — useful when users
            # want to review head morphology for spines with weak neck signal.
            self.spurious_keep_head_if_flagged = bool(_sn.get("keep_head_if_flagged", False))
            # Post-bridge cleanup: if a spine label's connected_necks has
            # multiple disconnected components after all bridges, keep only
            # the component touching (26-adjacent to) the spine head. Drops
            # rogue dendrite-side fragments that Pass 2 may have assigned
            # via proximity without a real corridor connection.
            self.drop_disconnected_neck_fragments = bool(_sn.get("drop_disconnected_neck_fragments", True))

            # Post-bridge cleanup: for spines with ≥2 neck CCs all touching the
            # head, keep only the CC whose min distance-to-dendrite is smallest.
            # Catches pathfinder-synthesized fragments going away from dendrite.
            # Default ON — biologically, real necks monotonically approach the
            # dendrite.
            self.drop_wrong_direction_neck_fragments = bool(_sn.get("drop_wrong_direction_neck_fragments", True))

            # Trim necks whose voxels touch another spine's head (26-conn).
            # Default OFF — truncating the neck leaves an inaccurate stub. The
            # preferred handling is to flag these as 'partial-spine' below.
            self.trim_necks_touching_other_spines = bool(_sn.get("trim_necks_touching_other_spines", False))

            # Spurious-neck intensity gate — legacy compat. When True, the
            # AND gate requires dim intensity too (length AND unsupported AND
            # dim). When False (default), the gate is (length AND unsupported)
            # only — which correctly catches pathfinder-synthesised bridges
            # that follow bright voxels (→ high intensity_ratio, evading the
            # legacy AND gate).
            self.spurious_require_intensity = bool(_sn.get("require_intensity", False))

            # Keep spines whose neck exists but does NOT reach the dendrite
            # (most commonly because a nearby spine occludes the path). Default
            # ON — the head is preserved with neck metrics zeroed and
            # spine_type='partial-spine'. Disable to drop these spines entirely.
            self.keep_partial_spines = bool(_sn.get("keep_partial_spines", True))
            # Distance threshold (voxels): a neck is considered NOT to reach
            # dendrite when min(dendrite_distance) over neck voxels exceeds
            # this value. Default 2 voxels ≈ 0.13 µm at 65 nm XY.
            self.partial_spine_dist_threshold_vox = float(
                _sn.get("partial_spine_dist_threshold_vox", 2.0))

            # Filopodia recovery: preserve nnU-Net neck CCs with no associated
            # spine head by carving a head from the tip of the orphan neck.
            # Default OFF — opt-in so existing benchmarks/reruns stay identical.
            # YAML: top-level `recover_filopodia: true` (or under SpuriousNeck
            # for legacy compat — both locations are checked).
            _rf = setting.get("recover_filopodia", None)
            if _rf is None:
                _rf = _sn.get("recover_filopodia", False)
            self.recover_filopodia = bool(_rf)

            # Dendrite repair: bridge dendrite fragments split by discontinuous
            # nnU-Net predictions (XY-only, gap-bounded). Default OFF so existing
            # benchmarks reproduce; turn on with YAML `dendrite_repair: true`
            # (legacy alias `intelligent_dendrite_repair` accepted for backward
            # compat) and optionally `dendrite_repair_max_dist_um: 2.0`.
            _dr = setting.get("dendrite_repair", None)
            if _dr is None:
                _dr = setting.get("intelligent_dendrite_repair", False)  # legacy alias
            self.dendrite_repair = bool(_dr)
            self.dendrite_repair_max_dist_um = float(
                setting.get("dendrite_repair_max_dist_um", 2.0))

            # Multi-head spine detection: group physically connected heads
            # under one parent ID (purely topological — no parameters). Default
            # OFF so existing benchmarks reproduce. Adds parent_spine_id,
            # head_index, multi_head_group_size columns to Detected_spines.csv,
            # plus a Multi_Head_Spine_Groups.csv aggregate file and a 5th
            # channel in the validation MIP.
            self.detect_multi_head_spines = bool(
                setting.get("detect_multi_head_spines", False))

            # Tiny spine-head backstop: drop any spine-head label with voxel
            # count <= `min_spine_head_voxels` BEFORE neck association, to stop
            # stray micro-labels from splitting real necks in Pass 1 of
            # associate_spines_with_necks_per_component. Neck mask is never
            # touched — only spines_filtered + matching spine_table rows.
            # Default 2: drops labels with 1 or 2 voxels (catches the observed
            # 1-voxel pathology plus 2-voxel micro-artifacts). Set to 1 to drop
            # only the strict 1-voxel case. Set to 0 (or negative) to disable.
            # The `neuron_spine_size[0]` µm³ filter in initial_spine_measurements
            # does the primary culling; this is a narrow backstop.
            _msh = setting.get("min_spine_head_voxels", None)
            if _msh is None:
                _msh = _sn.get("min_spine_head_voxels", 2)
            self.min_spine_head_voxels = int(_msh)

            # Wrong-direction neck fragment tolerance (µm). For each label
            # with ≥2 neck CCs, keep the CC whose min dist-to-dendrite is
            # closest; a CC is also kept if within this tolerance of the
            # winner (allows mildly branched necks while rejecting clearly
            # wrong-direction fragments like T2 spine 83's 11-vox stub).
            # Default 0.2 µm ≈ 3 voxels at 65 nm XY.
            _ndt = setting.get("neck_direction_tolerance_um", None)
            if _ndt is None:
                _ndt = _sn.get("neck_direction_tolerance_um", 0.2)
            self.neck_direction_tolerance_um = float(_ndt)

            self.Vaa3Dpath = setting["Vaa3D"]["path"]

    def inspect(self):
        for attr_name in dir(self):
            if not callable(getattr(self, attr_name)) and not attr_name.startswith("__"):
                value = getattr(self, attr_name)
                print(f"{attr_name}: {value}")

