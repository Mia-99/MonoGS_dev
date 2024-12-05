
import os
import glob
import torch



def run_slam_configs(configs_dict):
    for idx, configs in configs_dict.items():
        print(f"\nRunning Office {idx}")
        print(f"=======================================================================================")
        for config_file_path in configs:
            torch.cuda.empty_cache()
            # command = f"python slam.py --config {config_file_path} --eval"
            command = f"python slam.py --config {config_file_path} --disable_gui"
            print(f"Running: {command}")
            os.system(command)



if __name__ == "__main__":

    office_configs = {}
    image_info="1200x680_f600"

    if False:
        # only selfcalibration data
        office_configs[0] =  glob.glob(f"configs/mono/replica/office0_{image_info}_*.yaml")
        office_configs[1] =  glob.glob(f"configs/mono/replica/office1_{image_info}_*.yaml")
        office_configs[2] =  glob.glob(f"configs/mono/replica/office2_{image_info}_*.yaml")
        office_configs[3] =  glob.glob(f"configs/mono/replica/office3_{image_info}_*.yaml")
        office_configs[4] =  glob.glob(f"configs/mono/replica/office4_{image_info}_*.yaml")

    else:
        # all data
        office_configs[0] =  glob.glob(f"configs/mono/replica/office0_{image_info}*.yaml")
        office_configs[1] =  glob.glob(f"configs/mono/replica/office1_{image_info}*.yaml")
        office_configs[2] =  glob.glob(f"configs/mono/replica/office2_{image_info}*.yaml")
        office_configs[3] =  glob.glob(f"configs/mono/replica/office3_{image_info}*.yaml")
        office_configs[4] =  glob.glob(f"configs/mono/replica/office4_{image_info}*.yaml")

    run_slam_configs(office_configs)
        

