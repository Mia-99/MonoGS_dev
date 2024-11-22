
import os
import glob


office_configs = {}

office_configs[0] =  glob.glob("configs/mono/replica/office0_1200x680*.yaml")
office_configs[1] =  glob.glob("configs/mono/replica/office1_1200x680*.yaml")
office_configs[2] =  glob.glob("configs/mono/replica/office2_1200x680*.yaml")
office_configs[3] =  glob.glob("configs/mono/replica/office3_1200x680*.yaml")
office_configs[4] =  glob.glob("configs/mono/replica/office4_1200x680*.yaml")

for idx, configs in office_configs.items():
    print(f"\nRunning Office {idx}")
    print(f"=======================================================================================")
    for config_file_path in configs:
        command = f"python slam.py --config {config_file_path} --eval"                      
        print(f"Running: {command}")
        os.system(command)
    

