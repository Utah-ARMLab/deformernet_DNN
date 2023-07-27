import sys
import os
import numpy as np
import timeit
from itertools import product

start_time = timeit.default_timer()

obj_category = "cylinder_1kPa"
batch_size = 180    #128

#### Train manipulation point
# os.chdir("./learn_mp")

# # os.system(f"python3 process_seg_data.py --obj_category {obj_category}")
# os.system(f"python3 process_classifier_data2.py --obj_category {obj_category}")

# os.system(f"python3 single_box_seg_trainer.py --obj_category {obj_category} --batch_size {batch_size}")
# os.system(f"python3 single_box_classifier_trainer.py --obj_category {obj_category} --batch_size {batch_size}")


#### Train DeformerNet
os.chdir("./rotation")

# os.system(f"python3 resample_point_cloud.py --obj_category {obj_category}")
# os.system(f"python3 add_mani_pt.py --obj_category {obj_category}")

# for option in [False, True]:
for option in [True]:    
    # os.system(f"python3 generalization_tasks/multi_hemis_trainer.py --use_mp_input {str(option)} \
    #             --obj_category {obj_category} --batch_size {batch_size}") # w/o MP

    # os.system(f"python3 single_box_trainer.py --use_mp_input {str(option)} --obj_category {obj_category} --batch_size {batch_size}") # w MP
    os.system(f"python3 rotation_trainer_modified_ratio.py --use_mp_input {str(option)} --obj_category {obj_category} --batch_size {batch_size}")


# ### Train bimanual

# prim_names = ["hemis"] #["box", "cylinder", "hemis"]
# stiffnesses = ["10k"]
# os.chdir("./bimanual")

# for (prim_name, stiffness) in list(product(prim_names, stiffnesses)):
    
    
#     obj_category = f"{prim_name}_{stiffness}Pa"
    
#     # if stiffness == "5k":
#     os.system(f"python3 process_data_bimanual.py --obj_category {obj_category}")
    
#     os.system(f"python3 bimanual_trainer.py --obj_category {obj_category} --batch_size {batch_size}")

# # os.chdir("../")
# # for option in [False, True]:  
# # # for option in [False]:    
# #     # os.system(f"python3 generalization_tasks/multi_hemis_trainer.py --use_mp_input {str(option)}")
# #     os.system(f"python3 rotation/single_box_trainer.py --use_mp_input {str(option)}")


# ### Train bimanual occlusion

# prim_names = ["box"] #["box", "cylinder", "hemis"]
# stiffnesses = ["10k"]
# os.chdir("./bimanual")

# for (prim_name, stiffness) in list(product(prim_names, stiffnesses)):
    
    
#     obj_category = f"{prim_name}_{stiffness}Pa"

#     # os.system(f"python3 process_data_bimanual.py --obj_category {obj_category}")   
#     os.system(f"python3 bimanual_trainer_modified.py --obj_category {obj_category} --batch_size {batch_size}")

# print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees" )
    

