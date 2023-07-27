import pickle
from pointcloud_recon_2 import PointNetShapeServo
import torch
import numpy as np
from sklearn.metrics import mean_squared_error



with open("/home/baothach/shape_servo_data/data_for_training_end_to_end_new_task", 'rb') as f:
    data = pickle.load(f)

model = PointNetShapeServo(normal_channel=False)
model.load_state_dict(torch.load("/home/baothach/shape_servo_data/weights/PointNet/training_end_to_end_new_task(5)")) 
# params = list(model.parameters())
# print(params[-2])

# pc = torch.from_numpy(np.swapaxes(data["point clouds"][0],0,1)).float()
# output = []
# differences = []
# for i in range(len(data["point clouds"])):
# # for i in range(10):
#     model.eval()
#     pc_goal = torch.from_numpy(np.swapaxes(data["point clouds"][i],0,1)).float()
#     out = model(pc.unsqueeze(0), pc_goal.unsqueeze(0))
#     print(str(out[0][0].detach().numpy()) + "-----" + str(data["positions"][i]))
#     differences.append(out[0][0].detach().numpy() - data["positions"][i])
#     output.append(out[0][0].detach().numpy())


# print("====================================")
# print(differences)
# print(np.mean(differences))
# print(mean_squared_error(output[:], data["positions"][:]) )
# # print(mean_squared_error(output[50:], data["positions"][50:]) )
# model.eval()
with torch.no_grad():
    pc_goal = []
    pc = torch.from_numpy(np.swapaxes(data["point clouds"][0],0,1)).float().repeat(39,1,1)
    # print(pc.shape)
    # for i in range(len(data["point clouds"])-30):
    #     pc_goal.append(np.swapaxes(data["point clouds"][i],0,1))
    pc_goal = torch.tensor(data["point clouds"][:39])
    pc_goal = pc_goal.permute(0,2,1)
    # print(pc_goal.shape)

    outputs = model(pc, pc_goal)
    print(outputs.shape)
    # print(torch.tensor(data["positions"]).shape)
    print(outputs - torch.tensor(data["positions"][:39]).view(-1,1))
