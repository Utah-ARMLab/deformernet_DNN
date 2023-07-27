import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.append("../")


class ActionEmbeddingModel(nn.Module):

    def __init__(self):
        super(ActionEmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(3, 512)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.2)

        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        self.bn1 = nn.GroupNorm(1, 512)
        self.bn2 = nn.GroupNorm(1, 256)        

    def forward(self, action):
        # x = self.drop1(self.fc1(action))
        # act_embedding = self.drop2(self.fc2(x))        
        x = F.relu(self.bn1(self.fc1(action)))
        act_embedding = F.relu(self.bn2(self.fc2(x)))
        # x = self.drop1(F.relu(self.bn1(self.fc1(action))))
        # act_embedding = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return act_embedding
        #self.drop1(F.relu(self.bn1(self.fc1(x))))

class TransitionSDF(nn.Module):

    def __init__(self):
        super(TransitionSDF, self).__init__()
        self.action_embedding_model = ActionEmbeddingModel()
        self.fc1 = nn.Linear(512,512)        
        self.fc2 = nn.Linear(512,512)        
        self.fc3 = nn.Linear(512, 256)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(512)
        self.bn1 = nn.GroupNorm(1, 512)
        self.bn2 = nn.GroupNorm(1, 512)


    def forward(self, cloud_embedding, action):
        action_embedding = self.action_embedding_model(action)
        embedded_inputs = torch.cat((cloud_embedding, action_embedding),dim=-1)
        # print(embedded_inputs.shape)

        x = F.relu(self.bn1(self.fc1(embedded_inputs)))
        x = F.relu(self.bn2(self.fc2(x)))
        next_cloud_embedding = self.fc3(x)
        # next_cloud_embedding = F.relu(self.bn3(self.fc3(x)))

        return next_cloud_embedding


class PlaneSDF(nn.Module):

    def __init__(self):
        super(PlaneSDF, self).__init__()
        self.action_embedding_model = ActionEmbeddingModel()
        self.fc_p1 = nn.Linear(4,128)
        self.fc1 = nn.Linear(256,128)    
        self.fc2 = nn.Linear(256,128)  
        self.fc3 = nn.Linear(128,128)  
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,1)

        self.bn_p1 = nn.GroupNorm(1, 128)
        self.bn1 = nn.GroupNorm(1, 128)
        self.bn2 = nn.GroupNorm(1, 128)
        self.bn3 = nn.GroupNorm(1, 128)
        self.bn4 = nn.GroupNorm(1, 64)

    def forward(self, cloud_embedding, plane):
        # Get plane embedding:
        plane_embedding = F.relu(self.bn_p1(self.fc_p1(plane)))
        
        # concat two embeddings
        cloud_embedding = F.relu(self.bn1(self.fc1(cloud_embedding)))
        embedded_inputs = torch.cat((cloud_embedding, plane_embedding),dim=-1)

        x = F.relu(self.bn2(self.fc2(embedded_inputs)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        percent_passed = self.fc5(x)


        return percent_passed


if __name__ == '__main__':


    device = torch.device("cuda")    

    # model = TransitionSDF().to(device)
    # cloud_embedding = torch.randn((8,256)).to(device)
    # action = torch.randn((8,3)).to(device)
    # out = model(cloud_embedding, action)
    # print(out.shape)

    model = PlaneSDF().to(device)
    cloud_embedding = torch.randn((8,256)).to(device)
    plane = torch.randn((8,4)).to(device)
    out = model(cloud_embedding, plane)
    print(out.shape)    





