import cv2
import torch
import numpy as np
from torchvision import models,transforms
from torch.utils.data import DataLoader,Dataset,random_split
import os
from tqdm import tqdm
import shutil

def save_ckp(state, checkpoint_dir):
    f_path = f"{checkpoint_dir}\\checkpoint.pt"
    torch.save(state, f_path)

def load_Video(path, num_frames=16):
    vid = cv2.VideoCapture(path)
    frames = []
    # frame_cnt = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    # interval = max(1, frame_cnt//num_frames)
    cnt = 0
    while cnt<8 and vid.isOpened():
        ret,frame = vid.read()
        if not ret:
            break
    #     if cnt%interval == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        cnt+=1
    vid.release()
    if(cnt!=8):
        while cnt<8:
            frames.append(np.zeros((224,224,3)))
            cnt+=1

    frames = np.stack(frames, axis=0)#(H,W,C)->(T,H,W,C)
    frames = np.transpose(frames,(0,3,1,2))#(T,H,W,C)->(T,C,H,W)
    frames_tens = torch.from_numpy(frames).float()/255.0
    return frames_tens


class videoDataset(Dataset):
    def __init__(self,root_dir,transform = None, num_frames=16):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.classes = ['Fake','Real']
        self.data = []
        for class_idx,class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir,class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith('.mp4'):
                    file_path = os.path.join(class_path,file_name)
                    self.data.append((file_path,class_idx))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        video_frames = load_Video(video_path,self.num_frames)
        if self.transform:
            video_frames = torch.stack([self.transform(frame) for frame in video_frames])
        label = torch.tensor(label,dtype=torch.float32)
        # print(video_frames.shape,label.shape,"FRAMES")
        return video_frames,label


def split_Channel(T_in):
    C = T_in.shape[1]
    # print('Before split:',T_in.shape)
    X1,X2 = torch.split(T_in, C//2,dim = 1)
    # print('After split:',X1.shape,X2.shape)
    return X1,X2

class SIM(torch.nn.Module):
    def __init__(self, C_in,device):
        super(SIM, self).__init__()
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2).to(device)
        self.conv1 = torch.nn.Conv2d(C_in, C_in, kernel_size=(1,3), padding=(0,1)).to(device)
        self.conv2 = torch.nn.Conv2d(C_in, C_in, kernel_size=(3,1), padding=(1,0)).to(device)
        self.upSample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).to(device)
        self.conv3 = torch.nn.Conv2d(C_in, C_in, kernel_size=3, padding=1).to(device)
        self.conv4 = torch.nn.Conv2d(C_in, C_in, kernel_size=3, padding=1).to(device)
        self.device = device

    def forward(self, X1):
        pool_out = self.avg_pool(X1)
        K1_out = self.conv1(pool_out)
        K2_out = self.conv2(K1_out)
        S = self.upSample(K2_out)
        if X1.shape[2] % 2!=0:
            reshaper = torch.nn.Upsample(X1.shape[2],mode='bilinear',align_corners=False).to(self.device)
            S = reshaper(S)
        S = S + X1
        sig_S = torch.sigmoid(S).to(self.device)
        K3_out = self.conv3(X1)
        Y1 = sig_S * K3_out
        Y1 = self.conv4(Y1)
        Y1 = Y1.permute(0,2,3,1).contiguous()
        # print('SIM-OUT',Y1.shape)
        return Y1
    
class ISM(torch.nn.Module):
    def __init__(self,C_in,device):
        super(ISM, self).__init__()
        self.conv1 = torch.nn.Conv1d(1,1,kernel_size=3,padding = 1).to(device)
        self.device = device
    def forward(self,Y1):
        # print('ISM-IN',Y1.shape)
        Y1 = Y1.permute(0,3,1,2)
        Y1_ = torch.nn.functional.adaptive_avg_pool2d(Y1,(1,1)).to(self.device)
        T,C_2,H,W = Y1_.shape
        Y1_ = Y1_.reshape(T,C_2,H*W)
        Y1_ = Y1_.permute(0,2,1).contiguous()
        Y1_ = self.conv1(Y1_)
        Y1_ = Y1_.permute(0,2,1).contiguous()
        Y1_ = Y1_.reshape(T,C_2,H,W)
        Y1_ = torch.sigmoid(Y1_).to(self.device)
        Y1_ = Y1_*Y1
        # print('ISM-OUT',Y1_.shape)
        return Y1_
    
class TIM(torch.nn.Module):
    def __init__(self, C_in,device, r = 16):
        super(TIM, self).__init__()
        self.C_in = C_in
        self.r = r
        CC = C_in//r
        self.compressor = torch.nn.Conv2d(C_in,CC,kernel_size=(1,1)).to(device)
        self.HConv1 = torch.nn.Conv2d(CC,CC,kernel_size=(3,1),padding=(1,0)).to(device)
        self.HConv2 = torch.nn.Conv2d(CC,CC,kernel_size=(3,1),padding=(1,0)).to(device)
        self.HAvg_pool = torch.nn.AvgPool2d(kernel_size=2,stride=2).to(device)
        self.HConv3 = torch.nn.Conv2d(CC,CC,kernel_size=(3,1),padding=(1,0)).to(device)
        self.HUp_sample = torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False).to(device)
        self.HConv4 = torch.nn.Conv2d(CC,self.C_in,kernel_size=1).to(device)

        self.WConv1 = torch.nn.Conv2d(CC,CC,kernel_size=(1,3),padding=(0,1)).to(device)
        self.WConv2 = torch.nn.Conv2d(CC,CC,kernel_size=(1,3),padding=(0,1)).to(device)
        self.WAvg_pool = torch.nn.AvgPool2d(kernel_size=2,stride=2).to(device)
        self.WConv3 = torch.nn.Conv2d(CC,CC,kernel_size=(1,3),padding=(0,1)).to(device)
        self.WUp_sample = torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False).to(device)
        self.WConv4 = torch.nn.Conv2d(CC,self.C_in,kernel_size=1).to(device)
        self.device = device
    
    def forward(self,X2):
        # print('TIM-IN',X2.shape)
        T, C, H, W = X2.shape
        CC = C//self.r
        X2_h = self.compressor(X2).permute(3,1,2,0)
        s_h = []
        for t in range(T-1):#TODO: T or T-1 ?
            st_h = self.HConv1(X2_h[:, :, :,t+1:t+2]) - X2_h[:,:,:,t:t+1]
            s_h.append(st_h)
        s_h.append(torch.zeros_like(s_h[0]).to(self.device))
        s_h = torch.cat(s_h,dim=-1)
        Hbranch1= self.HConv2(s_h)
        reshaped = s_h
        if s_h.shape[2] % 2!=0:
            reshaped = torch.nn.functional.pad(s_h,(0,0,0,1,0,0)).to(self.device)
        Hbranch2 = self.HAvg_pool(reshaped)
        Hbranch2 = self.HConv3(Hbranch2)
        Hbranch2 = self.HUp_sample(Hbranch2)
        if s_h.shape[2] % 2!=0:
            Hbranch2 = Hbranch2[:,:,:s_h.shape[2],:]
        Hbranch3 = s_h
        
        h_out = Hbranch1 + Hbranch2 + Hbranch3
        h_out = self.HConv4(h_out)
        W, C_2, H, T = h_out.shape
        h_out = h_out.permute(1,3,2,0).contiguous()
        h_out = h_out.view(C_2, T, H*W)
        h_out = torch.sigmoid(h_out).to(self.device)

        X2_w = self.compressor(X2).permute(2,1,0,3)
        s_w = []
        for t in range(T-1):#TODO: T or T-1 ?
            st_w = self.WConv1(X2_w[:, :, t+1:t+2,:]) - X2_w[:,:,t:t+1,:]
            s_w.append(st_w)
        s_w.append(torch.zeros_like(s_w[0]).to(self.device))
        s_w = torch.cat(s_w,dim=-2)
        Wbranch1= self.WConv2(s_w)
        reshapedW = s_w
        if s_w.shape[3] % 2!=0:
            reshapedW = torch.nn.functional.pad(s_w,(0,1,0,0,0,0)).to(self.device)
        Wbranch2 = self.WAvg_pool(reshapedW)
        Wbranch2 = self.WConv3(Wbranch2)
        Wbranch2 = self.WUp_sample(Wbranch2)
        if s_w.shape[3] % 2!=0:
            Wbranch2 = Wbranch2[:,:,:,:s_w.shape[3]]
        Wbranch3 = s_w
        w_out = Wbranch1 + Wbranch2 + Wbranch3
        w_out = self.WConv4(w_out)
        H, C_2, T, W = w_out.shape
        w_out = w_out.permute(1,2,0,3).contiguous()
        w_out = w_out.view(C_2, T, H*W)
        w_out = torch.sigmoid(w_out).to(self.device)
        Y2 = ((h_out + w_out)/2)*(X2.permute(1,0,2,3).reshape(C_2,T,H*W))
        Y2 = Y2.permute(1,0,2)
        Y2 = Y2.view(T,C,H,W).contiguous()
        # print('TIM-OUT',Y2.shape)
        return Y2

class STIL(torch.nn.Module):
    def __init__(self, in_channels,out_channels,stride,device):
        super(STIL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.device = device
    def forward(self,X):
        # print('STIL-IN',X.shape)
        T,C,H,W =X.shape
        X1,X2 = split_Channel(X)
        T,C_1,H,W = X1.shape
        SIM_BLOCK = SIM(C_1,self.device)
        T,C_2,H,W = X2.shape
        ISM_BLOCK = ISM(C_1,self.device)
        TIM_BLOCK = TIM(C_2,self.device)
        Y1 = SIM_BLOCK(X1)
        Y1_= ISM_BLOCK(Y1)
        Y2 = TIM_BLOCK(X2)
        Y = Y1_+Y2
        conv2d = torch.nn.Conv2d(C//2,C//2,kernel_size=(3,3),padding=1).to(self.device)
        conv2d_out = conv2d(Y)
        Y1 = Y1.permute(0,3,1,2).contiguous()
        # X = X.reshape(T,C,H*W)
        # print(Y.shape,conv2d_out.shape,Y1.shape)
        Y_cat = torch.cat((Y1,conv2d_out),dim=1)
        out = X + Y_cat
        reshaped = torch.nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=(3,3),stride=self.stride,padding=1).to(self.device)
        out = reshaped(out)
        # print('STIL-OUT',out.shape)
        return out
        
class ResNet50_STIL(torch.nn.Module):
    def __init__(self, device, num_classes=1):
        super(ResNet50_STIL, self).__init__()
        resnet50 = models.resnet50(pretrained=True).to(device)
        self.device = device
        self.stem = torch.nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool
        )

        self.layer1 = self._replace_with_stil(resnet50.layer1)
        self.layer2 = self._replace_with_stil(resnet50.layer2)
        self.layer3 = self._replace_with_stil(resnet50.layer3)
        self.layer4 = self._replace_with_stil(resnet50.layer4)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * 4, num_classes)

    def _replace_with_stil(self,layer):
        for name,module in layer.named_children():
            if isinstance(module,models.resnet.Bottleneck):#TODO: could try changing layer channel sizes manually instead of reshaping?
                in_channels = module.conv2.in_channels
                out_channels = module.conv2.out_channels
                stride = module.conv2.stride
                module.conv2 = STIL(in_channels,out_channels,stride,self.device)
        return layer
    
    def forward(self, X):
        X_ = torch.empty(0,8,1).to(self.device)
        # x=x.view(16,3,224,224).contiguous()
        for i in range(X.shape[0]):
            x = X[i]
            x = self.stem(x)
            # print('_______________________Layer1_______________________')
            x = self.layer1(x)
            # print('_______________________Layer2_______________________')
            x = self.layer2(x)
            # print('_______________________Layer3_______________________')
            x = self.layer3(x)
            # print('_______________________Layer4_______________________')
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            X_ = torch.cat((X_, x.unsqueeze(0)), dim=0)
        # print('________________________________YAY_________________________________')
        return X_
    
def pad_collate(batch):
    # print('Pad_Collated')    
    target_frames = 8
    target_channels = 3
    target_height = 224
    target_width = 224
    
    padded_videos = []
    labels = []
    
    for frames, label in batch:
        num_frames = frames.size(0)        
        if num_frames < target_frames:
            pad_size = target_frames - num_frames
            padded_video = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, pad_size), "constant", 0)
        elif num_frames > target_frames:
            padded_video = frames[:target_frames]
        else:
            padded_video = frames
        resized_video = torch.stack([transforms.functional.resize(frame, (target_height, target_width)) for frame in padded_video])
        # print(f"Original frames: {frames.size()}, Processed frames: {padded_video.size()}")
        if frames.size(1) != target_channels:
            if frames.size(1) == 4:
                frames = frames[:, :3, :, :]
            else:
                raise ValueError(f"Expected 3 channels but got {frames.size(1)} channels.")
        padded_videos.append(resized_video)
        labels.append(label)
    return torch.stack(padded_videos), torch.tensor(labels)

def load_VideoTest(path, num_frames=16):
    vid = cv2.VideoCapture(path)
    frames = []
    # frame_cnt = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    # interval = max(1, frame_cnt//num_frames)
    cnt = 0
    while cnt<16 and vid.isOpened():
        ret,frame = vid.read()
        if not ret:
            break
        # if cnt%interval == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        cnt+=1
    vid.release()
    if(cnt!=16):
        while cnt<16:
            frames.append(np.zeros((224,224,3)))
            cnt+=1

    frames = np.stack(frames, axis=0)#(H,W,C)->(T,H,W,C)
    frames = np.transpose(frames,(0,3,1,2))#(T,H,W,C)->(T,C,H,W)
    frames_tens = torch.from_numpy(frames).float()/255.0
    # print(frames_tens.shape)
    return frames_tens

class videoDatasetTest(Dataset):
    def __init__(self,root_dir,transform = None, num_frames=16):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.classes = ['Fake','Real']
        self.data = []
        for file_name in os.listdir(self.root_dir):
            if file_name.endswith('.mp4'):
                file_path = os.path.join(self.root_dir,file_name)
                class_idx = 0
                if 'real' in file_name:
                    class_idx = 1
                self.data.append((file_path,class_idx))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        video_frames = load_VideoTest(video_path,self.num_frames)
        if self.transform:
            video_frames = torch.stack([self.transform(frame) for frame in video_frames])
        label = torch.tensor(label,dtype=torch.float32)
        # print(video_frames.shape,label.shape,"FRAMES")
        return video_frames,label
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50_STIL(device=device, num_classes=1).to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    criterion = torch.nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
    ])

    train_dataset = videoDataset(root_dir='DeepFakeDetect\Celeb-DF-v2\Cropped\Train', transform=data_transforms,num_frames=8)
    train_size = int(0.8*len(train_dataset))
    val_size = len(train_dataset)- train_size

    train_dataset,val_dataset = random_split(train_dataset,[train_size,val_size])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,collate_fn=pad_collate, num_workers=4)
    validation_loader = DataLoader(val_dataset, batch_size=2, shuffle=True,collate_fn=pad_collate, num_workers=4)
    num_epochs = 52
    epochBar = tqdm(total=num_epochs,leave=False)

    

    checkpoint = torch.load("DeepFakeDetect\CKPTS\checkpoint.pt")
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochBar.update(epoch)
    for epoch in range(epoch+1,num_epochs):
        epochBar.update(1)
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        loop = tqdm(enumerate(train_loader),total=len(train_loader),leave = False)
        for batch_idx,(inputs,labels) in loop:
            inputs = inputs.to(device)
            # print('IN--------------',inputs.shape)
            labels = labels.to(device).unsqueeze(1)

            # optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.mean(dim = 1)
            # print('OUT:',outputs.shape,"Labels",labels.shape,'\n',outputs,labels)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_description(f"Epoch[{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

            preds = torch.sigmoid(outputs).round()
            correct_preds+=(preds==labels).sum().item()
            total_preds += labels.size(0)
        epoch_loss = running_loss/len(train_loader)
        train_accuracy = correct_preds/total_preds

        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        loopVal = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        with torch.no_grad():
            for batch_idx,(inputs,labels) in loopVal:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(inputs)
                outputs = outputs.mean(dim = 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                loopVal.set_description(f"Epoch[{epoch}/{num_epochs}]")
                loopVal.set_postfix(loss=loss.item())
                preds = torch.sigmoid(outputs).round()
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)
        val_loss /= len(validation_loader)
        val_accuracy = correct_preds / total_preds
        scheduler.step()
        # print(f'Epoch {epoch+1}/{num_epochs}')
        epochBar.set_description(f'Epoch {epoch+1}/{num_epochs}')
        epochBar.set_postfix(TrainLoss=epoch_loss, TrainAcc= train_accuracy,valLoss=val_loss,valAcc=val_accuracy)
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
        # save_ckp(checkpoint, 'DeepFakeDetect\\CKPTS')
    # epochBar.close()

    # torch.save(model.state_dict(), 'DeepFakeDetect\\runs\\resnet50_stil.pth')


#-----------------------------TEST--------------------------------------
    # test_dataset = videoDatasetTest(root_dir='DeepFakeDetect\Celeb-DF-v2\Test', num_frames=16)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,collate_fn=pad_collate, num_workers=4)
    # model = ResNet50_STIL(device,num_classes=1).cuda()


    # model.load_state_dict(torch.load('DeepFakeDetect\\runs\\resnet50_stil.pth'))

    # model.eval()
    # test_loss = 0.0
    # correct_preds = 0
    # total_preds = 0
    # i = 0
    # testBar = tqdm(enumerate(test_loader),total=len(test_loader),leave=False)
    # with torch.no_grad():
    #     for inputs, labels in testBar:
    #         inputs = inputs.cuda()
    #         labels = labels.cuda().unsqueeze(1)

    #         outputs = model(inputs)
    #         outputs = outputs.mean(dim = 1)
    #         loss = criterion(outputs, labels)
    #         test_loss += loss.item()
    #         testBar.set_description(f"Completed[{i}/{len(test_loader)}]")
    #         i+=1
    #         preds = torch.sigmoid(outputs).round()
    #         correct_preds += (preds == labels).sum().item()
    #         total_preds += labels.size(0)

    # test_loss /= len(test_loader)
    # test_accuracy = correct_preds / total_preds

    # print(f'Test Loss: {test_loss:.4f}')
    # print(f'Test Accuracy: {test_accuracy:.4f}',correct_preds,total_preds)

if __name__ == '__main__':
    main()
