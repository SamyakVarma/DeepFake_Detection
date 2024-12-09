import os
import PIL
import numpy as np
import cv2
from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=True)


pathIn = 'DeepFakeDetect\Celeb-DF-v2\Train\Real'
pathOut = 'DeepFakeDetect\Celeb-DF-v2\Cropped\Train\Real\\'

def resize_with_padding(image, target_size):
    target_width, target_height = target_size
    old_size = image.shape[:2]
    ratio = min(target_width / old_size[1], target_height / old_size[0])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))

    resized_image = cv2.resize(image, new_size)

    delta_w = target_width - new_size[0]
    delta_h = target_height - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return new_image

def load_Video(root,fileName, num_frames=16):
    vid = cv2.VideoCapture('DeepFakeDetect\Celeb-DF-v2\Train\Real\\Celeb_real_id5_0008.mp4')
    frame_cnt = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    interval = int(max(1, frame_cnt//num_frames))
    cnt = 0
    currFrame = 0
    width = 0
    height = 0
    writer = cv2.VideoWriter(pathOut+fileName,cv2.VideoWriter_fourcc(*"mp4v"),30,(224,224))
    frames = []
    while cnt <= num_frames and vid.isOpened():
        ret,frame = vid.read()
        if not ret:
            break
        if currFrame%interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes,_ = mtcnn.detect(frame)
            if boxes is not None:
                if cnt == 0:
                    width = boxes[0][3]-boxes[0][1]+10
                    height = boxes[0][2]-boxes[0][0]+10
                centerX = (boxes[0][1]+boxes[0][3])//2
                centerY = (boxes[0][0]+boxes[0][2])//2
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_cropped = frame[max(int(centerX-(width//2)),0):min(int(centerX+(width//2)),frame.shape[0]),max(int(centerY-(height//2)),0):min(int(centerY+(height//2)),frame.shape[1])]
                frame_cropped = resize_with_padding(frame_cropped,(224,224))
                cv2.imshow('Face Detection', frame_cropped)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                writer.write(frame_cropped)
                cnt+=1
        currFrame+=1
    vid.release()

    # frames = np.stack(frames, axis=0)#(H,W,C)->(T,H,W,C)
    # frames = np.transpose(frames,(0,3,1,2))#(T,H,W,C)->(T,C,H,W)
    # frames_tens = torch.from_numpy(frames).float()/255.0
    # return frames_tens
# i=0
# for fileName in os.listdir(pathIn):
#     if fileName.endswith('.mp4'):
#         load_Video(pathIn,fileName,8)
#         print('processed',i,'videos',fileName)
#         i+=1
# print('done')
load_Video(pathIn,'Celeb_real_id5_0008.mp4',8)
print('done')