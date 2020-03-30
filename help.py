import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models
import os
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy import ndimage
import time
import resnet
from Base3DModel import Base3DModel
from medicalnet_model import generate_model

train_dir = 'train'
test_dir = 'test'
train_data = 'train_pre_data.h5'
train_label = 'train_pre_label.csv'
testa_data = 'testa.h5'
testb_data = 'testb.h5'
checkpoint_pretrain_resnet_10_23dataset = 'resnet_10_23dataset.pth'

input_D = 79
input_H = 95
input_W = 79
num_seg_classes = 3
basemodel_3d_f = 8
print_every = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = h5py.File(os.path.join(train_dir,train_data),'r')
labels = pd.read_csv(os.path.join(train_dir,train_label))
features = np.array(train['data'])

temp_data_a = h5py.File(os.path.join(test_dir,testa_data),'r')['data']
temp_data_b = h5py.File(os.path.join(test_dir,testb_data),'r')['data']
temp_data = np.concatenate((np.array(temp_data_a),np.array(temp_data_b)))
#temp_label = np.zeros(shape=(temp_data.shape[0],))
len_temp_data_a = len(np.array(temp_data_a))

basemodel_3d_checkpoint_path = 'basemodel_3d_checkpoint.pth'
resnet50_2d_checkpoint_path = 'resnet50_2d_checkpoint.pth'
medicanet_3d_resnet10_checkpoint_path = 'medicanet_3d_resnet10_checkpoint.pth'

result_3d_basemodel = 'result_3D_basemodel.csv'
result_2d_resnet50 = 'result_2D_Resnet50.csv'
result_medicanet_3d_resnet10 = 'result_medicanet_3d_resnet10.csv'
result_medicanet_3d_resnet10_skfold_most_label = 'result_medicanet_3d_resnet10_skfold_most_label.csv'
result_medicanet_3d_resnet10_skfold_max_prop = 'result_medicanet_3d_resnet10_skfold_max_prop.csv'

criterion = nn.CrossEntropyLoss()

def save_checkpoint(epochs,optimizer,model,filepath):

    #if isinstance(model,nn.DataParallel):
    #    model = model.module
    checkpoint = {'epochs':epochs,
                  'optimizer_state_dict':optimizer.state_dict(),
                  'model_state_dict':model.state_dict()}

    torch.save(checkpoint,filepath)

def load_checkpoint(filepath,model_name,phase='train',device='cpu'):
    checkpoint = torch.load(filepath,map_location=device)
    if model_name == 'basemodel_3d':
        model = Base3DModel(num_seg_classes=num_seg_classes,f=basemodel_3d_f)
    elif model_name == 'medicanet_resnet3d_10':
        model, _ = generate_model(sample_input_W=input_W,
                            sample_input_H=input_H,
                            sample_input_D=input_D,
                            num_seg_classes=num_seg_classes,
                            phase=phase,
                            pretrain_path=checkpoint_pretrain_resnet_10_23dataset)
    elif model_name == 'resnet50_2d':
        model = models.resnet50(pretrained=True)
        with torch.no_grad():
            pretrained_conv1 = model.conv1.weight.clone()
            # Assign new conv layer with 79 input channels
            model.conv1 = torch.nn.Conv2d(79, 64, 7, 2, 3, bias=False)
            # Use same initialization as vanilla ResNet (Don't know if good idea)
            torch.nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
            # Re-assign pretraiend weights to first 3 channels
            # (assuming alpha channel is last in your input data)
            model.conv1.weight[:, :3] = pretrained_conv1
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_seg_classes))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def train_data(model,train_dataloaders,valid_dataloaders,epochs,optimizer,scheduler,criterion,checkpoint_path,device='cpu'):
    '''
        model:创建的模型
        train_dataloaders:训练集
        epochs:训练次数
        optimizer:优化器
        scheduler:学习率动态调整函数
        criterion:误差函数
        checkpoint_path:模型保存的地址
        device:训练使用的是cpu还是GPU
    
    '''
    start = time.time()
    model_indicators = pd.DataFrame(columns=['epoch','train_loss','train_acc','train_f1_score','val_loss','val_acc','val_f1_score'])
    steps = 0
    n_epochs_stop = 10
    min_val_f1_score = 0
    epochs_no_improve = 0
    
    model.to(device)
    
    for e in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_correct_sum = 0
        train_simple_cnt = 0
        train_f1_score = 0
        y_train_true = []
        y_train_pred = []
        for ii,(images,labels) in enumerate(train_dataloaders):
            steps += 1
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            
            outputs = model.forward(images)
            
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            #ps = torch.exp(outputs).data
            _,train_predicted = torch.max(outputs.data,1)
            train_correct_sum += (labels.data == train_predicted).sum().item()
            train_simple_cnt += labels.size(0)
            y_train_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())
            #equality = (labels.data == ps.max(1)[1])
            #equality = (labels.data == train_predicted)
            #train_acc += equality.type_as(torch.FloatTensor()).mean()
        
        scheduler.step()
        
        val_acc = 0
        val_correct_sum = 0
        val_simple_cnt = 0
        val_loss = 0 
        val_f1_score = 0
        y_val_true = []
        y_val_pred = []
        with torch.no_grad():
            model.eval()
            for ii,(images,labels) in enumerate(valid_dataloaders):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs,labels).item()
                
                _,val_predicted = torch.max(outputs.data,1)
                #equality = (labels.data == ps.max(1)[1])
                val_correct_sum += (labels.data == val_predicted).sum().item()
                val_simple_cnt += labels.size(0)
                #val_acc += equality.type_as(torch.FloatTensor()).mean()
                y_val_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
                y_val_pred.extend(np.ravel(np.squeeze(val_predicted.cpu().detach().numpy())).tolist())
        
        train_loss = train_loss/len(train_dataloaders)
        val_loss = val_loss/len(valid_dataloaders)
        train_acc = train_correct_sum/train_simple_cnt
        val_acc = val_correct_sum/val_simple_cnt
        train_f1_score = f1_score(y_train_true,y_train_pred,average='macro')
        val_f1_score = f1_score(y_val_true,y_val_pred,average='macro')
        print('Epochs: {}/{}...'.format(e+1,epochs),
              'Trian Loss:{:.3f}...'.format(train_loss),
              'Trian Accuracy:{:.3f}...'.format(train_acc),
              'Trian F1 Score:{:.3f}...'.format(train_f1_score),
              'Val Loss:{:.3f}...'.format(val_loss),
              'Val Accuracy:{:.3f}...'.format(val_acc),
              'Val F1 Score:{:.3f}'.format(val_f1_score))
        #scheduler.step(val_loss/len(valid_dataloaders))
        model_indicators.loc[model_indicators.shape[0]] = [e,train_loss,train_acc,train_f1_score,val_loss,val_acc,val_f1_score]
        #早期停止,根据模型训练过程中在验证集上的损失来保存表现最好的模型
        if val_f1_score > min_val_f1_score:
            save_checkpoint(e+1,optimizer,model,checkpoint_path)
            epochs_no_improve = 0
            min_val_f1_score = val_f1_score
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')

    plt_result(model_indicators)
    end = time.time()
    runing_time = end - start
    print('Training time is {:.0f}m {:.0f}s'.format(runing_time//60,runing_time%60))    

def check_accuracy_on_test(model,test_dataloaders,criterion,device='cpu'):
    correct = 0
    test_loss = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.to(device)
        model.eval()
        for data in test_dataloaders:
            images,labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            test_loss += criterion(outputs,labels).item()
            
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            y_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_pred.extend(np.ravel(np.squeeze(predicted.cpu().detach().numpy())).tolist())
            correct += (predicted == labels).sum().item()

    print('Test Loss:{:.3f}...'.format(test_loss/len(test_dataloaders)),
        'Test Accuracy:{:.3f}...'.format(correct/total),
        'Test F1_Score:{}'.format(f1_score(y_true,y_pred,average='macro')))

def plt_result(dataframe):
    fig = plt.figure(figsize=(16,5))
    
    fig.add_subplot(1, 3, 1)
    plt.plot(dataframe['epoch'], dataframe['train_loss'], 'bo', label='Train loss')
    plt.plot(dataframe['epoch'], dataframe['val_loss'], 'b', label='Val loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    fig.add_subplot(1, 3, 2)
    plt.plot(dataframe['epoch'], dataframe['train_acc'], 'bo', label='Train Accuracy')
    plt.plot(dataframe['epoch'], dataframe['val_acc'], 'b', label='Val Accuracy')
    plt.title('Training and validation Accuracy')
    plt.legend()
    
    fig.add_subplot(1, 3, 3)
    plt.plot(dataframe['epoch'], dataframe['train_f1_score'], 'bo', label='Train F1 Score')
    plt.plot(dataframe['epoch'], dataframe['val_f1_score'], 'b', label='Val F1 Score')
    plt.title('Training and validation F1 Score')
    plt.legend()

    plt.show()

def one_predict(data, loadmodel,topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    pixels = data[data > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (data - mean)/std
    out_random = np.random.normal(0, 1, size = data.shape)
    out[data == 0] = out_random[data == 0]
    out_tensor = torch.tensor(out)
    out_tensor = out_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        out_tensor = out_tensor.cuda()
        loadmodel.cuda()
    else:
        out_tensor = out_tensor.cpu()
        loadmodel.cpu()

    # 将输入变为变量
    out_tensor = Variable(out_tensor)

    torch.no_grad()
    output = loadmodel(out_tensor)
    output = torch.nn.functional.softmax(output, dim=1)[0]
    probs,indexs = output.topk(topk)

    #tensor转为numpy，并降维
    probs = np.squeeze(probs.cpu().detach().numpy())
    indexs = np.squeeze(indexs.cpu().detach().numpy())

    return probs,indexs

def all_predict(test_dataloader,loadmodel,device,result_path):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    start = time.time()
    result_df = pd.DataFrame(columns=['testa_id','label'])
    
    with torch.no_grad():
        loadmodel.to(device)
        loadmodel.eval()
        for ii,image in enumerate(test_dataloader):
            image = image.to(device)
            output = loadmodel(image)
            _,indexs = torch.max(output.data,1)
            indexs = np.squeeze(indexs.cpu().detach().numpy()).tolist()
            if ii < len_temp_data_a:
                result_df.loc[result_df.shape[0]] = [('testa_{}'.format(ii)),indexs]
            else:
                result_df.loc[result_df.shape[0]] = [('testb_{}'.format(ii - len_temp_data_a)),indexs]
    
            if ii%20==0:
                print('{} test data have been predicted'.format(ii))
                print('--'*20)
    result_df.to_csv(result_path,index=False)
    end = time.time()
    runing_time = end - start
    print('Test time is {:.0f}m {:.0f}s'.format(runing_time//60,runing_time%60))

def all_predict_with_softmax(test_dataloader,loadmodel,device,result_path,topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    start = time.time()
    result_df = pd.DataFrame(columns=['testa_id','label_topk0','proportion_topk0','label_topk1','proportion_topk1','label_topk2','proportion_topk2'])
    
    with torch.no_grad():
        loadmodel.to(device)
        loadmodel.eval()
        for ii,image in enumerate(test_dataloader):
            image = image.to(device)
            output = loadmodel(image)

            output = torch.nn.functional.softmax(output, dim=1)[0]
            probs,indexs = output.topk(topk)
            indexs = np.squeeze(indexs.cpu().detach().numpy())
            probs = np.squeeze(probs.cpu().detach().numpy())

            if ii < len_temp_data_a:
                result_df.loc[result_df.shape[0]] = [('testa_{}'.format(ii)),indexs[0],probs[0],indexs[1],probs[1],indexs[2],probs[2]]
            else:
                result_df.loc[result_df.shape[0]] = [('testb_{}'.format(ii - len_temp_data_a)),indexs[0],probs[0],indexs[1],probs[1],indexs[2],probs[2]]
    
            if ii%20==0:
                print('{} test data have been predicted'.format(ii))
                print('--'*20)
    result_df.to_csv(result_path,index=False)
    end = time.time()
    runing_time = end - start
    print('Test time is {:.0f}m {:.0f}s'.format(runing_time//60,runing_time%60))