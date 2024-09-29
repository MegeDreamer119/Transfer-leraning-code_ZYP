
# The code is written by Yipeng Zhang
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import shap
import matplotlib
import math
import time
import numpy
from functools import partial

matplotlib.rcParams['font.family'] = 'Times New Roman'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 25
set_seed(seed)

numpy.random.seed(42)
random.seed(42)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        # 输出层
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.out(x)
        return x

input_size=15
hidden_size=64
output_size=1
dropout_rate=0.01
# 初始化模型
model = NeuralNetwork(input_size, hidden_size, output_size,dropout_rate)


model.load_state_dict(torch.load('C:\\Users\\zhang\\Desktop\\Academic papers\\model.pth'))

input_size = 8
hidden_size = 64
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size, dropout_rate)
# 冻结前三层的参数
for layer in [model.fc1, model.fc2]:
    for param in layer.parameters():
        param.requires_grad = False

original_lr = 0.01

adjusted_lr = original_lr /10

optimizer = torch.optim.Adam(
    [param for param in model.parameters() if param.requires_grad], 
    lr=adjusted_lr
)

new_data_path = r""
new_df = pd.read_excel(new_data_path)
new_X = new_df.iloc[:, :-1].values
new_y = new_df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.2, random_state=25)

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(new_X)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(new_X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, new_y, test_size=0.2, random_state=25)# 分割新的数据集为训练集和测试集

new_X_tensor=torch.FloatTensor(new_X)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

df.to_excel(save_path, index=False, engine='openpyxl')

print(f'文件已保存到 {save_path}')

train_losses = []
num_epochs = 1200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = F.mse_loss(outputs.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    r2 = r2_score(y_test_tensor.numpy(), predictions.numpy())
    rmse = np.sqrt(mean_squared_error(y_test_tensor.numpy(), predictions.numpy()))

print(f'R² score: {r2}')
print(f'RMSE: {rmse}')

background = torch.FloatTensor(X_scaled)

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(background)
X_original = scaler.inverse_transform(X_scaled)
feature_names = list(new_df.columns[:-1])
# 指定您感兴趣的特征名称
interested_features = [
    'Nafion content', 'Solid content', 'Ethanol content', 'Isopropanol content',
    'Propylene glycol content', 'Glycerol content', 'Water content'
]
interested_feature_indices = [feature_names.index(feat) for feat in interested_features]
# 逆标准化测试数据
X_test_original = scaler.inverse_transform(X_test_tensor.numpy())
X_trian_original = scaler.inverse_transform(X_train_tensor.numpy())
# 绘制总体特征重要性
shap.summary_plot(shap_values[:, interested_feature_indices], X_original[:, interested_feature_indices], feature_names=interested_features)

for feature_name in interested_features:
    feature_index = feature_names.index(feature_name)
    shap.dependence_plot(feature_index, shap_values, X_original, feature_names=feature_names, interaction_index=None)

df_shap = pd.DataFrame(X_original, columns=feature_names)
df_shap_values = pd.DataFrame(shap_values, columns=[f"{name}_shap" for name in feature_names])

df_combined = pd.concat([df_shap, df_shap_values], axis=1)

output_file_path = r"C:\Users\zhang\Desktop\新建文件夹\SHAP.xlsx"

df_combined.to_excel(output_file_path, index=False)

print(f"SHAP values and feature values have been exported to {output_file_path}")

model.eval()
X_scaled = scaler.transform(new_X)
X_tensor = torch.FloatTensor(X_scaled).requires_grad_(True)
explainer = shap.GradientExplainer(model, X_tensor)

shap_values = explainer.shap_values(X_tensor)

print(np.array(shap_values).shape)
sample_index=251

explainer = shap.GradientExplainer(model, X_tensor)

sample = X_tensor[sample_index].unsqueeze(0)
shap_values = explainer.shap_values(sample)

interested_indices = [feature_names.index(feat) for feat in interested_features]
shap_values_interested = shap_values[0][interested_indices]
sample_features_interested = sample.detach().cpu().numpy()[0, interested_indices]
X_tensor = torch.FloatTensor(X_scaled)

with torch.no_grad():
    expected_value = model(X_tensor).mean().cpu().numpy()

expected_value = np.array([expected_value])

explanation = shap.Explanation(
    values=shap_values_interested,
    base_values=expected_value,
    data = np.array([0.003, 0.001, 1, 0, 0, 0, 0]),
    feature_names=interested_features
)

shap.plots.waterfall(explanation, max_display=len(interested_features))

class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0

def inverse_transform_features(scaler, standardized_features):
    return scaler.inverse_transform([standardized_features])[0]

def adjust_and_restandardize_features(features, scaler):
    features_with_placeholder = np.append(features, 0)

    original_features = scaler.inverse_transform([features_with_placeholder])[0]

    original_features = original_features[:-1]

    sum_of_selected_features = sum(original_features[-6:-1])  # 选择倒数第二到倒数第五个特征

    difference = 1 - sum_of_selected_features

    adjustment = difference / 5
    for i in range(-6, -1):
        original_features[i] += adjustment

    adjusted_standardized_features = scaler.transform([np.append(original_features, 0)])[0]

    adjusted_standardized_features = adjusted_standardized_features[:-1]
    
    return adjusted_standardized_features

def objective_function_adjusted(features, scaler):
    adjusted_features = adjust_and_restandardize_features(features, scaler)

    current_density_value = 2.028
    features_with_cd = np.insert(adjusted_features, len(adjusted_features), current_density_value)
    input_tensor = torch.FloatTensor(features_with_cd).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        predicted_voltage = model(input_tensor).item()
    return predicted_voltage

def HHO(objf,lb,ub,dim,SearchAgents_no,Max_iter):

        
    numpy.random.seed(18)
    random.seed(18)

    Rabbit_Location=numpy.zeros(dim)
    Rabbit_Energy=float("inf")

    X=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb

    convergence_curve=numpy.zeros(Max_iter)
    
    
    ############################
    s=solution()

    print("HHO is now tackling  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    
    t=0  # Loop counter
    
    # Main loop
    while t<Max_iter:
        for i in range(0,SearchAgents_no):
            
            # Check boundries
                      
            X[i,:]=numpy.clip(X[i,:], lb, ub)
            
            # fitness of locations
            fitness=objf(X[i,:])
            
            # Update the location of Rabbit
            if fitness<Rabbit_Energy: # Change this to > for maximization problem
                Rabbit_Energy=fitness 
                Rabbit_Location=X[i,:].copy() 
            
        E1=2*(1-(t/Max_iter)) # factor to show the decreaing energy of rabbit    
        
        # Update the location of Harris' hawks 
        for i in range(0,SearchAgents_no):

            E0=2*random.random()-1;  # -1<E0<1
            Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy)>=1:
                #Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no*random.random())
                X_rand = X[rand_Hawk_index, :]
                if q<0.5:
                    # perch based on other family members
                    X[i,:]=X_rand-random.random()*abs(X_rand-2*random.random()*X[i,:])

                elif q>=0.5:
                    #perch on a random tall tree (random site inside group's home range)
                    X[i,:]=(Rabbit_Location - X.mean(0))-random.random()*((ub-lb)*random.random()+lb)

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy)<1:
                #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                #phase 1: ----- surprise pounce (seven kills) ----------
                #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r=random.random() # probablity of each event
                
                if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                    X[i,:]=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X[i,:])

                if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                    Jump_strength=2*(1- random.random()); # random jump strength of the rabbit
                    X[i,:]=(Rabbit_Location-X[i,:])-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])
                
                #phase 2: --------performing team rapid dives (leapfrog movements)----------

                if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                    #rabbit try to escape by many zigzag deceptive motions
                    Jump_strength=2*(1-random.random())
                    X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:]);

                    if objf(X1)< fitness: # improved move?
                        X[i,:] = X1.copy()
                    else: # hawks perform levy-based short rapid dives around the rabbit
                        X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                        if objf(X2)< fitness:
                            X[i,:] = X2.copy()
                if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                     Jump_strength=2*(1-random.random())
                     X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))
                     
                     if objf(X1)< fitness: # improved move?
                        X[i,:] = X1.copy()
                     else: # Perform levy-based short rapid dives around the rabbit
                         X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                         if objf(X2)< fitness:
                            X[i,:] = X2.copy()
                
        convergence_curve[t]=Rabbit_Energy
        if (t%1==0):
               print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Rabbit_Energy)])
        t=t+1
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="HHO"   
    s.objfname=objf.__name__
    s.best =Rabbit_Energy 
    s.bestIndividual = Rabbit_Location

    return s

def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*numpy.random.randn(dim)*sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v),(1/beta))
    step = numpy.divide(u,zz)
    return step

lb = [-0.9, -1.1, -2.5, -2.5, -2.5, -2.5, -2.5]
ub = [0.3, -0.6, 2.5, 2.5, 2.5, 2.5, 2.5]
dim = 7
SearchAgents_no = 50
Max_iter = 250

objective_with_scaler = partial(objective_function_adjusted, scaler=scaler)

objective_with_scaler.__name__ = objective_function_adjusted.__name__ + "_with_scaler"

optimized_result = HHO(objf=objective_with_scaler, lb=np.full(dim, lb), ub=np.full(dim, ub), dim=dim, SearchAgents_no=SearchAgents_no, Max_iter=Max_iter)
print("Optimized Feature Set:", optimized_result.bestIndividual)
print("Minimum Predicted Voltage:", optimized_result.best)


scaled_features = np.array([[0.3, -0.6,  -0.74567583, 2.02884016,  0.38163072, -0.37479887, 2.5 , 2.038]])

original_features = scaler.inverse_transform(scaled_features)
selected_features_sum = np.sum(original_features[0, -6:-1])
adjustment_factor = 1 / selected_features_sum
original_features[0, -6:-1] *= adjustment_factor
print("调整后的逆标准化特征值:", original_features)

def predict_with_manual_input(model):
    # 假设您的模型有N个特征
    N = 8  # 这里改成您模型实际的特征数量
    user_input = []
    for i in range(N):
        # 对于每个特征，从用户那里获取输入值
        inp = float(input(f"请输入特征{i+1}的值: "))
        user_input.append(inp)

    input_tensor = torch.FloatTensor([user_input])
    
    with torch.no_grad():
        output = model(input_tensor)
    print("模型的预测结果是:", output.numpy())  # 根据您的模型输出类型，这里可能需要调整

predict_with_manual_input(model)




