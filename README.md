
# NSL KDD Dataset Feature Importance Calculation With Forests of Trees

## importing of required libraries 



```python
import os
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
```


```python
def fi(df,output,number_of_feature):
    X =df[df.columns[0:-1]]
    X=np.array(X)
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    y=df[df.columns[-1]].cat.codes  
    
    
    
    ################## this part taken: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    temp=list(df.columns)[0:-1]
    header=[]
    for f in range(X.shape[1]):
        print("%d. feature %d %s (%f)" % (f + 1, indices[f], temp[ indices[f]] ,importances[indices[f]]))
        header.append(temp[ indices[f]])
    # Plot the impurity-based feature importances of the forest
    plt.figure(figsize=(18,10))
    plt.title(output+"   Feature importances Features Based")

    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), header,rotation='vertical')
    plt.xlim([-1, X.shape[1]])


    graph_name=output+"_fi.pdf"
    plt.savefig(graph_name,bbox_inches='tight',format="pdf")#, dpi=400)
    plt.show()


    print("\n\n\n") 
    
    ##############################################################################################
    
    
    
    
    #FOCUSED GRAPH
   
    
    imp=[]
    for i,ii in enumerate (importances[indices]):
        imp.append(ii)
        print(i,ii)
        if i==number_of_feature-1:break
    st=[]
    for i,ii in enumerate (std[indices]):
        st.append(ii)
        print(i,ii)
        if i==number_of_feature-1:break
    hd=[]

    for i,ii in enumerate (header):
        hd.append(ii)
        print(i,ii)
        if i==number_of_feature-1:break
            

    plt.figure(figsize=(18,10))
    plt.title(output+" Feature Importances Packet Feature Based")


    plt.bar(range(number_of_feature), imp,
            color="r", yerr=st, align="center")
    plt.xticks(range(number_of_feature), hd,rotation='vertical')
    plt.xlim([-1, number_of_feature])
    plt.ylabel("Importance Scores")
    #ax.set_ylim([0,2])
    plt.grid()
    graph_name=output+"fi_focused.pdf"
    plt.savefig(graph_name,bbox_inches='tight',format="pdf")#, dpi=400)
    plt.show()


    print("\n\n\n") 
    
    return imp,st,hd
            
```

# prepare NSL KDD dataset


## reading CSV files


```python
# c_names --->  column names
c_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","difficulty_degree"]

train = pd.read_csv( "data/KDDTrain+.csv", names=c_names) # train file
```

## deletion of unnecessary feature (difficulty_degree)


```python
del train["difficulty_degree"] 
```

## Converting object features to categories first and then to dummy tables (except "labels")


```python
for i in c_names:
    print((train[i].dtypes))
    if train[i].dtypes==object:
        train[i] = train[i].astype('category')
        if i=="labels":
            break
        train=pd.get_dummies(train, columns=[i])

```

    int64
    object
    object
    object
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    int64
    float64
    float64
    float64
    float64
    float64
    float64
    float64
    int64
    int64
    float64
    float64
    float64
    float64
    float64
    float64
    float64
    float64
    object
    

# Move the labels property to the end of the dataset


```python
label=train["labels"]
del train["labels"]
train["labels"]=label
```

# What does the dataset look like?


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>src_bytes</th>
      <th>dst_bytes</th>
      <th>land</th>
      <th>wrong_fragment</th>
      <th>urgent</th>
      <th>hot</th>
      <th>num_failed_logins</th>
      <th>logged_in</th>
      <th>num_compromised</th>
      <th>...</th>
      <th>flag_RSTO</th>
      <th>flag_RSTOS0</th>
      <th>flag_RSTR</th>
      <th>flag_S0</th>
      <th>flag_S1</th>
      <th>flag_S2</th>
      <th>flag_S3</th>
      <th>flag_SF</th>
      <th>flag_SH</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>491</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>neptune</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>232</td>
      <td>8153</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>199</td>
      <td>420</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>125968</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>neptune</td>
    </tr>
    <tr>
      <th>125969</th>
      <td>8</td>
      <td>105</td>
      <td>145</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>125970</th>
      <td>0</td>
      <td>2231</td>
      <td>384</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>125971</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>neptune</td>
    </tr>
    <tr>
      <th>125972</th>
      <td>0</td>
      <td>151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>normal</td>
    </tr>
  </tbody>
</table>
<p>125973 rows × 123 columns</p>
</div>



# Calling the function to calculate the feature importance score


```python
graph_name="name" # save graphics with this name as pdf
number_of_feature= 45 # selects the first x feature with the highest score in the focused section
imp,st,hd=fi(train,graph_name,number_of_feature)
```

# List of top 45 features


```python


print ('%-5s %-25s %-25s %-25s' %("No","Importance Score","Standard Deviation","Feature Name"))
print ('%-5s %-25s %-25s %-25s' %("__","________________","__________________","____________"))
for i in range(len(imp)):
    print ('%-5s %-25s %-25s %-25s' %(i+1,imp[i],st[i],hd[i]))
    
    

```

    No    Importance Score          Standard Deviation        Feature Name             
    __    ________________          __________________        ____________             
    1     0.08541877835459762       0.15608762847225938       same_srv_rate            
    2     0.0715126329495212        0.14410100258757014       flag_SF                  
    3     0.06687675746585174       0.14786516391116508       dst_host_srv_serror_rate 
    4     0.066346211301629         0.14250820001379977       dst_host_serror_rate     
    5     0.0570168996187475        0.13270548222467024       flag_S0                  
    6     0.05059824741777188       0.13034275310379637       serror_rate              
    7     0.037683440633368356      0.11001664192796141       srv_serror_rate          
    8     0.035117018974980915      0.07569012892951156       dst_host_same_srv_rate   
    9     0.03479502386534906       0.07558018702662714       dst_host_srv_count       
    10    0.03467098639653562       0.08092480348147767       logged_in                
    11    0.030032863755838128      0.06309257831608235       count                    
    12    0.026609183533229924      0.01980281216988479       dst_host_same_src_port_rate
    13    0.024967587607099915      0.02708397211110129       protocol_type_icmp       
    14    0.02494787057724674       0.032536513778512965      dst_host_diff_srv_rate   
    15    0.024469259058899785      0.028096100636653777      dst_host_count           
    16    0.023990746384894297      0.011256508929679575      src_bytes                
    17    0.02331880947065069       0.012893965911553678      dst_host_srv_diff_host_rate
    18    0.022917093189360766      0.04705502666630768       diff_srv_rate            
    19    0.02216655165418429       0.032418970079455114      dst_host_rerror_rate     
    20    0.021525546658721823      0.023876827159024363      service_private          
    21    0.019489997063772555      0.02313708483432          service_eco_i            
    22    0.018549058889223726      0.039968499359808986      service_http             
    23    0.017647807545782702      0.01866707716633733       service_ecr_i            
    24    0.015792794774789773      0.028931186860282038      rerror_rate              
    25    0.013534433685204256      0.027039526105976115      protocol_type_tcp        
    26    0.013499123367919011      0.025548951543056725      dst_host_srv_rerror_rate 
    27    0.013110355377077934      0.013235343340733204      flag_RSTR                
    28    0.012428691461188109      0.01261493054322828       srv_count                
    29    0.011624521158259788      0.024828144954195674      srv_rerror_rate          
    30    0.01046277664077479       0.006027397164132387      wrong_fragment           
    31    0.008018051638573262      0.016600520246751186      flag_REJ                 
    32    0.007973786697546406      0.014926602170677613      protocol_type_udp        
    33    0.007426192863160025      0.009783061626885605      srv_diff_host_rate       
    34    0.006307124180762807      0.007646789455526549      service_other            
    35    0.006124228743258873      0.014459532046283802      service_domain_u         
    36    0.005640541788106428      0.00454943379110553       hot                      
    37    0.005231150633370187      0.003909570925593708      dst_bytes                
    38    0.004339292247485638      0.005449154150303683      num_compromised          
    39    0.0027606414006143007     0.003051403385162724      service_ftp_data         
    40    0.002621327285194019      0.0025849482116323996     duration                 
    41    0.0016221219674300818     0.005993559860935889      service_smtp             
    42    0.0015544607273336034     0.002195082539489958      flag_SH                  
    43    0.0014081665239780098     0.0032952392942288615     flag_RSTO                
    44    0.0008729895279251953     0.0009833945954346203     is_guest_login           
    45    0.0008294258464790644     0.0008731533908523693     service_ftp              
    
