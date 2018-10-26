# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:17:27 2018

@author: niewei
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:23:11 2018

@author: niewei
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
from collections import Counter



os.chdir(r'E:\niewei\Desktop\2018_progress\201810\复贷phead模型')
df  =pd.read_csv('nw_vjxl_phead_04.csv')
df['phead'] = df.phead.apply(lambda x: 'CALL_LEN_' + str(x).zfill(7) if str(x).__len__() == 6 else 'CALL_LEN_' + str(x))
df_pivot = pd.pivot_table(df,values = 'call_len_sum',index = 'report_id',columns = 'phead')
df_pivot = df_pivot.fillna(0)
df_pivot = df_pivot.reset_index()
df_model = pd.read_csv('nw_fd_phead_03.csv')
df_model = df_model[df_model.status  == 372]
df_model = df_model[(df_model.add_date >= 20180601)]
df_model =  df_model.drop(['stat_date','report_time','rn','add_time','should_time','fact_time','is_patron'],axis = 1)
df_model = df_model[df_model.report_id.isin(df_pivot.report_id)]
df_model = df_model[~df_model.qudao.isnull()]
df_merge = pd.merge(df_model,df_pivot,how = 'left',on = 'report_id')


df_merge['y'] = np.where(df_merge.ind1_dpd >= 5,1,0)
df_merge_35 = df_merge[df_merge.product_type == 35]
df_merge_59 = df_merge[df_merge.product_type == 59]
#复贷嘉卡
dataSet_35 = df_merge_35[(df_merge_35.add_date < 20180801)]
validation_35 = df_merge_35[df_merge_35.add_date >= 20180801]

dataSetTarget = dataSet_35.y
dataSetData = dataSet_35.drop(['cid','status','add_date','qudao','product_type','sloop_pass','ind1_dpd','ind1_rpd','report_id','y'],axis = 1)
validationTarget = validation_35.y
validationData = validation_35.drop(['cid','status','add_date','qudao','product_type','sloop_pass','ind1_dpd','ind1_rpd','report_id','y'],axis = 1)


#复贷秒啦
dataSet_59 = df_merge_59[(df_merge_59.add_date < 20180801) & (df_merge_59.add_date >= 20180601)]
validation_59 = df_merge_59[df_merge_59.add_date >= 20180801]

dataSetTarget = dataSet_59.y
dataSetData = dataSet_59.drop(['cid','status','add_date','qudao','product_type','sloop_pass','ind1_dpd','ind1_rpd','report_id','y'],axis = 1)
validationTarget = validation_59.y
validationData = validation_59.drop(['cid','status','add_date','qudao','product_type','sloop_pass','ind1_dpd','ind1_rpd','report_id','y'],axis = 1)



trainData, testData, trainTarget, testTarget = train_test_split(
    dataSetData, dataSetTarget, test_size=0.3,random_state=3)

print("TrainData len : %d" % (len(trainData)))         # 训练集
print("Validation len : %d" % (len(validationData)))   # 测试集
print("TestData len : %d" % (len(testData))) 


def modelfit(other_params,cv_params):
    model = xgb.XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc',
                                 cv=StratifiedKFold(trainTarget, n_folds=5, shuffle=True), verbose=1, n_jobs=4)
    optimized_GBM.fit(trainData, trainTarget)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return optimized_GBM.best_params_



def printProbResult(testTarget,y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(testTarget,y_pred)
    p,r,_ = metrics.precision_recall_curve(testTarget,y_pred)
    print("PR Curve AUC %0.2f%%"%(metrics.auc(p,r,reorder=True)*100))
    print("ROC Curve AUC %0.2f%%"%(metrics.auc(fpr,tpr)*100))
    print("KS %0.2f%%"%(max(tpr - fpr)*100))



def create_feature_map(features,filename='feature.fmap'):
    outfile = open(filename, 'w',encoding='utf8')
    for i, feat in enumerate(features):
        try:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        except Exception as e:
            print(feat)
            print(e)
    outfile.close()


dtrain = xgb.DMatrix(trainData, label=trainTarget)
dvalid = xgb.DMatrix(validationData, label=validationTarget)
dtest = xgb.DMatrix(testData, label=testTarget)
ks,auc = 0,0
params = {
    'booster': 'gbtree',  # 用树来训练
    'objective': 'binary:logistic',  # 学习目标，二进制，逻辑回归
    'n_estimators': 600,
    'eval_metric': 'auc',  # 用auc来验证模型效果
    'max_depth': 5,   # 最深深度
    'reg_lambda': 1,  # L2正则化参数，越大越不容易过拟合
    'reg_alpha':5,    # 权重的L1正则化项
    'subsample': 0.8,  # 样本随机采样率
    'colsample_bytree': 0.8,  # 特征随机采样率
    'min_child_weight': 4,  #
    'eta': 0.02,  # 学习率，步长
    'seed': 1,  # 随机数种子
    'nthread': 8,  # 进程数
    'gamma':0.2,   # 指定了节点分裂所需的最小损失函数下降值
    'silent': 0,  # 运行中是否打印信息，1为不打印
    'scale_pos_weight': 1  # 对分类中较少的加权
}


watchlist = [(dtrain,'train'), (dvalid, 'eval')]
clf = xgb.train(params, dtrain, evals=watchlist, num_boost_round=1000, early_stopping_rounds=100, verbose_eval=True)

y_pred = clf.predict(dtrain)
printProbResult(trainTarget, y_pred)
y_pred = clf.predict(dvalid)
printProbResult(validationTarget, y_pred)
y_pred = clf.predict(dtest)
printProbResult(testTarget, y_pred)



print(clf.best_score, clf.best_iteration,clf.best_ntree_limit) 

importances = clf.get_score(importance_type='gain')
fscoreDf = pd.DataFrame({'feature': list(importances.keys()), 'fscore': list(importances.values())})
fscoreDf['fscoreNew'] = fscoreDf['fscore'] / sum(fscoreDf['fscore'])
fscoreDf = fscoreDf.sort_values(by='fscoreNew', ascending=False)
print(fscoreDf)

#模型保存
columnList = trainData.columns
create_feature_map(columnList,filename='xgb_applist.fmap')
clf.save_model("xgb_applist.model")
clf.dump_model("xgb_applist.dump",with_stats=True)



"""psi计算"""
#嘉卡
df_merge_35_train =  df_merge_35.drop(['cid','status','add_date','qudao','product_type','sloop_pass','ind1_dpd','ind1_rpd','report_id','y'],axis = 1)
df_merge_35_target = df_merge_35.y
df_merge_35_dtrain = xgb.DMatrix(df_merge_35_train, label=df_merge_35_target)
y_pred = clf.predict(df_merge_35_dtrain)
df_merge_35['preb'] = y_pred
df_merge_35_psi = df_merge_35[['add_date','preb','y']]

#秒啦
df_merge_59_train =  df_merge_59.drop(['cid','status','add_date','qudao','product_type','sloop_pass','ind1_dpd','ind1_rpd','report_id','y'],axis = 1)
df_merge_59_target = df_merge_59.y
df_merge_59_dtrain = xgb.DMatrix(df_merge_59_train, label=df_merge_59_target)
y_pred = clf.predict(df_merge_59_dtrain)
df_merge_59['preb'] = y_pred
df_merge_59_psi = df_merge_59[['add_date','preb','y']]


cut_value = df_merge_59_psi.query("add_date < 20180801")['preb'].quantile(np.linspace(0,1,6)).values
by_psi = pd.cut(df_merge_59_psi.query("add_date >= 20180801")['preb'],bins = cut_value,labels = range(1,6)).value_counts().pipe(lambda x: x/x.sum())
valid_by_psi = df_merge_59_psi.query("add_date >= 20180801").groupby(['add_date'])['preb'].apply(lambda x: pd.cut(x,bins = cut_value,labels = range(1,6)).value_counts()).reset_index()
valid_by_psi.columns = ['add_date','level','cnt']
#排序性
"""嘉卡"""
valid_fenzu = pd.cut(df_merge_35_psi.query("add_date >= 20180801")['preb'],bins = valid_cut_value,labels = range(1,10))
valid_fenzu['y'] = df_merge_35_psi.query("add_date >= 20180801")['y']
pd.pivot_table(pd.DataFrame([valid_fenzu,df_merge_35_psi.query("add_date >= 20180801")['y']]).T,index = 'preb',values = 'y',aggfunc = lambda x: sum(x)/count(x))

"""秒啦"""
valid_cut_value = df_merge_59_psi.query("add_date >= 20180801")['preb'].quantile(np.linspace(0,1,10)).values
valid_fenzu = pd.cut(df_merge_59_psi.query("add_date >= 20180801")['preb'],bins = valid_cut_value,labels = range(1,10))
pd.pivot_table(pd.DataFrame([valid_fenzu,df_merge_59_psi.query("add_date >= 20180801")['y']]).T,index = 'preb',values = 'y',aggfunc = lambda x: sum(x)/x.count())



def psi_single(data,varbin,date,cnt,overby,byday):
    df=pd.pivot_table(data,index=[date]+ [varbin],values=cnt,aggfunc=[sum]).reset_index()
    df.columns=[date]+[varbin]+[cnt]
    df['pct']=df.groupby(date)[cnt].apply(lambda x: x/x.sum())
    df_F = df.assign(woe = varbin)
    df_F.columns = [date] + ['woe'] + [cnt] + ['pct'] + ['var_name']
    if overby == 0:
        df['pcts']=df.groupby(varbin)['pct'].shift(1)
        df['psi']=(df['pct']/df['pcts']).apply(math.log)*(df['pct']-df['pcts'])
        df_psi=pd.pivot_table(df,index = date,values=['psi'],aggfunc=[sum])
        df_psi.columns=[varbin]
    elif overby == 1:
        byday_pct = df[df.add_date == byday][[varbin,'pct']]
        df_merge = pd.merge(df,byday_pct,on = varbin).rename(columns = {'pct_x':'pct','pct_y':'pctf'})
        df_merge['psi']=(df_merge['pct']/df_merge['pctf']).apply(math.log)*(df_merge['pct']-df_merge['pctf'])
        df_psi=pd.pivot_table(df_merge,index = date,values=['psi'],aggfunc=[sum])
        df_psi.columns=[varbin]
    return df_psi,df_F



