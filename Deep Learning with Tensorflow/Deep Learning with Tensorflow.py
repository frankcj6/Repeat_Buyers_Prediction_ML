#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import tempfile
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE


# In[2]:


train = pd.read_csv('C:/Stats Consulting/train_final.csv')
train.info()


# In[3]:


test = pd.read_csv('C:/Stats Consulting/test_final.csv')
test.head()


# In[4]:


#class label imbalance
neg, pos = np.bincount(train['label'])
total = neg+pos
print('Total:{}\n'
      'Positive:{}({:2f}% of total)\n'.format(total, pos, 100*pos/total))


# In[5]:


#fill all infinite entry with 0
train = train.replace(np.inf, 0)
test = test.replace(np.inf, 0)


# In[5]:


#data split (training_set, testing_set)
data_output = train.label
data_input = train.drop('label',axis=1)
data_input = data_input.drop('user_id', axis=1)
data_input = data_input.drop('seller_id', axis=1)


# In[6]:


training_set_x, testing_set_x, training_set_y, testing_set_y = train_test_split(data_input, data_output, test_size=0.3, random_state=1)


# In[7]:


train_x = training_set_x.values
train_y = training_set_y.values
test_x = testing_set_x.values
test_y = testing_set_y.values

#form labels
bool_label = train_y != 0


# In[8]:


scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)


# In[9]:


train_x


# In[10]:


pos_df = pd.DataFrame(train_x[ bool_label], columns = training_set_x.columns)
neg_df = pd.DataFrame(train_x[~bool_label], columns = training_set_x.columns)


# In[11]:


sns.jointplot(pos_df['user_click_cnt'], pos_df['user_purchase_cnt'], xlim=(-5,5), ylim=(-5,5), kind = 'hex')
plt.suptitle('Positive Distribution')

sns.jointplot(neg_df['user_click_cnt'], neg_df['user_purchase_cnt'], xlim=(-5,5), ylim=(-5,5), kind= 'hex')
plt.suptitle('Negative Distribution')


# In[12]:


Metrics = [keras.metrics.BinaryAccuracy(name='accuracy'),
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall'),
           keras.metrics.AUC(name='auc')]


# In[13]:


def model_build(metrics=Metrics, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([keras.layers.Dense(75, activation='relu', input_shape= (train_x.shape[-1],)),
                             keras.layers.Dropout(0.5),
                             keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),])
    model.compile(optimizer= keras.optimizers.Adam(lr=1e-3),
                 loss= keras.losses.BinaryCrossentropy(),
                  metrics = metrics)
    
    return model


# In[14]:


EPOCHS = 100
BATCH_SIZE = 10

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                 verbose=1,
                                                 patience=10,
                                                 mode='max',
                                                 restore_best_weights=True)

model_initial = model_build()
model_initial.summary()


# In[15]:


model_initial.predict(train_x)


# In[16]:


results = model_initial.evaluate(train_x, train_y, batch_size = BATCH_SIZE, verbose=0)
print('Loss:{:0.4f}'.format(results[0]))


# In[17]:


initial_bias = np.log([pos/neg])
initial_bias


# In[18]:


model_fix_bias = model_build(output_bias = initial_bias)
model_fix_bias.predict(train_x)


# In[19]:


results = model_fix_bias.evaluate(train_x, train_y, batch_size=BATCH_SIZE, verbose=0)
print('Loss: {:0.4f}'.format(results[0]))


# In[20]:


initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model_fix_bias.save_weights(initial_weights)
encoder = LabelEncoder()
encoder.fit(train_y)
encoded_Y = encoder.transform(train_y)
encoder.fit(test_y)
encoded_Y2 = encoder.transform(test_y)


# In[21]:


model_zero_bias = model_build()
model_zero_bias.load_weights(initial_weights)
model_zero_bias.layers[-1].bias.assign([0.0])
with tf.device('cpu:0'):
    zero_bias_history = model_zero_bias.fit(train_x, encoded_Y, batch_size = BATCH_SIZE, epochs=20, validation_data=(test_x, encoded_Y2), verbose=0)


# In[22]:


model_bias = model_build()
model_bias.load_weights(initial_weights)
with tf.device('cpu:0'):
    careful_bias = model_bias.fit(train_x, encoded_Y, batch_size = BATCH_SIZE, epochs=20, validation_data=(test_x, encoded_Y2), verbose=0)


# In[23]:


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[24]:


def plot_loss(history, label, n):
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val '+label, linestyle='--')
    plt.xlabel('Epoch Count')
    plt.ylabel('Loss')
    
    plt.legend()


# In[25]:


plot_loss(zero_bias_history, 'No Bias', 0)
plot_loss(careful_bias, 'Model with Careful Bias', 1)


# In[26]:


#model training
model = model_build()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
baseline_history = model.fit(
    train_x,
    encoded_Y,
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    validation_data=(test_x, encoded_Y2))


# In[27]:


def plot_metrics(history):
    metrics= ['loss','auc','precision','recall']
    for n, metric in enumerate(metrics):
        name = metric.replace('_',' ').capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train Data')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle='--', label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric=='loss':
            plt.ylim([0.2, 0.25])
        elif metric=='auc':
            plt.ylim([0.6, 0.7])
        elif metric=='recall':
            plt.ylim([0, 0.02])
        else:
            plt.ylim([0,1])
        
        plt.legend()


# In[28]:


plot_metrics(baseline_history)


# In[29]:


train_pred_baseline = model.predict(train_x, batch_size=BATCH_SIZE)
test_pred_baseline = model.predict(test_x, batch_size=BATCH_SIZE)


# In[30]:


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print('True Non-Repeat customers Detected (True Negatives): ', cm[0][0])
    print('Non-repeat customers Incorrectly Detected (False Positives): ', cm[0][1])
    print('repeat customers Missed (False Negatives): ', cm[1][0])
    print('repeat customers Detected (True Positives): ', cm[1][1])
    print('Total repeat customers ', np.sum(cm[1]))


# In[31]:


baseline_results = model.evaluate(test_x, encoded_Y2, batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plot_cm(encoded_Y2, test_pred_baseline)


# In[32]:


#Oversampling with smote
sm = SMOTE()
training_x_sm, training_y_sm = sm.fit_sample(training_set_x, training_set_y)
train_x_sm = training_x_sm.values
train_y_sm = training_y_sm.values
test_x = testing_set_x.values
test_y = testing_set_y.values
scaler = StandardScaler()
train_x_sm = scaler.fit_transform(train_x_sm)
test_x = scaler.fit_transform(test_x)
encoder = LabelEncoder()
encoder.fit(train_y_sm)
encoded_Y_sm = encoder.transform(train_y_sm)
encoder.fit(test_y)
encoded_Y2_sm = encoder.transform(test_y)


# In[33]:


train_x_sm


# In[34]:


train_y_sm


# In[35]:


test_x


# In[36]:


train_y_sm.mean()


# In[37]:


train_y_sm.sum()


# In[38]:


resample_steps_per_epoch = np.ceil((2.0*171495)/10)


# In[39]:


resample_steps_per_epoch


# In[45]:


resample_model = model_build()
resample_model.load_weights(initial_weights)

output_layer = resample_model.layers[-1]
output_layer.bias.assign([0])

resample_history = resample_model.fit(
    train_x_sm,
    encoded_Y_sm,
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS,
    validation_data=(test_x, encoded_Y2_sm))


# In[46]:


def plot_metrics(history):
    metrics= ['loss','auc','precision','recall']
    for n, metric in enumerate(metrics):
        name = metric.replace('_',' ').capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train Data')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle='--', label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric=='loss':
            plt.ylim([0, 1])
        elif metric=='auc':
            plt.ylim([0.4, 1])
        elif metric=='recall':
            plt.ylim([0,1])
        else:
            plt.ylim([0,1])
        
        plt.legend()


# In[47]:


plot_metrics(resample_history)


# In[48]:


resample_model = model_build()
resample_model.load_weights(initial_weights)

output_layer = resample_model.layers[-1]
output_layer.bias.assign([0])

resample_history = resample_model.fit(
    train_x_sm,
    encoded_Y_sm,
    epochs=EPOCHS, 
    validation_data=(test_x, encoded_Y2_sm))


# In[49]:


plot_metrics(resample_history)


# In[50]:


train_prediction_resample = resample_model.predict(train_x, batch_size=BATCH_SIZE)
test_prediction_resample = resample_model.predict(test_x, batch_size=BATCH_SIZE)


# In[51]:


resampled_results = resample_model.evaluate(test_x, test_y, batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(resample_model.metrics_names, resampled_results):
    print(name, ':', value)
print()

plot_cm(test_y, test_prediction_resample)


# In[52]:


test_prediction_resample


# In[53]:


test


# In[54]:


test_x_id = test[['user_id','seller_id']]
test_x = test.drop(['user_id','seller_id'], axis=1)
test_x_id.head()


# In[55]:


test_x.head()


# In[56]:


test_x = test_x.values
scaler = StandardScaler()
test_x = scaler.fit_transform(test_x)


# In[57]:


final_prediction = resample_model.predict(test_x, batch_size=BATCH_SIZE)
final_prediction


# In[58]:


final_submission = pd.DataFrame(final_prediction, columns=['Prob'])
final_output_dl = pd.concat([test_x_id.reset_index(drop=True), final_submission.reset_index(drop=True)], axis=1)


# In[59]:


final_output_dl.head()


# In[60]:


final_output_dl.to_csv('C:/Stats Consulting/final_submission_dl.csv', index=False)


# In[ ]:




