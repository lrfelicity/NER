import bilsm_crf_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

EPOCHS = 1
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
# train model
model.fit(train_x, train_y,batch_size=16,epochs=EPOCHS, validation_data=[test_x, test_y])
model.save('model/crf.h5')
