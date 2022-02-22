from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('TRAINING GENERATOR')
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'/data/s2732815/unet/data/train','image','label',data_gen_args,save_to_dir = None)
print('TRAININED')


model = unet()
model_checkpoint = ModelCheckpoint('/data/s2732815/unet/data/unet_cells.hdf5', monitor='loss', verbose=1, save_best_only=True)
print('FITTING GENERATOR')
model.fit_generator(myGene,steps_per_epoch=1000,epochs=5,callbacks=[model_checkpoint])
print('FITTED')


print('TESTING GENERATOR')
testGene = testGenerator(test_path="data/s2732815/unet/data/test", num_image=125, flag_multi_class=True, target_size=target_size, as_gray=False)
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/s2732815/unet/data/test",results)
print('TESTED AND SAVED')
