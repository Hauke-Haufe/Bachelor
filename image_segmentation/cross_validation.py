import random

class Options():

    def __init__(self):
        
        #model
        self.model = "deeplabv3plus_resnet50"
        self.num_classes = 3
        self.output_stride = 8

        self.dataset = "Cow_segmentation"
        self.save_val_results = True

        #constants
        self.val_interval = 5
        self.total_itrs = 100 
        self.val_batch_size = 100
        self.loss_type = 'cross_entropy'

        #hyperparameter
        self.batch_size = 2
        self.lr = 0.01
        self.weight_decay = 0.01
        self.lr_policy = "step"
        self.step_size = 0.001

        #visualize Option
        self.enable_vis = None

        #seed
        self.random_seed = random.randint(0,1000000)

        #continue traning
        self.ckpt= None
        self.continue_training = False

def create_folds(images_path, n_folds):
   
   pass