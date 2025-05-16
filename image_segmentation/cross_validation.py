import random

class Options():

    def __init__(self):
        
        #model
        self.model = "deeplabv3plus_mobilenet"
        self.num_classes = 3
        self.output_stride = 8

        self.dataset = "Cow_segmentation"
        self.save_val_results = True

        #constants
        self.val_interval = 50
        self.total_itrs = 4000
        self.val_batch_size = 10
        self.loss_type = 'cross_entropy'

        #hyperparameter
        self.batch_size = 15

        self.lr = 0.01
        self.weight_decay = 0.01
        self.lr_policy = "step"
        self.step_size = 0.001

        #visualize Option
        self.enable_vis = None

        #seed
        self.random_seed = random.randint(0,1000000)

        #continue traning
        self.ckpt= "data/checkpoints/best_deeplabv3plus_mobilenet_Cow_segmentation_os8.pth" #"data\checkpoints\latest_deeplabv3plus_resnet50_Cow_segmentation_os8.pth" 
        self.continue_training = True

def create_folds(images_path, n_folds):
   
   pass