import torch
import random

class Options():
    
    # Klasse für Trainings Kofiguration

    def __init__(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clases = {"backgroud": 0, "cow": 1, "heu": 2} #registered classes

        #model
        self.model = "deeplabv3plus_resnet50" # backbone Model and Segmentation Head 
        self.num_classes = 3 #number of classes
        
        self.dataset = "Cow_segmentation" # name of the dataset
        self.save_val_results = False #if True saves the results of the validation as images

        #constants
        self.val_interval =2 #epoch interval for validation
        self.total_epochs = 50 # total numbr of epochs
        self.val_batch_size = 10 #validation batchsize
        self.max_decrease = 0.35 #value for early stopping if Mean IoU decreacse too much

        #hyperparameter
        self.batch_size = 15 #trainings batchsize
        self.class_weights = torch.tensor([0.2, 1.0, 1.0], device=device) #class weights for the cross entropy loss
        self.lr = 0.01 #learning rate 
        self.weight_decay = 0.01 #weight decay value

        #paramters for the lr Sceduler
        self.lr_policy = "step" #Learning rate scheduler policy ["step", "poly", "none"]
        self.poly_power = 0.9 # power value for the polynomial Learning rate scheduler
        self.step_size = 5 #step sieze for the step based Learning rate scheduler
        self.gamma = 0.1 #gamma value for the step based Learning rate scheduler

        self.loss_type = 'cross_entropy' #loss function  ["cross_entropy", "focal_loss"]

        self.freeze_backbone = False  # if True, freezes backbone weights during training.
        self.output_stride = 16 # Output stride of backbone

        #seed
        self.random_seed = random.randint(0,1000000)

        #continue traning
        self.save_param = False  #Whether to save parameter state dictionaries
        self.ckpt= None  # Path to checkpoint file to load
        self.continue_training = False # if True, resume training from checkpoint

    def to_dict(self):
        """
        Export Hyperparamter configuration as a dictionary.

        Returns
        -------
        dict
            Dictionary with Hyperparamter configuration.
        """
         
        return{"batchsize": self.batch_size,
                "output_stride":self.output_stride,
                "background_weighting": round(float(self.class_weights[0]), 6),
                "learning_rate": round(self.lr, 6), 
                "weight_decay": round(self.weight_decay, 6), 
                "loss_type": self.loss_type,
                "lr_policy": self.lr_policy,
                "freeze_backbone": self.freeze_backbone, 
                "gamma": self.gamma,
                "power": self.poly_power, 
                "step_size": self.step_size}
    
    def from_dict(self, params):

        """
        Load training options from a dictionary. If a Values is not configured a deafult values will be loaded

        Parameters
        ----------
        params : dict
            Dictionary of configuration values.
        """

        self.batch_size = params.get("batchsize", self.batch_size)
        self.output_stride = params.get("output_stride", self.output_stride)
        self.class_weights[0] = params.get("background_weighting", self.class_weights[0])
        self.lr = params.get("learning_rate", self.lr)
        self.weight_decay = params.get("weight_decay", self.weight_decay)
        self.loss_type = params.get("loss_type", self.loss_type)
        self.lr_policy = params.get("lr_policy", self.lr_policy)
        self.freeze_backbone = params.get("freeze_backbone", self.freeze_backbone)


        self.gamma = params.get("gamma", self.gamma)
        self.poly_power = params.get("power", self.poly_power)
        self.step_size = params.get("step_size", self.step_size)


    def create_dir_name(opts):
        params = [
            f"bt={opts.batch_size}", 
            f"str={opts.output_stride}",
            f"cw={round(float(opts.class_weights[0]), 4)}", 
            f"wd={round(opts.weight_decay, 6)}", 
            f"lr={round(opts.lr, 6)}",
            f"l={opts.loss_type}", 
            f"lp={opts.lr_policy}",
            f"fb={opts.freeze_backbone}"
        ]

        return "_".join(params)