import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")


train_arg.add_argument("--data_dir", type=str,
                       default="C:/Users/shen/Desktop",
                       help="Directory with CIFAR10 data")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=100,
                       help="Size of each training batch")

train_arg.add_argument("--num_epoch", type=int,
                       default=100,
                       help="Number of epochs to train")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--feature_type", type=str,
                       default="hog",
                       choices=["hog", "h_histogram", "rgb"],
                       help="Type of feature to be used")

model_arg.add_argument("--reg_lambda", type=float,
                       default=1e-4,
                       help="Regularization strength")

model_arg.add_argument("--num_unit", type=int,
                       default=64,
                       help="Number of neurons in the hidden layer")

model_arg.add_argument("--num_hidden", type=int,
                       default=3,
                       help="Number of hidden layers")

model_arg.add_argument("--num_class", type=int,
                       default=10,
                       help="Number of classes in the dataset")

model_arg.add_argument("--activ_type", type=str,
                       default="relu",
                       choices=["relu", "tanh"],
                       help="Activation type")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

