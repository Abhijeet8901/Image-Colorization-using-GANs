
 ██████╗ ██████╗ ██╗      ██████╗ ██████╗ ██╗███████╗███████╗
██╔════╝██╔═══██╗██║     ██╔═══██╗██╔══██╗██║╚══███╔╝██╔════╝
██║     ██║   ██║██║     ██║   ██║██████╔╝██║  ███╔╝ █████╗  
██║     ██║   ██║██║     ██║   ██║██╔══██╗██║ ███╔╝  ██╔══╝  
╚██████╗╚██████╔╝███████╗╚██████╔╝██║  ██║██║███████╗███████╗
 ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝
                                                             

DESCRIPTION
    Grayscale image colorization using a conditional DCGAN

USAGE
    First make sure that you have every packages in the requirements.txt file.

    The input image resolution should be either 32x32 or 256x256.
    To train using DCGAN change the config/train.json file to put the correct
    parameters and launch the training using:
        python train.py --config config/train.json

    To test the trained model, change the config/colorize.json file to put the
    correct parameters and launch the inference using:
        python colorize.py --config config/colorize.json

REFERENCES
    Colorful Image Colorization (https://arxiv.org/abs/1603.08511)
    Image Colorization with Generative Adversarial Networks (https://arxiv.org/abs/1803.05400)

CONTRIBUTORS
    Hussem Ben Belgacem
