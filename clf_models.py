"""Need to add a simple classifier code here that takes in a simple input image shape and number of output classes
and generates a simple classifier output"""
import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import resnet
from efficientnet_pytorch import EfficientNet
from pretrainedmodels import se_resnext101_32x4d, inceptionresnetv2


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting == True:  # made sure to change to '== True' since just 'if feat:' is true for 'partial'
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(config):
    """
    config requires:
        config.model_name: one of the model names
        config.num_classes:
        config.feature_extract
        config.use_pretrained

    Supported models {model name: specific model}:
        {
            simplenet_M_N: custom simple classifier for an image shape of MxMxN (height, width, channels)
            alexnet: Alexnet,
            vgg11bn: VGG11_bn,
            vgg19bn: VGG19_bn,
            resnet9: Modified Resnet18 with only one block per layer so that it's 'half',
            resnet18: Resnet18,
            resnet50, Resnet50,
            resnet101: Resnet101,
            resnet152: Resnet152,
            densenet121: Densenet121
            densenet161: Densenet161,
            densenet169: Densenet169,
            efficientnet-b0: EfficientNet-B0,
            efficientnet-b1: EfficientNet-B1,
            efficientnet-b2: EfficientNet-B2,
            efficientnet-b3: EfficientNet-B3,
            efficientnet-b5: EfficientNet-B5,
            efficientnet-b7: EfficientNet-B7,
            squeezenet: Squeezenet1.1,
            mobilenet: MobileNetV2,
            inception: Inception v3,
            inceptionresnetv2: Inception ResNet v2,
            resnext101: resnext101_32x8d,
            se-resnext101: se_resnext101_32x4d,
        }

        see link for more models and details on each: https://pytorch.org/docs/stable/torchvision/models.html

    To add another model into this function:
        1) install the model
        2) create a model_name
        3) copy the format of the other models and create the model_ft, set_parameter, etc.
            3.5) the num_ftrs might need to load different names (i.e. model_ft.fc or model_ft._fc - different models
            will have different final layers but will generally be 'fc' for pytorch pretrained models; '_fc' for
            efficientNet and 'last_layer' for pretrainedmodels)
        4) add the final layer as just a linear layer*
            4.5)* final layer needs to be a linear layer because pytorch loss functions include softmax and
            sigmoid activations within them
        5) make sure to read the originial architecture and set the input size
    """

    print('initializing {} model'.format(config.model_name))

    if "simplenet" in config.model_name:
        print('simplenet not in place yet...')
        exit()

    elif config.model_name == "resnext101":
        """resnext101_32x8d
        """
        model_ft = models.resnext101_32x8d(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "se-resnext101":
        """ SE-ResNeXt101
        """
        if config.use_pretrained:
            model_ft = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            model_ft = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "inceptionresnetv2":
        """ Inception ResNet v2
        """
        if config.use_pretrained:
            model_ft = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        else:
            model_ft = inceptionresnetv2(num_classes=1000, pretrained=None)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, config.num_classes)
        input_size = 299

    elif config.model_name == "efficientnet-b7":
        """ EfficientNet-B7
        """
        if config.use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b7')
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "efficientnet-b5":
        """ EfficientNet-B5
        """
        if config.use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b5')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b5')
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "efficientnet-b3":
        """ EfficientNet-B3
        """
        if config.use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b3')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b3')
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "efficientnet-b2":
        """ EfficientNet-B2
        """
        if config.use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b2')
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "efficientnet-b1":
        """ EfficientNet-B1
        """
        if config.use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b1')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b1')
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "efficientnet-b0":
        """ EfficientNet-B0
        """
        if config.use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b0')
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == 'resnet9':
        """ Modified Resnet18 with only one block per layer so that it's 'half'
        - this is a good way to make custom models from pytorch classes
        """
        model_ft = resnet._resnet('resnet18_half', resnet.BasicBlock, [1, 1, 1, 1], False, False)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "vgg11bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "vgg19bn":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "squeezenet":
        """ Squeezenet1.1
        """
        model_ft = models.squeezenet1_1(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, config.num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = config.num_classes
        input_size = 224

    elif config.model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "densenet169":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "densenet161":
        """ Densenet161
        """
        model_ft = models.densenet161(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "mobilenet":
        """ MobileNetV2
        """
        model_ft = models.mobilenet_v2(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, config.num_classes)
        input_size = 224

    elif config.model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model_ft, config.feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, config.num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, config.num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size