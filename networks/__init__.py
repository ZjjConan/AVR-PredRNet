from .resnet4b import resnet4b
from .predrnet import predrnet_raven, predrnet_analogy

model_dict = {
    "resnet4b": resnet4b,
    "predrnet_raven": predrnet_raven,
    "predrnet_analogy": predrnet_analogy
}


def create_net(args):
    net = None

    kwargs = {}
    kwargs["block_drop"] = args.block_drop
    kwargs["classifier_drop"] = args.classifier_drop
    kwargs["classifier_hidreduce"] = args.classifier_hidreduce
    kwargs["num_filters"] = args.num_filters
    kwargs["num_extra_stages"] = args.num_extra_stages
    kwargs["in_channels"] = args.in_channels

    net = model_dict[args.arch.lower()](**kwargs)

    return net

