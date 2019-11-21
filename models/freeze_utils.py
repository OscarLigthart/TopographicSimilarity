from utils import *
from data import *
from models import *

def freeze_sender(args, model, run_folder):
    """
    This function freezes a
    :param args: arguments given by user
    :param model: trained models
    :param run_folder: path to experiment
    :return: model holding the sender and receiver
    """

    # get the vocabulary
    vocab = AgentVocab(args.vocab_size)

    # get objects attribute early, to determine input size of models
    gen_attr = get_attributes(args.attributes)

    # flag sender parameters to not be updated anymore
    for param in model.sender.parameters():
        param.requires_grad = False

    # set the freeze seed
    seed_torch(args.freeze_seed)

    # intialize a new receiver model
    receiver = Receiver(
        vocab,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        output_size=sum(gen_attr),
    )

    # initialize new model
    model = ReferentialTrainer(model.sender, receiver)

    # reset iteration and epoch counter
    epoch = 0
    iteration = 0

    # reset seed
    seed_torch(args.seed)

    # set new path
    model_path = run_folder + '/freezes_sender/' + str(args.freeze_seed) + "/model.p"

    # create folder for new freeze model
    create_folder_if_not_exists(run_folder + '/freezes_sender/' + str(args.freeze_seed))

    # return new model
    return model, epoch, iteration, model_path


def freeze_receiver(args, model, run_folder):
    """
    This function freezes a receiver in the referential game setup
    :param args: arguments given by user
    :param model: trained models
    :param run_folder: path to experiment
    :return:
    """

    # get the vocabulary
    vocab = AgentVocab(args.vocab_size)

    # get objects attribute early, to determine input size of models
    gen_attr = get_attributes(args.attributes)

    # flag sender parameters to not be updated anymore
    for param in model.receiver.parameters():
        param.requires_grad = False

    # set the freeze seed
    seed_torch(args.freeze_seed)

    # intialize a new receiver model
    sender = Sender(
        vocab,
        args.max_length,
        input_size=sum(gen_attr),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        greedy=True,
    )

    # initialize new model
    model = ReferentialTrainer(sender, model.receiver)

    # reset iteration and epoch counter
    epoch = 0
    iteration = 0

    # reset seed
    seed_torch(args.seed)

    # set new path
    model_path = run_folder + '/freezes_receiver/' + str(args.freeze_seed) + "/model.p"

    # create folder for new freeze model
    create_folder_if_not_exists(run_folder + '/freezes_receiver/' + str(args.freeze_seed))

    # return new model
    return model, epoch, iteration, model_path

