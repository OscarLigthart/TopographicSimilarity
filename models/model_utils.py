from utils import *
from data import *
from models import *



def load_trained_models(args, vocab, gen_attr):
    """
    This function loads pretrained models (if available)
    :param args:
    :param vocab:
    :param gen_attr:
    :return:
    """

    all_models = []

    for seed in range(1,11):
        # get sender and receiver models
        sender = Sender(
            vocab,
            args.max_length,
            input_size=sum(gen_attr),
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            greedy=True,
        )

        receiver = Receiver(
            vocab,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            output_size=sum(gen_attr),
        )

        model = ReferentialTrainer(sender, receiver)

        # get model path
        model_name = get_filename(args)
        run_folder = "runs/" + model_name + "/" + str(seed)
        model_path = run_folder + "/model.p"

        # if model exists, load its state
        if os.path.isfile(model_path):
            epoch, iteration = load_model_state(model, model_path)

            if iteration < 9800:
                print("This model has not finished training yet")
        else:
            print("no model found")

        # append model to list
        all_models.append(model)

    return all_models
