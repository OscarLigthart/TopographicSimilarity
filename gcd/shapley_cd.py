import torch
import pickle

from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions.factory import DecomposerFactory
from diagnnose.models.import_model import import_model
from diagnnose.typedefs.corpus import Corpus
from diagnnose.typedefs.models import LanguageModel
from diagnnose.vocab import get_vocab_from_config

if __name__ == "__main__":
    arg_groups = {"model", "init_states", "corpus", "vocab", "activations", "decompose"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args)

    model: LanguageModel = import_model(config_dict)
    vocab_path = get_vocab_from_config(config_dict)
    corpus: Corpus = import_corpus(vocab_path=vocab_path, **config_dict["corpus"])

    decompose_args = {**config_dict["decompose"], **config_dict["activations"]}

    print("Initializing decomposition")

    sen_ids = slice(0, 625)

    factory = DecomposerFactory(model, corpus=corpus, sen_ids=sen_ids, **decompose_args)

    decomposer = factory.create(
        sen_ids, classes=torch.arange(20), subsen_index=slice(0, None)
    )

    # get current seed
    seed = config_dict['model']['state_dict'][:2]

    print(seed)

    print("Decomposing...")

    partition = decomposer.decompose(None, gate_bias_rel=False)

    print(partition.shape)

    matrix = partition[:, :, -1, :]
    pickle.dump(matrix, open("Experiments/1/GCDmatrix.p", "wb"))
    print(matrix.shape)
