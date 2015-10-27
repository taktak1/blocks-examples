from __future__ import print_function

import inspect
import logging
import numpy
import operator
import os
import re
import signal
import time
import argparse
import logging
import pprint
import configurations

from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.datasets import TextFile
from six.moves import cPickle
from theano import tensor
from subprocess import Popen, PIPE
from toolz import merge
from collections import Counter
from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta, CompositeRule)
from blocks.select import Selector
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions import SimpleExtension
from blocks.search import BeamSearch
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.filter import VariableFilter
from blocks.main_loop import MainLoop
from machine_translation.checkpoint import CheckpointNMT, LoadNMT
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import BleuValidator, Sampler
from machine_translation.stream import get_tr_stream, get_dev_stream
from machine_translation.model import BidirectionalEncoder, Decoder
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)


logger = logging.getLogger(__name__)








parser = argparse.ArgumentParser()
parser.add_argument("--proto",  default="get_config_cs2en",
                    help="Prototype config to use for config")
parser.add_argument("--bokeh",  default=False, action="store_true",
                    help="Use bokeh server for plotting")
args = parser.parse_args()


















def _get_attr_rec(self, obj, attr):
    return self._get_attr_rec(getattr(obj, attr), attr) \
        if hasattr(obj, attr) else obj

def ensure_special_tokens(vocab, bos_idx=0, eos_idx=0, unk_idx=1):
    """Ensures special tokens exist in the dictionary."""

    # remove tokens if they exist in some other index
    tokens_to_remove = [k for k, v in vocab.items()
                        if v in [bos_idx, eos_idx, unk_idx]]
    for token in tokens_to_remove:
        vocab.pop(token)
    # put corresponding item
    vocab['<S>'] = bos_idx
    vocab['</S>'] = eos_idx
    vocab['<UNK>'] = unk_idx
    return vocab






def oov_to_unk( seq, vocab_size, unk_idx):
    return [x if x < vocab_size else unk_idx for x in seq]


def idx_to_word( seq, ivocab):
    return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])





def main(config, tr_stream, dev_stream, use_bokeh=False):
    print("~def main")

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')
    sampling_input = tensor.lmatrix('input')

    print("~sampling_input = tensor.lmatrix")


    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2)
    cost = decoder.cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask)

    print("~source_sentence_mask, target_sentence, target_sentence_mask")

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    print("~ComputationGraph")

    # Initialize model
    logger.info('Initializing model')
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        config['weight_scale'])
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()


    print("~decoder.initialize()")



    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])



    print("~cg = apply_dropout")

    # Apply weight noise for regularization
    if config['weight_noise_ff'] > 0.0:
        logger.info('Applying weight noise to ff layers')
        enc_params = Selector(encoder.lookup).get_params().values()
        enc_params += Selector(encoder.fwd_fork).get_params().values()
        enc_params += Selector(encoder.back_fork).get_params().values()
        dec_params = Selector(
            decoder.sequence_generator.readout).get_params().values()
        dec_params += Selector(
            decoder.sequence_generator.fork).get_params().values()
        dec_params += Selector(decoder.state_init).get_params().values()
        cg = apply_noise(cg, enc_params+dec_params, config['weight_noise_ff'])


    print("~cg = apply_noise")

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
    logger.info("Total number of parameters: {}".format(len(shapes)))

    print("~logger.info")



    # Print parameter names
    enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                               Selector(decoder).get_parameters())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.items():
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info("Total number of parameters: {}"
                .format(len(enc_dec_param_dict)))

    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)
    print("~training_model")


    # Set extensions
    logger.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
        CheckpointNMT(config['saveto'],
                      every_n_batches=config['save_freq'])
    ]
    print("~every_n_batches=config")

    # Set up beam search and sampling computation graphs if necessary
    if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
        logger.info("Building sampling model")
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape))
        generated = decoder.generate(sampling_input, sampling_representation)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs

    sample = Sampler(model=search_model, data_stream=tr_stream,
                hook_samples=config['hook_samples'],
                every_n_batches=config['sampling_freq'],
                src_vocab_size=config['src_vocab_size'])

    # Add sampling
    if config['hook_samples'] >= 1:
        logger.info("Building sampler")
        extensions.append( sample )

    # Add early stopping based on bleu
    if config['bleu_script'] is not None:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(sampling_input, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot('Cs-En', channels=[['decoder_cost_cost']],
                 after_batch=True))










    sampling_fn = search_model.get_theano_function()



    print(" - - - - - - - - - - - - - - "  )


    sort_k_batches = 12
    batch_size = 80
    seq_len = 50
    trg_ivocab = None
    src_vocab_size = config['src_vocab_size']
    trg_vocab_size = config['trg_vocab_size']
    unk_id = config['unk_id'] 

    src_vocab = config['src_vocab']
    trg_vocab = config['trg_vocab']
    src_vocab = ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab)),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    trg_vocab = ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab)),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)
    if not trg_ivocab:
        trg_ivocab = {v: k for k, v in trg_vocab.items()}


    src_data = config['src_data']
    trg_data = config['trg_data']
    src_dataset = TextFile([src_data], src_vocab, None)
    trg_dataset = TextFile([trg_data], trg_vocab, None)




    inputstringfile="inputstringfile.cs"
    input_dataset = TextFile([inputstringfile], src_vocab, None)







    stream = Merge([input_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))
    stream2 = Filter(stream,
                    predicate=_too_long(seq_len=seq_len))
    stream3 = Mapping(stream2,
                     _oov_to_unk(src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))
    stream4 = Batch(stream3,
                   iteration_scheme=ConstantScheme(
                       batch_size*sort_k_batches))
                       
    stream5 = Mapping(stream4, SortMapping(_length))
    stream6 = Unpack(stream5)
    stream7 = Batch(
        stream6, iteration_scheme=ConstantScheme(batch_size))

    input_stream = DataStream(input_dataset)





    print("dev_stream : ", type( dev_stream )   )
    print("input_stream : ",  type( input_stream )   )






    epochone = input_stream.get_epoch_iterator() 
    vocab = input_stream.dataset.dictionary
    unk_sym = input_stream.dataset.unk_token
    eos_sym = input_stream.dataset.eos_token




    for i, line in enumerate(epochone):
        seq = oov_to_unk(
            line[0], config['src_vocab_size'], unk_id)
        input_ = numpy.tile(seq, ( 1 , 1))


        print("seq : " ,   type( seq )  ,  seq   )
        print("input_ : ", type( input_ )  , input_ ,  inspect.getmembers( input_ )    )



        _1, outputs, _2, _3, costs = ( sampling_fn(  input_  ) )

        outputs = outputs.flatten()
        costs = costs.T

        print(" outputs : "    ,   outputs   ,   type( outputs )  )
        print("idx_to_word: ", idx_to_word(outputs  ,  trg_ivocab))












    print(" - - - - - - - - - - - - - - "  )
#.....



































































































































def _length(sentence_pair):
    """Assumes target is the last element in the tuple."""
    return len(sentence_pair[-1])



class _too_long(object):
    """Filters sequences longer than given sequence length."""
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all([len(sentence) <= self.seq_len
                    for sentence in sentence_pair])





class _oov_to_unk(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self, src_vocab_size=30000, trg_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.unk_id = unk_id

    def __call__(self, sentence_pair):
        return ([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[0]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[1]])






























class SamplingBase(object):
    """Utility class for BleuValidator and Sampler."""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])


class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()























    def do(self, which_callback, *args):
        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Sample
        print()
        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            _1, outputs, _2, _3, costs = (self.sampling_fn(inp[None, :]))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)

            print("Input : ", self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab))
            print("Target: ", self._idx_to_word(target_[i][:target_length],
                                                self.trg_ivocab))
            print("Sample: ", self._idx_to_word(outputs[:sample_length],
                                                self.trg_ivocab))
            print("Sample cost: ", costs[:sample_length].sum())
            print()





















class BleuValidator(SimpleExtension, SamplingBase):
    # TODO: a lot has been changed in NMT, sync respectively
    """Implements early stopping based on BLEU score."""

    def __init__(self, source_sentence, samples, model, data_stream,
                 config, n_best=1, track_n_models=1, trg_ivocab=None,
                 normalize=True, **kwargs):
        # TODO: change config structure
        super(BleuValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = config.get('val_set_out', None)

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.trg_ivocab = trg_ivocab
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(samples=samples)
        self.multibleu_cmd = ['perl', self.config['bleu_script'],
                              self.config['val_set_grndtruth'], '<']

        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                bleu_score = numpy.load(os.path.join(self.config['saveto'],
                                        'val_bleu_scores.npz'))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted(self.val_bleu_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu))
                logger.info("BleuScores Reloaded")
            except:
                logger.info("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= \
                self.config['val_burn_in']:
            return

        # Evaluate and save if necessary
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):

        logger.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        # Get target vocabulary
        if not self.trg_ivocab:
            sources = self._get_attr_rec(self.main_loop, 'data_stream')
            trg_vocab = sources.data_streams[1].dataset.dictionary
            self.trg_ivocab = {v: k for k, v in trg_vocab.items()}

        if self.verbose:
            ftrans = open(self.config['val_set_out'], 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = self._oov_to_unk(
                line[0], self.config['src_vocab_size'], self.unk_idx)
            input_ = numpy.tile(seq, (self.config['beam_size'], 1))

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.source_sentence: input_},
                    max_length=3*len(seq), eol_symbol=self.eos_idx,
                    ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    trans_out = self._idx_to_word(trans_out, self.trg_ivocab)

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print(trans_out, file=mb_subprocess.stdin)
                    if self.verbose:
                        print(trans_out, file=ftrans)

            if i != 0 and i % 100 == 0:
                logger.info(
                    "Translated {} lines of validation set...".format(i))

            mb_subprocess.stdin.flush()

        logger.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        logger.info(bleu_score)
        mb_subprocess.terminate()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.config['saveto'])

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
            numpy.savez(
                model.path, **self.main_loop.model.get_parameter_dict())
            numpy.savez(
                os.path.join(self.config['saveto'], 'val_bleu_scores.npz'),
                bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)


class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, bleu_score, path=None):
        self.bleu_score = bleu_score
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_bleu_model_%d_BLEU%.2f.npz' %
            (int(time.time()), self.bleu_score) if path else None)
        return gen_path




























































if __name__ == "__main__":
    print("~__main__")

    # Get configurations for model
    configuration = getattr(configurations, args.proto)()
    print("~configuration")

    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
    print("~logger.info")

    # Get data streams and call main
    main(configuration, get_tr_stream(**configuration),
         get_dev_stream(**configuration), args.bokeh)







