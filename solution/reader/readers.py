import argparse

from datasets import Dataset

from solution.trainers import QuestionAnsweringTrainer, QuestionAnsweringSeq2SeqTrainer

from solution.utils import compute_metrics

from solution.reader.core import ReaderBase
from solution.utils.postprocessing import post_processing_function
from solution.utils.preprocessing import ext_prepare_features, gen_prepare_features


class ExtractiveReader(ReaderBase):
    """ Base class for Extractive Reader module """
    def __init__(self, data_args, training_args, model_args,
                 compute_metrics=compute_metrics,
                 pre_process_function=ext_prepare_features,
                 post_process_function=post_processing_function,
                 logger=None,):
        super(ExtractiveReader, self).__init__(data_args, training_args, model_args)
        self.logger = logger
        self._set_initial_setup()
        self.pre_process_function = pre_process_function
        self._set_preprocessing()
        self.post_process_function = post_process_function
        self.compute_metrics = compute_metrics
        self.trainer = None

    def set_trainer(self):
        """ Set up the Trainer """

        eval_dataset = self.eval_dataset
        eval_samples = self.datasets['validation']
        
        """ Set Trainer """
        self.trainer = QuestionAnsweringTrainer( 
                            model=self.model,
                            args=self.args.training_args,
                            train_dataset=\
                                self.train_dataset if self.args.training_args.do_train else None,
                            eval_dataset=\
                                eval_dataset if self.args.training_args.do_eval \
                                    or self.args.training_args.do_predict else None,
                            eval_examples=\
                                eval_samples if self.args.training_args.do_eval \
                                    or self.args.training_args.do_predict else None,
                            tokenizer=self.tokenizer,
                            data_collator=self.data_collator,
                            post_process_function=self.post_process_function,
                            compute_metrics=self.compute_metrics,
                            )

        self.logger.info("*** Set up the Trainer ***")

        return self.trainer
    
    def train(self, *args, **kwargs):
        """ Call train method of self.trainer """
        return self.trainer.train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """ Call evaluate method of self.trainer """
        return self.trainer.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """ Call predict method of self.trainer """
        return self.trainer.predict(*args, **kwargs)

class GenerativeReader(ReaderBase):
    """ Base class for Generative Reader module """
    def __init__(self, data_args, training_args, model_args,
                 compute_metrics=compute_metrics,
                 pre_process_function=gen_prepare_features,
                 post_process_function=post_processing_function,
                 logger=None,):
        super(GenerativeReader, self).__init__(data_args, training_args, model_args)
        self._set_initial_setup()
        self.pre_process_function = pre_process_function
        self._set_preprocessing()
        self.post_process_function = post_process_function
        self.compute_metrics = compute_metrics
        self.trainer = None

    def set_trainer(self, retrieved_dataset:Dataset=None):
        """ Set up the Trainer """

        eval_dataset = self.eval_dataset
        eval_samples = self.datasets['validation']
        # Retireved Dataset이 Predict를 위해 주어졌을 때, 기존 저장된 eval_dataset과 swap
        if retrieved_dataset is not None:
            eval_dataset = self.preprocessing_retrieved_doc(retrieved_dataset)
            eval_samples = retrieved_dataset['validation']

        """ Set Trainer """
        self.trainer = QuestionAnsweringSeq2SeqTrainer( 
                            model=self.model,
                            args=self.args.training_args,
                            train_dataset=\
                                self.train_dataset if self.args.training_args.do_train else None,
                            eval_dataset=\
                                eval_dataset if self.args.training_args.do_eval \
                                    or self.args.training_args.do_predict else None,
                            eval_examples=\
                                eval_samples if self.args.training_args.do_eval \
                                    or self.args.training_args.do_predict else None,
                            tokenizer=self.tokenizer,
                            data_collator=self.data_collator,
                            post_process_function=self.post_processing_function,
                            compute_metrics=self.compute_metric,
                            )

        self.logger.info("*** Set up the Trainer ***")

        return self.trainer

    def train(self, *args, **kwargs):
        """ Call train method of self.trainer """
        return self.trainer.train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """ Call evaluate method of self.trainer """
        return self.trainer.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """ Call predict method of self.trainer """
        return self.trainer.predict(*args, **kwargs)