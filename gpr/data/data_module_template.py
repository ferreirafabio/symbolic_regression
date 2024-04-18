



class AbstractDataModule():

    def __init__(self, num_variables: int, set_binary_op: dict, set_uni_op: dict, num_realisations: int, val_samples: int):
        """
        :param num_variables: int, the maximum numer of variables in the equation
        :param set_binary_op: dict, a dictionary of binary operations
        :param set_uni_op: dict, a dictionary of unary operations
        :param num_realisations: int; the number of concrete input values for each equation
        :param val_samples: int
        """
        # TODO: treat NaNs
        pass

        self.ignore_index = -100
        self.pad_index = 0
        self.create_validation_set()


    @abstractmethod
    def create_sample(self, rng=None):
        # return a tbl of n inference samples of the equation, and the target token sequnce of the latex equation
        pass

    @abstractmethod
    def get_vocab(self):
        # return the vocabulary of the equation tokens
        return ["1", "2", ... "sin", "cos", ..., "<EOS>"] # tokens for the late equation

    def get_vocab_size(self):
        return len(self.get_vocab())

    def create_validation_set(self):
        pass


    def latex_equation_to_function(self, latex_equation):
        pass

    def check_if_latex_equation_is_valid(self):
        return True

    @staticmethod
    def collator(samples):
        # get a set of samples
        # return a batch for tbl and trg_tex
        # pad target sequence with ignore index and input table with pad index
        return batch_tbl_significant, batch_tbl_exponent, batch_equation_tokens

    def get_train_dataloader(self, batch_size, num_workers):

        # return a dataloader over an infinite set of training data
        pass

    def get_valid_dataloder(self):

        # return a dataloader over a finite set of validation data, which was created in the create_validation_set method
        pass


class SimPySimpleDataModule(AbstractDataModule):
    def __init__(self, num_variables, set_binary_op, set_uni_op, num_realisations, val_samples):
        super().__init__(num_variables, set_binary_op, set_uni_op, num_realisations, val_samples)


if __name__ == "__main__":
    pass
