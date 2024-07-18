import unittest
import torch
from torch.utils.data import DataLoader
from gpr.data.generators import PolynomialGenerator, RandomGenerator
from gpr.data.utils import characters
from gpr.data.datasets import EquationDataset
from gpr.utils.configuration import Config
import sympy as sp
from sympy import Eq, symbols


from gpr.data.loaders import SymPySimpleDataModule

class TestSymPySimpleDataModule(unittest.TestCase):

    def setUp(self):
        self.config_path = 'config/default_config.yaml'
        self.generator = PolynomialGenerator
        self.data_module = SymPySimpleDataModule(generator=self.generator, config_path=self.config_path)
        self.train_loader = self.data_module.get_train_loader()
        self.valid_loader = self.data_module.get_valid_loader()

    def test_expansion_and_factorization(self):
        """Test if y = a^2 + 2ab + b^2 gets transformed into y = (a + b)^2 and vice versa"""
        a, b = symbols('a b')

        # Test expansion to factorization
        original_expanded_eq = Eq(a**2 + 2*a*b + b**2, 0)
        factored_eq = Eq(sp.factor(original_expanded_eq.lhs), 0)
        expected_factored_eq = Eq((a + b)**2, 0)
        
        self.assertTrue(sp.simplify(factored_eq.lhs - expected_factored_eq.lhs) == 0, 
                        f"Expected {expected_factored_eq}, but got {factored_eq}")

        # Test factorization to expansion
        original_factored_eq = Eq((a + b)**2, 0)
        expanded_eq = Eq(sp.expand(original_factored_eq.lhs), 0)
        expected_expanded_eq = Eq(a**2 + 2*a*b + b**2, 0)
        
        self.assertTrue(sp.simplify(expanded_eq.lhs - expected_expanded_eq.lhs) == 0, 
                        f"Expected {expected_expanded_eq}, but got {expanded_eq}")

    def test_collator_output(self):
        """Test the output of the collator function."""
        # Create sample tensors and a dummy equation
        a, b = symbols('a b')
        mantissa = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        exponent = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        equation = Eq(a**2 + 2*a*b + b**2, 0)
        
        # Create a batch of tuples
        batch = [(mantissa, exponent, torch.tensor([ord(char) for char in str(equation)]), equation)]
        
        # Ensure batch items are of expected format
        for item in batch:
            self.assertIsInstance(item[0], torch.Tensor)
            self.assertIsInstance(item[1], torch.Tensor)
            self.assertIsInstance(item[2], torch.Tensor)
            self.assertIsInstance(item[3], Eq)
        
        collated_batch = self.data_module.collator(batch)
        
        # Check if collated batch has the correct keys and types
        self.assertIn('mantissa', collated_batch)
        self.assertIn('exponent', collated_batch)
        self.assertIn('latex_token', collated_batch)
        self.assertIn('equation', collated_batch)
        self.assertIsInstance(collated_batch['mantissa'], torch.Tensor)
        self.assertIsInstance(collated_batch['exponent'], torch.Tensor)
        self.assertIsInstance(collated_batch['latex_token'], torch.Tensor)
        self.assertIsInstance(collated_batch['equation'], list)
        self.assertIsInstance(collated_batch['equation'][0], Eq)

        # Check if collated batch has the correct keys and types
        self.assertIn('mantissa', collated_batch)
        self.assertIn('exponent', collated_batch)
        self.assertIn('latex_token', collated_batch)
        self.assertIn('equation', collated_batch)
        self.assertIsInstance(collated_batch['mantissa'], torch.Tensor)
        self.assertIsInstance(collated_batch['exponent'], torch.Tensor)
        self.assertIsInstance(collated_batch['latex_token'], torch.Tensor)
        self.assertIsInstance(collated_batch['equation'], list)

    def test_seeded_equation_generation(self):
        """Test that equations generated with the same seed are identical."""
        # Instantiate the first data module and train loader
        data_module_1 = SymPySimpleDataModule(generator=self.generator, config_path=self.config_path)
        train_loader_1 = data_module_1.get_train_loader()

        # Instantiate the second data module and train loader
        data_module_2 = SymPySimpleDataModule(generator=self.generator, config_path=self.config_path)
        train_loader_2 = data_module_2.get_train_loader()

        # Get the first batch from both train loaders
        batch_1 = next(iter(train_loader_1))
        batch_2 = next(iter(train_loader_2))

        # Compare the first equation from both batches
        equation_1 = batch_1['equation'][0]
        equation_2 = batch_2['equation'][0]

        self.assertEqual(equation_1, equation_2, f"Expected equations to be the same but got {equation_1} and {equation_2}")



    def test_train_loader_not_empty(self):
        """Test that the training loader is not empty."""
        try:
            for batch in self.train_loader:
                self.assertTrue(len(batch['equation']) > 0)
                break
        except AttributeError as e:
            self.fail(f"AttributeError occurred: {e}")

    def test_valid_loader_not_empty(self):
        """Test that the validation loader is not empty."""
        try:
            for batch in self.valid_loader:
                self.assertTrue(len(batch['equation']) > 0)
                break
        except AttributeError as e:
            self.fail(f"AttributeError occurred: {e}")

    def test_equation_format(self):
        """Test that the equations are of the expected format."""
        for batch in self.valid_loader:
            for equation in batch['equation']:
                self.assertIsInstance(equation, Eq)
            break

    def test_mantissa_and_exponent_shape(self):
        """Test that the mantissa and exponent tensors have the correct shapes."""
        for batch in self.valid_loader:
            mantissa = batch['mantissa']
            exponent = batch['exponent']
            self.assertEqual(mantissa.shape, exponent.shape)
            break

    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        invalid_config_path = 'config/invalid_config.yaml'
        with self.assertRaises(Exception):
            invalid_data_module = SymPySimpleDataModule(generator=self.generator, config_path=invalid_config_path)
            invalid_data_module.get_train_loader()

    def test_vocab_size(self):
        """Test that the vocab size is as expected."""
        expected_vocab_size = len(characters)
        self.assertEqual(self.data_module.vocab_size, expected_vocab_size)

    def test_worker_init(self):
        """Test that the worker initialization function sets seeds correctly."""
        worker_id = 0
        self.data_module.worker_init_fn(worker_id)
        self.assertEqual(self.data_module.worker_seeds[worker_id], self.data_module.config.generator.seed + worker_id)

    def test_create_sample(self):
        """Test the creation of a sample."""
        sample = self.data_module.create_sample()
        self.assertIsInstance(sample, tuple)
        self.assertEqual(len(sample), 3)
        self.assertIsInstance(sample[0], torch.Tensor)
        self.assertIsInstance(sample[1], torch.Tensor)
        self.assertIsInstance(sample[2], Eq)

if __name__ == "__main__":
    unittest.main()
