# kepler ps
# unit testing script

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from score_sentiment.py import model_scores, polarity


class TestSentimentModel(unittest.TestCase):
    @unittest.expectedFailure
    @patch("your_script_name.AutoTokenizer.from_pretrained")
    @patch("your_script_name.AutoModelForSequenceClassification.from_pretrained")
    # tests whether raw scores are formatted correctly
    def test_model_scores(self, mock_model, mock_tokenizer):
        try:
            # mocking the tokenizer
            mock_tokenizer.return_value = MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    
            # mocking the model output
            mock_output = MagicMock()
            mock_output[0][0].detach.return_value.numpy.return_value = np.array([1.0, 2.0, 3.0])
            mock_model.return_value = mock_output
    
            text = "I love this"
            raw_scores = model_scores(text, mock_model())
    
            # check if the output is as expected
            expected_scores = np.array([1.0, 2.0, 3.0])  # This should match the mock output
            np.testing.assert_array_equal(raw_scores, expected_scores)
        except FileNotFoundError:

    # testing processing of raw scores 
    def test_polarity(self):
        raw_scores = np.array([1.0, 2.0, 3.0])
        expected_polarity = np.sum(np.round(softmax(raw_scores), 2) * torch.tensor([-1, 0, 1]))

        polarity_result = polarity(raw_scores)

        # check if polarity is correct
        self.assertAlmostEqual(polarity_result, expected_polarity.numpy(), places=5)


# run the tests
if __name__ == "__main__":
    unittest.main()

