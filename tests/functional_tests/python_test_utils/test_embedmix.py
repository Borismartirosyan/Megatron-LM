import pytest
import torch
from megatron.core.models.gpt.gpt_model import GPTModel  # Import the class where embedmix_augment is defined





class TestEmbedmix:

    @pytest.mark.unit
    def test_embedmix_augment_no_perturb(your_class_instance):
        input_tensor = torch.randn(144, 1, 2048)  # Example input tensor with batch size 1
        output_tensor = GPTModel.embedmix_augment(input_tensor, 0, 0, 0)
        assert torch.allclose(input_tensor, output_tensor)

    @pytest.mark.unit
    def test_embedmix_augment_addition(your_class_instance):
        input_tensor = torch.randn(144, 1, 2048)  # Example input tensor with batch size 3
        output_tensor = GPTModel.embedmix_augment(input_tensor, 1, 0.5, 0.5, augment_type='addition')
        assert output_tensor.shape == input_tensor.shape
        assert (output_tensor != input_tensor).any()

    @pytest.mark.unit
    def test_embedmix_augment_swap(your_class_instance):
        input_tensor = torch.randn(144, 1, 2048)  # Example input tensor with batch size 3
        output_tensor = GPTModel.embedmix_augment(input_tensor, 1, 0.5, 0.5, augment_type='swap')
        assert output_tensor.shape == input_tensor.shape
        assert (output_tensor != input_tensor).any()
