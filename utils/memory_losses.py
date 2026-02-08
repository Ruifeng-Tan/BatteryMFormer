"""
Memory-related losses for MemoryNet2
1. ValueMemoryLoss: Ensures value memory can reconstruct ground truth embeddings
2. DiversityLoss: Encourages diversity in value memory prototypes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueMemoryLoss(nn.Module):
    """
    Value Memory Reconstruction Loss
    Ensures value memory can effectively reconstruct ground truth embeddings
    """
    def __init__(self):
        super(ValueMemoryLoss, self).__init__()

    def forward(self, alpha_val, value_memory, gt_embedding):
        """
        Compute value memory reconstruction loss

        Args:
            alpha_val: [B, S] value-side attention weights
            value_memory: [S, D] value memory
            gt_embedding: [B, D] ground truth trajectory embeddings

        Returns:
            loss: scalar reconstruction loss
        """
        # Reconstruct embedding using value-side weights
        # reconstructed = sum(alpha_val * value_memory) = alpha_val @ value_memory
        reconstructed = torch.matmul(alpha_val, value_memory)  # [B, D]

        # MSE loss
        loss = F.mse_loss(reconstructed, gt_embedding)

        return loss


class DiversityLoss(nn.Module):
    """
    Diversity Loss
    Encourages value memory prototypes to be diverse (orthogonal)
    """
    def __init__(self):
        super(DiversityLoss, self).__init__()

    def forward(self, value_memory):
        """
        Compute diversity loss

        Args:
            value_memory: [S, D] value memory

        Returns:
            loss: scalar diversity loss
        """
        # Normalize value memory rows
        value_memory_norm = F.normalize(value_memory, p=2, dim=-1)  # [S, D]

        # Compute Gram matrix (similarity matrix)
        # G = value_memory_norm @ value_memory_norm^T
        gram = torch.matmul(value_memory_norm, value_memory_norm.t())  # [S, S]

        # Ideal Gram matrix is identity (orthogonal prototypes)
        identity = torch.eye(value_memory.size(0), device=value_memory.device)

        # Frobenius norm of difference
        loss = F.mse_loss(gram, identity)

        return loss


if __name__ == "__main__":
    # Test ValueMemoryLoss
    print("Test ValueMemoryLoss:")
    batch_size = 4
    num_slots = 128
    embed_dim = 256

    # Create dummy data
    alpha_val = F.softmax(torch.randn(batch_size, num_slots), dim=-1)
    value_memory = torch.randn(num_slots, embed_dim)
    gt_embedding = torch.randn(batch_size, embed_dim)

    val_mem_loss = ValueMemoryLoss()
    loss = val_mem_loss(alpha_val, value_memory, gt_embedding)
    print(f"Value memory loss: {loss.item():.6f}")

    # Test DiversityLoss
    print("\nTest DiversityLoss:")

    # Test with random memory
    print("Random memory:")
    diversity_loss = DiversityLoss()
    loss_random = diversity_loss(value_memory)
    print(f"Diversity loss (random): {loss_random.item():.6f}")

    # Test with orthogonal memory
    print("\nOrthogonal memory:")
    # Create orthogonal matrix using QR decomposition
    random_matrix = torch.randn(num_slots, embed_dim)
    q, r = torch.qr(random_matrix)
    orthogonal_memory = q[:num_slots, :] if embed_dim >= num_slots else q
    loss_ortho = diversity_loss(orthogonal_memory)
    print(f"Diversity loss (orthogonal): {loss_ortho.item():.6f}")

    # Test with identical memory (worst case)
    print("\nIdentical memory:")
    identical_memory = torch.ones(num_slots, embed_dim)
    loss_identical = diversity_loss(identical_memory)
    print(f"Diversity loss (identical): {loss_identical.item():.6f}")

    print("\nExpected: orthogonal < random < identical")
