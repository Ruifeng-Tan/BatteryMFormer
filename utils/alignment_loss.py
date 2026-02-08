"""
Alignment Loss for MemoryNet2
Uses Jensen-Shannon (JS) divergence to align key-side and value-side addressing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_value_side_addressing(gt_embedding, value_memory, temperature=0.1):
    """
    Compute value-side addressing weights
    Uses cosine similarity between ground truth embedding and value memory

    Args:
        gt_embedding: [B, D] ground truth trajectory embeddings
        value_memory: [S, D] value memory
        temperature: Temperature for softmax (τ)

    Returns:
        weights: [B, S] value-side attention weights
    """
    # Normalize for cosine similarity
    gt_embedding_norm = F.normalize(gt_embedding, p=2, dim=-1)  # [B, D]
    value_memory_norm = F.normalize(value_memory, p=2, dim=-1)  # [S, D]

    # Compute cosine similarity: [B, D] @ [D, S] = [B, S]
    similarity = torch.matmul(gt_embedding_norm, value_memory_norm.t())  # [B, S]

    # Apply temperature and softmax
    weights = F.softmax(similarity / temperature, dim=-1)  # [B, S]

    return weights


def kl_divergence(p, q):
    """
    Compute KL divergence KL(p || q)
    Args:
        p: [B, S] probability distribution
        q: [B, S] probability distribution
    Returns:
        kl: [B] KL divergence for each sample
    """
    # Add small epsilon for numerical stability
    epsilon = 1e-8
    p = p + epsilon
    q = q + epsilon

    # KL(p || q) = sum(p * log(p / q))
    kl = torch.sum(p * torch.log(p / q), dim=-1)

    return kl


def js_divergence(p, q):
    """
    Compute Jensen-Shannon divergence JS(p || q)
    JS(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
    where m = 0.5 * (p + q)

    Args:
        p: [B, S] probability distribution
        q: [B, S] probability distribution (with stop gradient)

    Returns:
        js: [B] JS divergence for each sample
    """
    # Compute mixture distribution
    m = 0.5 * (p + q)

    # Compute KL divergences
    kl_p_m = kl_divergence(p, m)
    kl_q_m = kl_divergence(q, m)

    # JS divergence
    js = 0.5 * kl_p_m + 0.5 * kl_q_m

    return js


class AlignmentLoss(nn.Module):
    """
    Alignment Loss using Jensen-Shannon divergence
    Aligns key-side and value-side addressing distributions
    """
    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: Temperature for value-side addressing softmax (τ)
        """
        super(AlignmentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, alpha_key, gt_embedding, value_memory):
        """
        Compute alignment loss

        Args:
            alpha_key: [B, S] key-side attention weights (from MemoryNetPath)
            gt_embedding: [B, D] ground truth trajectory embeddings
            value_memory: [S, D] value memory

        Returns:
            loss: scalar alignment loss
        """
        # Compute value-side addressing weights
        alpha_val = compute_value_side_addressing(
            gt_embedding, value_memory, temperature=self.temperature
        )  # [B, S]

        # Stop gradient on alpha_val to prevent gradient flow to value memory
        # through this path
        alpha_val_sg = alpha_val.detach()

        # Compute JS divergence
        js = js_divergence(alpha_key, alpha_val_sg)  # [B]

        # Return mean loss
        loss = js.mean()

        return loss, alpha_val


if __name__ == "__main__":
    # Test alignment loss
    batch_size = 4
    num_slots = 128
    embed_dim = 256

    # Create dummy data
    alpha_key = F.softmax(torch.randn(batch_size, num_slots), dim=-1)
    gt_embedding = torch.randn(batch_size, embed_dim)
    value_memory = torch.randn(num_slots, embed_dim)

    # Test value-side addressing
    alpha_val = compute_value_side_addressing(gt_embedding, value_memory)
    print(f"Value-side weights shape: {alpha_val.shape}")
    print(f"Value-side weights sum: {alpha_val.sum(dim=-1)}")

    # Test KL divergence
    p = F.softmax(torch.randn(batch_size, num_slots), dim=-1)
    q = F.softmax(torch.randn(batch_size, num_slots), dim=-1)
    kl = kl_divergence(p, q)
    print(f"\nKL divergence shape: {kl.shape}")
    print(f"KL divergence values: {kl}")

    # Test JS divergence
    js = js_divergence(p, q)
    print(f"\nJS divergence shape: {js.shape}")
    print(f"JS divergence values: {js}")

    # Test AlignmentLoss
    alignment_loss = AlignmentLoss(temperature=0.1)
    loss, alpha_val = alignment_loss(alpha_key, gt_embedding, value_memory)
    print(f"\nAlignment loss: {loss.item():.6f}")

    # Test that JS divergence is symmetric
    js_pq = js_divergence(p, q)
    js_qp = js_divergence(q, p)
    print(f"\nJS(p||q): {js_pq.mean():.6f}")
    print(f"JS(q||p): {js_qp.mean():.6f}")
    print(f"Symmetric: {torch.allclose(js_pq, js_qp, atol=1e-5)}")
