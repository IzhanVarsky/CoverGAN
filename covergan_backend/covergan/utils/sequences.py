import torch
import torch.nn.utils.rnn as rnn_utils


def clean_pad_matrix(t: torch.Tensor, t_seq_lengths: torch.Tensor, t_limit: int) -> torch.Tensor:
    """
    Given batch-first sequence matrix with sequence lengths, remove any trailing data and pad
    to the specified limit with zeros.
    :param t: batch-first sequence matrix
    :param t_seq_lengths: sequence lengths
    :param t_limit: padding limit, at least as big as the maximum sequence length
    :return: 0-padded batch-fist sequence tensor
    """
    # `pack_padded_sequence` cannot accept 0-length sequences
    clamped_lengths = torch.clamp(t_seq_lengths, min=1)

    packed = rnn_utils.pack_padded_sequence(t, clamped_lengths.cpu(), batch_first=True, enforce_sorted=False)
    clean = rnn_utils.pad_packed_sequence(packed, batch_first=True)[0]

    assert clean.shape[1] <= t_limit
    if clean.shape[1] < t_limit:
        zeros = torch.zeros(len(clean), t_limit - clean.shape[1], device=clean.device)
        clean = torch.cat((clean, zeros), dim=1)

    assert clean.shape == torch.Size([len(t), t_limit])
    return clean


def to_packed_with_one_hot(vectors: torch.Tensor, vector_lengths: torch.Tensor, one_hot_labels: torch.Tensor):
    assert len(vectors) == len(vector_lengths)
    fake_seq_lengths, perm_idx = vector_lengths.sort(0, descending=True)
    vectors = vectors[perm_idx]
    perm_one_hot_labels = one_hot_labels[perm_idx]

    vectors = vectors.reshape(vectors.shape[0], vectors.shape[1], 1)
    perm_one_hot_labels = perm_one_hot_labels.repeat(1, vectors.shape[1])
    perm_one_hot_labels = perm_one_hot_labels.reshape(vectors.shape[0], vectors.shape[1], -1)
    vectors = torch.cat((vectors, perm_one_hot_labels), dim=2)

    # `pack_padded_sequence` cannot accept 0-length sequences (and they are not valid anyway)
    fake_seq_lengths = torch.clamp(fake_seq_lengths, min=1)
    packed_input = rnn_utils.pack_padded_sequence(vectors, fake_seq_lengths.cpu(), batch_first=True)

    return packed_input
