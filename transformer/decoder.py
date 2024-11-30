from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding1D(nn.Module):
    """
    1D positional encoding for transformer decoders.

    Args:
        max_len (int): The maximum sequence length.
        emb_dim (int): The embedding dimension.
        dropout_p (float, optional): The dropout probability. Defaults to 0.1.
    """

    def __init__(self, max_len: int, emb_dim: int, dropout_p: float = 0.1):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        pos = torch.arange(max_len).unsqueeze(1)
        den = torch.pow(10000, torch.arange(0, emb_dim, 2) / emb_dim)

        pe = torch.zeros(1, max_len, emb_dim)
        pe[0, :, 0::2] = torch.sin(pos / den)
        pe[0, :, 1::2] = torch.cos(pos / den)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch_size, sec_len, emb_dim]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Decoder(nn.Module):
    """
    Transformer decoder with 1D positional encoding.
    The final classification layer is a 1D convolution.

    Args:
        output_size (int): The number of classes to predict.
        max_seq_len (int): The maximum sequence length.
        num_embeddings (int): The number of embeddings in the vocabulary.
        embedding_dim (int, optional): The embedding dimension. Defaults to 256.
        padding_idx (int, optional): The padding index. Defaults to 0.
        ff_dim (int, optional): The feedforward dimension. Defaults to 256.
        dropout_p (float, optional): The dropout probability. Defaults to 0.1.
        nhead (int, optional): The number of attention heads. Defaults to 4.
        num_transformer_layers (int, optional): The number of transformer layers. Defaults to 8.
        attn_window (int, optional): The attention window size. Defaults to -1 (attends to all past tokens).
    """

    def __init__(
        self,
        # Classification layer
        output_size: int,
        # PE
        max_seq_len: int,
        # Embedding
        num_embeddings: int,
        embedding_dim: int = 256,
        padding_idx: int = 0,
        # Transformer
        ff_dim: int = 256,
        dropout_p: float = 0.1,
        nhead: int = 4,
        num_transformer_layers: int = 8,
        attn_window: int = -1,  # -1 means "no limit"
    ):
        super(Decoder, self).__init__()

        # Input block
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.pos_1d = PositionalEncoding1D(
            max_len=max_seq_len,
            emb_dim=embedding_dim,
            dropout_p=dropout_p,
        )

        # Transformer block
        self.attn_window = attn_window
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout_p,
                batch_first=True,
            ),
            num_layers=num_transformer_layers,
        )

        # Output/classification block
        self.out_layer = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=output_size,
            kernel_size=1,
        )

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor, memory_len: torch.Tensor
    ):
        """
        Forward pass of the transformer decoder.

        Args:
            tgt (torch.Tensor): The target sequence. Shape: [batch_size, tgt_sec_len].
            memory (torch.Tensor): The encoder output with positional encoding. Shape: [batch_size, src_sec_len, emb_dim].
            memory_len (torch.Tensor): The actual length of each encoder output. Shape: [batch_size].

        Returns:
            torch.Tensor: The predicted target sequence. Shape: [batch_size, output_size, tgt_sec_len].
        """
        # memory is the output of the encoder with the 2D PE added, flattened and permuted
        # memory.shape = [batch_size, src_sec_len, emb_dim]
        # src_sec_len = h * w (IMAGE/SPECTROGRAM UNFOLDING); emb_dim = out channels from encoder

        # tgt is the target sequence shifted to the right
        # tgt.shape = [batch_size, tgt_sec_len]

        # Embedding + 1D PE
        tgt_emb = self.pos_1d(
            self.embedding(tgt)
        )  # tgt_emb.shape = [batch_size, tgt_sec_len, emb_dim]

        # Get memory key padding mask
        # Ignore padding in the encoder output
        memory_key_padding_mask = self.get_memory_key_padding_mask(memory, memory_len)

        # Get tgt masks
        tgt_mask, tgt_key_padding_mask = self.get_tgt_masks(tgt)
        tgt_key_padding_mask = (
            None if memory_key_padding_mask is None else tgt_key_padding_mask
        )  # memory_key_padding_mask is None during inference

        # Transformer decoder
        tgt_pred = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None,  # We let it see the whole encoder output
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # tgt_pred.shape = [batch_size, tgt_sec_len, emb_dim]

        # Classification block
        tgt_pred = tgt_pred.permute(0, 2, 1).contiguous()
        tgt_pred = self.out_layer(
            tgt_pred
        )  # tgt_pred.shape = [batch_size, output_size, tgt_sec_len]

        return tgt_pred

    def get_memory_key_padding_mask(
        self, memory: torch.Tensor, memory_len: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Generates the memory key padding mask needed for the transformer decoder forward pass.
        The mask is used to ignore padding in the encoder output.

        Args:
            memory (torch.Tensor): The encoder output. Shape: [batch_size, src_sec_len, emb_dim].
            memory_len (Optional[torch.Tensor], optional): The actual length of each encoder output. Shape: [batch_size]. Defaults to None.

        Returns:
            Optional[torch.Tensor]: The generated memory key padding mask.
        """
        if memory_len is None:
            # During inference, the encoder output is not padded
            # We perform inference one sample at a time
            return None

        # When using batches, the spectrograms are padded to the same length
        # We need to mask the padding so the attention mechanism ignores it

        # memory.shape = [batch_size, src_sec_len, emb_dim]
        # memory_len.shape = [batch_size]
        # memory_pad_mask.shape = [batch_size, src_sec_len]
        # Value 1 (True) means "ignored" and value 0 (False) means "not ignored"
        memory_pad_mask = torch.zeros(
            memory.shape[:2], dtype=torch.bool, device=memory.device
        )
        for i, l in enumerate(memory_len):
            memory_pad_mask[i, l:] = True
        return memory_pad_mask

    @staticmethod
    def create_variable_window_mask(
        size: int,
        window_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Creates a mask for the target sequence with a variable window size.

        Args:
            size (int): The size of the target sequence.
            window_size (int): The size of the window to focus on the last X tokens.
            dtype (torch.dtype, optional): The data type of the mask. Defaults to torch.float32.
            device (torch.device, optional): The device to store the mask. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: The generated mask.
        """
        mask = torch.full((size, size), float("-inf"), dtype=dtype, device=device)
        for i in range(size):
            if window_size < size:
                start = max(0, i - window_size)
                mask[i, start : i + 1] = 0
            else:
                mask[i, : i + 1] = 0
        return mask

    def get_tgt_masks(self, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the target mask and the target padding mask needed for the transformer decoder forward pass.
            - Target mask is an upper triangular matrix with the masked positions filled with float('-inf'). We only let the decoder see the past.
            Since it is a FloatTensor, it will be added to the attention weights

            - Target padding mask is a binary mask with value 1 (True) in the positions where the target sequence is padded.
            We want the attention mechanism to ignore these positions.

        Args:
            tgt (torch.Tensor): The target sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target mask and the target padding mask.
        """
        # tgt.shape = [batch_size, tgt_sec_len]
        tgt_sec_len = tgt.shape[1]

        # Target = Decoder (we only let it see the past)
        # Upper triangular matrix of size (tgt_sec_len, tgt_sec_len)
        # The masked positions are filled with float('-inf')
        # Unmasked positions are filled with float(0.0)

        # ATTENTION WINDOW MECHANISM
        # We limit the number of past tokens the decoder can see
        if self.attn_window > 0:
            tgt_mask = self.create_variable_window_mask(
                tgt_sec_len, self.attn_window, device=tgt.device
            )
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_sec_len, tgt.device
            )

        # 0 == "<PAD>"
        # Pad token to be ignored by the attention mechanism
        # Value 1 (True) means "ignored" and value 0 (False) means "not ignored"
        # tgt_pad_mask.shape = [batch_size, tgt_sec_len]
        tgt_pad_mask = tgt == 0
        return tgt_mask, tgt_pad_mask
