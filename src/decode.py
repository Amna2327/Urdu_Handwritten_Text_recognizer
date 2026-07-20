import torch
import numpy as np

def ctc_greedy_decode(log_probs, Index2word, blank=117):
    """
    log_probs: (batch, time, num_classes) - your model's output, already log_softmax'd
    Index2word: dict mapping index -> character
    Returns: list of decoded strings, one per sample in the batch
    """
    pred_indices = torch.argmax(log_probs, dim=2)  # (batch, time)
    decoded_strings = []

    for sample in pred_indices:
        sample = sample.tolist()
        collapsed = []
        prev = None
        for idx in sample:
            if idx != prev:          # collapse consecutive repeats
                collapsed.append(idx)
            prev = idx
        final = [idx for idx in collapsed if idx != blank]  # strip blanks
        decoded_strings.append(''.join(Index2word[idx] for idx in final))

    return decoded_strings

def ctc_beam_search_decode(log_probs, Index2word, blank=117, beam_width=10):
    """
    log_probs: (batch, time, num_classes) - your model's log_softmax output
    Index2word: dict mapping index -> character
    beam_width: number of candidate sequences kept at each timestep
    Returns: list of decoded strings, one per sample in the batch
    """
    log_probs = log_probs.detach().cpu().numpy()
    batch_size, T, C = log_probs.shape
    decoded_strings = []

    for b in range(batch_size):
        probs = log_probs[b]  # (T, C)

        # beam: dict mapping (collapsed_sequence_tuple) -> (log_prob_blank_ending, log_prob_nonblank_ending)
        beam = {(): (0.0, -np.inf)}  # start: empty seq, log(1)=0 for blank-ending, -inf for nonblank

        for t in range(T):
            next_beam = {}
            top_classes = np.argsort(probs[t])[-beam_width:]  # prune to top-k classes per timestep, speeds this up a lot

            for seq, (p_blank, p_nonblank) in beam.items():
                p_total = np.logaddexp(p_blank, p_nonblank)

                for c in top_classes:
                    p_c = probs[t, c]

                    if c == blank:
                        # extends with blank: sequence doesn't change, just accumulate into blank-ending prob
                        nb, nnb = next_beam.get(seq, (-np.inf, -np.inf))
                        next_beam[seq] = (np.logaddexp(nb, p_total + p_c), nnb)
                    else:
                        if len(seq) > 0 and seq[-1] == c:
                            # same char repeated: extends the nonblank-ending path (no new char),
                            # but a *new* char c only appends if previous ended in blank
                            new_seq = seq
                            nb, nnb = next_beam.get(new_seq, (-np.inf, -np.inf))
                            next_beam[new_seq] = (nb, np.logaddexp(nnb, p_nonblank + p_c))

                            new_seq2 = seq + (c,)
                            nb2, nnb2 = next_beam.get(new_seq2, (-np.inf, -np.inf))
                            next_beam[new_seq2] = (nb2, np.logaddexp(nnb2, p_blank + p_c))
                        else:
                            new_seq = seq + (c,)
                            nb, nnb = next_beam.get(new_seq, (-np.inf, -np.inf))
                            next_beam[new_seq] = (nb, np.logaddexp(nnb, p_total + p_c))

            # keep only top beam_width sequences by total probability
            scored = [(seq, np.logaddexp(nb, nnb)) for seq, (nb, nnb) in next_beam.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            beam = {seq: next_beam[seq] for seq, _ in scored[:beam_width]}

        best_seq = max(beam.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]))[0]
        decoded_strings.append(''.join(Index2word[idx] for idx in best_seq))

    return decoded_strings