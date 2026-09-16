# Main Reference to learn about CTC: https://distill.pub/2017/ctc/
# Additional video: https://www.youtube.com/watch?v=tzMXV2cAK04&list=PLoEMreTa9CNlQpbAJHTQSSym1Gw9mWNV3&index=4

import math
from abc import ABC, abstractmethod
import torch.nn.functional as F

NEG_INF = float("-inf")

# Reference: https://mc-stan.org/docs/stan-users-guide/floating-point.html#log-sum-of-exponentials
# Implemented instead of torch.logaddexp to avoid the overhead of tensors
def log_add(a: float, b: float) -> float:
    if a == NEG_INF:
        return b

    if b == NEG_INF:
        return a

    if a >= b:
        return a + math.log1p(math.exp(b - a))

    return b + math.log1p(math.exp(a - b))

class BeamEntry:
    """
    log_pb  : log P(paths ending in blank that decode to this prefix)
    log_pnb : log P(paths ending in non-blank that decode to this prefix)
    """
    __slots__ = ("log_pb", "log_pnb", "token_probs", "token_counts", "token_frames", "token_max_pc")

    def __init__(self):
        self.log_pb  = NEG_INF
        self.log_pnb = NEG_INF
        self.token_probs  = []
        self.token_counts = []
        self.token_frames = []
        self.token_max_pc = []

    def total(self):
        return log_add(self.log_pb, self.log_pnb)


class CTCDecoder(ABC):
    @abstractmethod
    # The ouput formatted to match the model expected output (token_ids, probs, frames)
    def __call__(self, logits, frame_start: int, frame_end: int) -> tuple[list[int], list[float], list[int]]:
        pass

class GreedyCTCDecoder(CTCDecoder):

    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id

    # Logits are in the format (time, vocab_props) 
    def __call__(self, logits, frame_start: int, frame_end: int) -> tuple[list[int], list[float], list[int]]:
        frame_end = min(frame_end, logits.shape[0])
        if frame_start >= frame_end:
            return [], [], []

        # First, convert from logits to probabilities
        probs_tensor = F.softmax(logits[frame_start:frame_end].float().cpu(), dim=-1)
        
        # We pick the most likely token at each time step
        max_probs, ids = probs_tensor.max(dim=-1)

        decoded_ids    = []
        decoded_probs  = []
        decoded_frames = []
        
        current_sum         = 0.0
        current_count       = 0
        current_max_prob    = -1.0
        current_max_frame   = -1
        prev                = None

        for i, (tid, prob) in enumerate(zip(ids.tolist(), max_probs.tolist())):
            # If new token, we collapse the previous one and avrg the probabilities if repeated
            if tid != prev:
                if prev is not None and prev != self.blank_id:
                    decoded_ids.append(prev)
                    decoded_probs.append(current_sum / current_count)
                    decoded_frames.append(current_max_frame + frame_start)
                
                current_sum         = prob
                current_count       = 1
                current_max_prob    = prob
                current_max_frame   = i
                prev                = tid
            else:
                current_sum   += prob
                current_count += 1
                if prob > current_max_prob:
                    current_max_prob  = prob
                    current_max_frame = i

        # We handle the last token
        if prev is not None and prev != self.blank_id:
            decoded_ids.append(prev)
            decoded_probs.append(current_sum / current_count)
            decoded_frames.append(current_max_frame + frame_start)

        return decoded_ids, decoded_probs, decoded_frames

# Reference: https://medium.com/@anukurian16/prefix-beam-search-5a32e673f78b (prefix beam search)
class BeamCTCDecoder(CTCDecoder):

    def __init__(self, beam_width: int = 10, blank_id: int = 0, prune_logp: float = -20.0):
        self.beam_width = beam_width
        self.blank_id   = blank_id
        self.prune_th   = prune_logp

        if self.beam_width < 1:
            raise ValueError(f"beam_width must be greater than or equal to 1, got {self.beam_width}")
        
        if self.blank_id < 0:
            raise ValueError(f"blank_id must be greater than or equal to 0, got {self.blank_id}")

    # Logits are in the format (frames, vocab_props) 
    def __call__(self, logits, frame_start: int, frame_end: int) -> tuple[list[int], list[float], list[int]]:
        
        frame_end = min(frame_end, logits.shape[0])
        if frame_start >= frame_end:
            return [], [], []

        # Convert logits to log-probabilities.
        probs     = F.softmax(logits[frame_start:frame_end].float().cpu(), dim=-1)
        log_probs = F.log_softmax(logits[frame_start:frame_end].float().cpu(), dim=-1)

        # T: number of frames, V: vocabulary size
        T, V = log_probs.shape

        # { (phoneme1, phoneme2, ...): BeamEntry}
        beams       = {}
        root        = BeamEntry()
        root.log_pb = 0.0     # log(1.0): starts with blank "emitted"
        beams[()]   = root
        lp_rows     = log_probs.tolist()
        p_rows      = probs.tolist()

        for t in range(T):
            lp = lp_rows[t]
            p  = p_rows[t]
            new_beams = {}

            for prefix, entry in beams.items():
                total = entry.total()
                last  = prefix[-1] if prefix else None

                # Case 1: extend with blank -> prefix unchanged
                e = self.find_prefix(new_beams, prefix, entry)
                e.log_pb = log_add(e.log_pb, total + lp[self.blank_id]) # P_blank_t += P_total_t-1 * P(blank)

                # Case 2 and 3: extend with each non-blank token
                for c in range(V):
                    if c == self.blank_id:
                        continue
                    lpc = lp[c]
                    pc  = p[c]

                    # Prune tokens with very low probability
                    if total + lpc < self.prune_th:
                        continue

                    if last != c:
                        # Case 2: different token -> we extend to a new prefix
                        e_ext = self.find_prefix(new_beams, prefix + (c,), entry)
                        e_ext.log_pnb = log_add(e_ext.log_pnb, total + lpc) # P_non_blank_t += P_total_t-1 * P(c)
                        
                        # If new prefix, we add token to list of props
                        if len(e_ext.token_probs) < len(prefix) + 1:
                            e_ext.token_probs.append(pc)
                            e_ext.token_counts.append(1)
                            e_ext.token_frames.append(t + frame_start)
                            e_ext.token_max_pc.append(pc)
                        else:
                            # Otherwise, we merge the path
                            e_ext.token_probs[-1] += pc
                            e_ext.token_counts[-1] += 1
                            if pc > e_ext.token_max_pc[-1]:
                                e_ext.token_max_pc[-1] = pc
                                e_ext.token_frames[-1] = t + frame_start
                    else:
                        # Case 3: same token as last

                        ## Path A: blank preceded -> we extend to a new prefix
                        e_ext = self.find_prefix(new_beams, prefix + (c,), entry)
                        e_ext.log_pnb = log_add(e_ext.log_pnb, entry.log_pb + lpc) # P_non_blank_t += P_blank_t-1 * P(c)
                        
                        # If new prefix, we add token to list of props
                        if len(e_ext.token_probs) < len(prefix) + 1:
                            e_ext.token_probs.append(pc)
                            e_ext.token_counts.append(1)
                            e_ext.token_frames.append(t + frame_start)
                            e_ext.token_max_pc.append(pc)
                        else:
                            # Otherwise, we merge the path
                            e_ext.token_probs[-1] += pc
                            e_ext.token_counts[-1] += 1
                            if pc > e_ext.token_max_pc[-1]:
                                e_ext.token_max_pc[-1] = pc
                                e_ext.token_frames[-1] = t + frame_start

                        # Path B: non-blank -> CTC collapse, same prefix, no extension
                        e_stay = self.find_prefix(new_beams, prefix, entry)
                        e_stay.log_pnb = log_add(e_stay.log_pnb, entry.log_pnb + lpc) # P_non_blank_t += P_non_blank_t-1 * P(c)
                        
                        # Merge the probs
                        if len(e_stay.token_probs) == len(prefix) and len(prefix) > 0:
                            e_stay.token_probs[-1] += pc
                            e_stay.token_counts[-1] += 1
                            if pc > e_stay.token_max_pc[-1]:
                                e_stay.token_max_pc[-1] = pc
                                e_stay.token_frames[-1] = t + frame_start

            if len(new_beams) > self.beam_width:
                # Sort descending by total log-probability and keep top beam_width
                beams = dict(sorted(new_beams.items(), key=lambda kv: kv[1].total(), reverse=True)[:self.beam_width])
            else:
                beams = new_beams

        if not beams:
            return [], [], []

        # I applied length normalization to avoid bias towards shorter prefixes as proposed here in section 2.6 https://arxiv.org/pdf/1211.3711
        def length_normalized_score(p):
            length = len(p) if len(p) > 0 else 1
            return beams[p].total() / length
 
        best_prefix = max(beams, key=length_normalized_score)
        
        # calc. the avg. token probability as in Greedy
        probs = []
        for p, c in zip(beams[best_prefix].token_probs, beams[best_prefix].token_counts):
            probs.append(p / c if c > 0 else 0.0)

        return list(best_prefix), probs, beams[best_prefix].token_frames

    def find_prefix(self, new_beams, prefix, parent_entry = None):
        if prefix not in new_beams:
            e = BeamEntry()
            
            # Copy the history of the parent prefix
            if parent_entry is not None:
                e.token_probs  = parent_entry.token_probs.copy()
                e.token_counts = parent_entry.token_counts.copy()
                e.token_frames = parent_entry.token_frames.copy()
                e.token_max_pc = parent_entry.token_max_pc.copy()

            new_beams[prefix] = e
        return new_beams[prefix]

def print_comparison(greedy_result, beam_result):
    match = greedy_result == beam_result
    
    print("\n" + "="*70)
    print("Greedy vs Beam")
    print("="*70)
    print(f"\nGreedy Decoder Output: {greedy_result} (len: {len(greedy_result)})")
    print(f"Beam Decoder Output:  {beam_result} (len: {len(beam_result)})")
    print(f"\nOutputs Match: {match}")
    
    if not match:
        print(f"\n Decoders produced different results!")
    
    print("="*70)
    return beam_result  # Return beam decoder result (higher quality)
