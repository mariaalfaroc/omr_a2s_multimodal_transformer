from typing import List, Dict, Tuple

import swalign

SWALIGN_RESERVED_WORDS = ["¡", "!", "|", ".", "-", " ", "/", "-//-", "    " ]
SWALIGN_VOCAB = [chr(i).upper() for i in range(300)]
SWALIGN_VOCAB = [i for i in SWALIGN_VOCAB if i not in SWALIGN_RESERVED_WORDS]
SWALIGN_VOCAB = [i for i in SWALIGN_VOCAB if len(i) == 1]
SWALIGN_VOCAB = sorted(set(SWALIGN_VOCAB)) 
# len(SWALIGN_VOCAB) = 214; the maximum number of unique tokens in a sequence is 175

def swalign_preprocess(r: List[str], q: List[str]) -> Tuple[str, str, Dict[str, str]]:
    """
    Converts a string sequence to a compatible swalign string.
    :param r: Reference string sequence
    :param q: Query string sequence
    :return: Tuple of (reference, query, swa2w), where swa2w is a dictionary that maps swalign tokens to original tokens
    """
    current_vocab = sorted(set(r + q))
    assert len(current_vocab) < len(SWALIGN_VOCAB), f"Too many tokens for swalign! (len_current_vocab: {len(current_vocab)}, len_swalign_vocab: {len(SWALIGN_VOCAB)})"
    w2swa = dict(zip(current_vocab, SWALIGN_VOCAB))
    swa2w = dict(zip(SWALIGN_VOCAB, current_vocab))
    r = ["¡"] + [w2swa[i] for i in r] + ["!"]
    q = ["¡"] + [w2swa[i] for i in q] + ["!"]
    return "".join(r), "".join(q), swa2w


def dump(alignment: swalign.Alignment) -> Tuple[str, str, str]:
    """
    Modified version of the original dump() method for the Alignment class of the swalign library.
    We have modified it to obtain (in the following order) the query, the matches, and the reference sequences; all of them have the same length.
    Matches is a string that contains either "|" if sequences match on a token, or "." if they disagree,
    or " " if one of them misses a token (in this case the token "-" is included at such position in the corresponding sequence).
    :param alignment: Alignment object from swalign
    :return: Tuple of (query, matches, reference)
    """
    i = alignment.r_pos
    j = alignment.q_pos

    q = ""
    m = ""
    r = ""
    qlen = 0
    rlen = 0

    for count, op in alignment.cigar:
        if op == "M":
            qlen += count
            rlen += count
            for k in range(count):
                q += alignment.orig_query[j]
                r += alignment.orig_ref[i]
                if alignment.query[j] == alignment.ref[i] or (
                    alignment.wildcard
                    and (
                        alignment.query[j] in alignment.wildcard
                        or alignment.ref[i] in alignment.wildcard
                    )
                ):
                    m += "|"
                else:
                    m += "."
                i += 1
                j += 1
        elif op == "D":
            rlen += count
            for k in range(count):
                q += "-"
                r += alignment.orig_ref[i]
                m += " "
                i += 1
        elif op == "I":
            qlen += count
            for k in range(count):
                q += alignment.orig_query[j]
                r += "-"
                m += " "
                j += 1
        elif op == "N":
            q += "-//-"
            r += "-//-"
            m += "    "

    while q and r and m:
        return q, m, r


def preprocess_prob(s: str, prob: List[float]) -> List[float]:
    """
    Adapts the probability sequence after the swalign computation to be able to obtain the final alignment.
    :param s: String
    :param prob: Probability sequence
    :return: New probability sequence
    """
    new_prob = prob.copy()
    count = 0
    for id, v in enumerate(s):
        if v == "¡" or v == "!":
            new_prob.insert(id + count, 1)
            count += 1
        elif v == "-":
            new_prob.insert(id + count, 0)
            count += 1
    return new_prob


def get_alignment(
    q: str,
    m: str,
    r: str,
    q_prob: List[float],
    r_prob: List[float],
) -> str:
    """
    Obtains the final alignment string based on the fixed fusion policy:
    1) Both strings match on a token -> included
    2) strings disagree on a token -> include that of the highest probability
    3) A string misses a token -> include that of the other
    :param q: Query string
    :param m: Matches string
    :param r: Reference string
    :param q_prob: Query probability sequence
    :param r_prob: Reference probability sequence
    :return: Final alignment string
    """
    alignment = ""
    for qv, mv, rv, qv_prob, rv_prob in zip(q, m, r, q_prob, r_prob):
        # There are three possible scenarios:
        # 1) Both sequences match on a token (mv == "|", qv == rv) -> included
        # 2) Sequences disagree on a token (mv == ".", qv != rv) -> include that of the highest probability
        # 3) A sequence misses a token (mv == " ", (qv or rv) == "-")-> include that of the other
        if mv == "|":
            # Scenario 1
            assert qv == rv, f"qv and rv should be equal! (qv: {qv}, rv: {rv})"
            alignment += qv
        elif mv == ".":
            # Scenario 2
            assert qv != rv, f"qv and rv should be different! (qv: {qv}, rv: {rv})"
            alignment += qv if qv_prob >= rv_prob else rv
        elif mv == " ":
            # Scenario 3
            assert (
                qv == "-" or rv == "-"
            ), f"qv or rv should be '-'! (qv: {qv}, rv: {rv})"
            alignment += qv if rv == "-" else rv
    return alignment


def undo_swalign_preprocess(alignment: str, swa2w: Dict[str, str]) -> List[str]:
    """
    Converts a swalign alignment string to the original token vocabulary.
    :param alignment: Alignment string
    :param swa2w: Dictionary that maps swalign tokens to original tokens
    :return: Alignment as a string sequence that uses the original token vocabulary
    """
    return [swa2w[i] for i in alignment if i not in ["¡", "!"]]
