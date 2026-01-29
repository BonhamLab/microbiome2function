"""
Generate a realistic synthetic protein graph:
- Nodes: UniProt-like accessions with amino acid sequences and mock GO MF labels
- Edges: weighted by positive PMI of co-occurrence across synthetic metagenomic samples
- Outputs: PyG Data object (.pt) plus CSV tables for inspection
"""

from __future__ import annotations

import math
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# Fixed RNG for reproducibility
SEED = 42
random.seed(SEED)
RNG = np.random.default_rng(SEED)

# Approximate amino acid frequencies in real proteins
AA_FREQ = {
    "A": 0.078,
    "R": 0.052,
    "N": 0.044,
    "D": 0.053,
    "C": 0.015,
    "Q": 0.040,
    "E": 0.063,
    "G": 0.073,
    "H": 0.022,
    "I": 0.060,
    "L": 0.096,
    "K": 0.058,
    "M": 0.024,
    "F": 0.040,
    "P": 0.050,
    "S": 0.067,
    "T": 0.055,
    "W": 0.010,
    "Y": 0.029,
    "V": 0.068,
}
AA_LIST = list(AA_FREQ.keys())
AA_PROBS = np.array(list(AA_FREQ.values()))
AA_PROBS = AA_PROBS / AA_PROBS.sum()

# A handful of real GO MF terms to keep labels realistic
GO_TERMS = [
    "GO:0005524",  # ATP binding
    "GO:0003677",  # DNA binding
    "GO:0016787",  # hydrolase activity
    "GO:0003735",  # structural constituent of ribosome
    "GO:0000166",  # nucleotide binding
]
GO_INDEX: Dict[str, int] = {go: i for i, go in enumerate(GO_TERMS)}


def sample_accession(idx: int) -> str:
    """Create a UniProt-like accession."""
    prefix = random.choice(["A0A", "Q9", "P0", "O0"])
    middle = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    suffix = random.choice(string.ascii_uppercase + string.digits)
    return f"{prefix}{middle}{suffix}"


def sample_sequence() -> str:
    """Generate an amino acid sequence with realistic length and composition."""
    # Log-normal around ~300 aa, capped to avoid extreme outliers
    length = int(np.clip(RNG.lognormal(mean=5.65, sigma=0.35), 80, 1100))
    aa_choices = RNG.choice(AA_LIST, size=length, p=AA_PROBS)
    return "".join(aa_choices)


def amino_acid_composition(seq: str) -> np.ndarray:
    counts = np.array([seq.count(a) for a in AA_LIST], dtype=np.float32)
    return counts / len(seq)


def build_presence_matrix(
    community_ids: List[int], num_samples: int, base_prob: float = 0.03
) -> np.ndarray:
    """Simulate protein presence/absence across metagenomic samples."""
    num_proteins = len(community_ids)
    num_comms = max(community_ids) + 1
    presence = np.zeros((num_samples, num_proteins), dtype=np.int64)

    for s in range(num_samples):
        active = RNG.choice(num_comms, size=RNG.integers(1, min(num_comms, 3)), replace=False)
        for i, comm in enumerate(community_ids):
            p = base_prob
            if comm in active:
                p += 0.20 + 0.15 * RNG.random()
            presence[s, i] = RNG.random() < p

    # Ensure every protein appears at least once
    freq = presence.sum(axis=0)
    for i, f in enumerate(freq):
        if f == 0:
            presence[RNG.integers(0, num_samples), i] = 1
    return presence


def compute_pmi_edges(
    presence: np.ndarray, top_k: int = 15, eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute positive PMI edges and keep top_k per node."""
    num_samples, num_proteins = presence.shape
    freq = presence.sum(axis=0) + eps
    p_i = freq / num_samples
    co_counts = presence.T @ presence

    edge_map: Dict[Tuple[int, int], float] = {}
    for i in range(num_proteins):
        candidates: List[Tuple[int, float]] = []
        for j in range(num_proteins):
            if i == j or co_counts[i, j] == 0:
                continue
            p_ij = (co_counts[i, j] + eps) / num_samples
            pmi = math.log(p_ij / (p_i[i] * p_i[j]))
            if pmi > 0:
                candidates.append((j, pmi))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for j, pmi in candidates[:top_k]:
            a, b = sorted((i, j))
            edge_map[(a, b)] = max(edge_map.get((a, b), 0.0), pmi)

    edge_index: List[List[int]] = []
    edge_weight: List[float] = []
    for (a, b), w in edge_map.items():
        edge_index.append([a, b])
        edge_index.append([b, a])
        edge_weight.append(w)
        edge_weight.append(w)
    return np.array(edge_index, dtype=np.int64).T, np.array(edge_weight, dtype=np.float32)


def assign_labels(community_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Assign GO MF labels with community correlation and missing labels."""
    num_proteins = len(community_ids)
    labels = np.zeros((num_proteins, len(GO_TERMS)), dtype=np.float32)
    labeled_mask = np.zeros(num_proteins, dtype=bool)

    comm_label_probs = {
        0: {"GO:0005524": 0.65, "GO:0000166": 0.35},
        1: {"GO:0003677": 0.55, "GO:0000166": 0.30},
        2: {"GO:0016787": 0.60},
        3: {"GO:0003735": 0.55, "GO:0016787": 0.25},
    }

    for i, comm in enumerate(community_ids):
        labeled_mask[i] = RNG.random() < 0.8  # leave ~20% unlabeled for inference
        if not labeled_mask[i]:
            continue
        probs = comm_label_probs.get(comm, {})
        assigned = []
        for go_id, p in probs.items():
            if RNG.random() < p:
                labels[i, GO_INDEX[go_id]] = 1.0
                assigned.append(go_id)
        if not assigned and probs:
            # Guarantee at least one label when a protein is marked as labeled
            go_id = max(probs.items(), key=lambda x: x[1])[0]
            labels[i, GO_INDEX[go_id]] = 1.0
    return labels, labeled_mask


def build_graph(
    num_proteins: int = 220,
    num_samples: int = 60,
    num_communities: int = 4,
    out_dir: Path | str = Path("model/synthetic_data"),
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    community_ids = RNG.integers(0, num_communities, size=num_proteins).tolist()
    accessions = [sample_accession(i) for i in range(num_proteins)]
    sequences = [sample_sequence() for _ in range(num_proteins)]

    presence = build_presence_matrix(community_ids, num_samples)
    edge_index, edge_weight = compute_pmi_edges(presence, top_k=15)
    labels, labeled_mask = assign_labels(community_ids)

    features = []
    for seq in sequences:
        comp = amino_acid_composition(seq)
        log_len = math.log(len(seq))
        features.append(np.concatenate([comp, [log_len]]))
    x = torch.tensor(np.stack(features), dtype=torch.float32)

    y = torch.tensor(labels, dtype=torch.float32)
    train_mask = torch.zeros(num_proteins, dtype=torch.bool)
    val_mask = torch.zeros(num_proteins, dtype=torch.bool)
    test_mask = torch.zeros(num_proteins, dtype=torch.bool)
    labeled_indices = np.where(labeled_mask)[0]
    RNG.shuffle(labeled_indices)
    n_train = int(0.7 * len(labeled_indices))
    n_val = int(0.15 * len(labeled_indices))
    train_idx = labeled_indices[:n_train]
    val_idx = labeled_indices[n_train : n_train + n_val]
    test_idx = labeled_indices[n_train + n_val :]
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.accessions = accessions
    data.sequences = sequences
    data.go_terms = GO_TERMS
    data.community = torch.tensor(community_ids, dtype=torch.long)

    torch.save(data, out_dir / "synthetic_graph.pt")

    node_df = pd.DataFrame(
        {
            "accession": accessions,
            "community": community_ids,
            "sequence": sequences,
            "go_terms": [
                ";".join([GO_TERMS[j] for j, v in enumerate(row) if v > 0.5]) for row in labels
            ],
            "is_labeled": labeled_mask,
        }
    )
    edge_df = pd.DataFrame(
        {
            "src_idx": edge_index[0],
            "dst_idx": edge_index[1],
            "src_accession": [accessions[i] for i in edge_index[0]],
            "dst_accession": [accessions[i] for i in edge_index[1]],
            "pmi_weight": edge_weight,
        }
    )
    sample_records = []
    for s in range(num_samples):
        for i, acc in enumerate(accessions):
            if presence[s, i] == 1:
                sample_records.append({"sample": f"S{s:03d}", "accession": acc})
    sample_df = pd.DataFrame(sample_records)

    node_df.to_csv(out_dir / "nodes.csv", index=False)
    edge_df.to_csv(out_dir / "edges.csv", index=False)
    sample_df.to_csv(out_dir / "sample_presence.csv", index=False)

    return {
        "data_pt": out_dir / "synthetic_graph.pt",
        "nodes_csv": out_dir / "nodes.csv",
        "edges_csv": out_dir / "edges.csv",
        "presence_csv": out_dir / "sample_presence.csv",
    }


if __name__ == "__main__":
    paths = build_graph()
    print("Synthetic graph written to:")
    for k, v in paths.items():
        print(f"- {k}: {v}")
