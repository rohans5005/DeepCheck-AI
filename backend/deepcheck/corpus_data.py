from typing import List, Dict, Any

DEFAULT_CORPUS_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "doc_id": "DOC-001",
        "title": "Introduction to Deep Neural Networks and Transformers",
        "author": "Dr. A. Vaswani et al.",
        "category": "Computer Science / AI",
        "text": """
Deep learning architectures, particularly Transformer models, have revolutionized natural language processing and computer vision.
Transformers utilize multi-head self-attention mechanisms to compute contextual representation vectors across sequence tokens.
Unlike traditional recurrent networks, transformers process sequences in parallel, allowing significant scalability on GPU accelerators.
Sentence-BERT (SBERT) fine-tunes standard BERT models using Siamese network architectures to derive semantically meaningful sentence embeddings.
By mapping semantically similar sentences close together in a high-dimensional vector space, cosine similarity can effectively measure semantic paraphrase.
Contrastive loss and triplet loss functions ensure that positive paraphrased pairs are pulled together while dissimilar sentences are pushed apart beyond a target margin.
"""
    },
    {
        "doc_id": "DOC-002",
        "title": "Principles of Climate Change and Global Warming Impact",
        "author": "Prof. E. IPCC",
        "category": "Environmental Science",
        "text": """
Global temperature rise driven by greenhouse gas emissions continues to alter planetary weather ecosystems.
Glacial retreat, sea level elevation, and thermal expansion pose catastrophic threats to coastal biodiversity and human habitats.
Renewable energy transitions involving solar, wind, and hydrothermal systems are essential to mitigate atmospheric carbon concentration.
International climate accords mandate decarbonization targets to cap rising global temperatures below two degrees Celsius above pre-industrial levels.
Severe drought cycles, forest fires, and torrential monsoons are accelerating ecological displacement across tropical and subtropical zones.
"""
    },
    {
        "doc_id": "DOC-003",
        "title": "Quantum Information Science and Superconducting Qubits",
        "author": "Dr. R. Feynman & Lab",
        "category": "Physics",
        "text": """
Quantum computing relies on quantum bits or qubits capable of existing in superpositions of 0 and 1 states simultaneously.
Entanglement enables non-local quantum state correlations between distant qubits, providing exponential computational parallelism.
Quantum error correction protocols using surface codes are required to stabilize fragile quantum memory against environmental decoherence.
Shor's algorithm demonstrates polynomial time factorization of large integers, posing fundamental challenges to classical RSA public-key encryption.
Superconducting Josephson junctions and trapped ion architectures represent leading experimental platforms for fault-tolerant quantum hardware.
"""
    },
    {
        "doc_id": "DOC-004",
        "title": "Modern Microeconomic Theory and Market Equilibrium",
        "author": "Dr. J. Keynes",
        "category": "Economics",
        "text": """
Market price determination occurs at the intersection of supply and demand curves under competitive equilibrium conditions.
Elasticity measures consumer demand sensitivity to price fluctuations across essential versus luxury commodities.
Central bank monetary policy uses interest rate adjustments to balance inflationary pressures with employment growth objectives.
Fiscal stimulus programs deploy government expenditure to revive aggregate demand during economic recessions and liquidity traps.
Corporate monopoly power restricts output and inflates consumer prices, causing deadweight welfare loss to society.
"""
    }
]
